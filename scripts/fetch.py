#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 热点聚合脚本（GitHub 版 · v2 全面改版）
=========================================
只聚合 GitHub 上最热门的 AI 项目，按 AI 子领域自动分类，
调用大模型为每条生成一句中文摘要，输出 public/data.json。

v2 底层逻辑改版（调研自 GitTrend / trending-repos / apifyforge 信号栈）：
  1. 候选池：多查询扇出（多个 AI topic + 关键词 + 上升新星），去重合并，覆盖远不止一个 topic。
  2. 历史：同时记录 stars 与 forks 快照，用于计算双指标增量。
  3. 增量：各周期（当天/当月/当年）有真实历史则算真实增量；
     首日无基线则用「总星/库龄 × 周期天数」估算，使三档从第一天起数值与排序就不同。
  4. 复合动量分 TrendScore = (规模归一 star 增量 + 0.3 × 规模归一 fork 增量) × 新鲜度。
     · 规模归一：gain / ln(total+10)，让小体量暴涨库能压过巨头（GitTrend 思路）。
     · 新鲜度：随库龄指数衰减，新库最高约 1.6×，让「上升新星」冒头。
  5. 排名变化：对比上次运行的名次，给出「↑N / ↓N」，让榜单有动态感。

用法：
  python scripts/fetch.py            # 正式抓取（需联网）
  python scripts/fetch.py --demo     # 使用内置示例数据，离线预览前端 UI

环境变量（正式模式）：
  云端构建已显式禁用实时 LLM（见 build.yml LLM_API_KEY 置空），仅走缓存 + 离线词典兜底。
  摘要命中 summaries_cache.json 直接返回（免网络、必中文），未命中走离线中文兜底。
"""
import os
import sys
import re
import json
import math
import time
import copy
import argparse
import datetime
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

def env(key, default):
    """读环境变量；为空或仅空白时回退到默认值（避免空 Secret 覆盖默认配置）。"""
    v = os.environ.get(key, "")
    v = v.strip()
    return v if v else default


# ---------------- 配置 ----------------
def load_dotenv(path=".env"):
    """轻量加载 .env（不引入额外依赖）。不覆盖已存在的环境变量。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
load_dotenv()

# GitHub 抓取：候选池由多路查询扇出组成（见 QUERIES），
# 去重合并后形成「稳定头部 + 上升新星」的混合池，覆盖远不止一个 topic。
GH_PER_PAGE    = 100
GH_POOL_CAP    = 900          # 合并所有查询后的总候选池上限
GH_REQ_BUDGET  = 22           # Search API 调用次数预算（认证下 30/min，双构建足够）

KEEP_TOP_N     = 30           # 每个时间窗（当天/当月/当年）各取 Top 30

# star / fork 增量窗口（天）
DELTA_DAY   = 1
DELTA_MONTH = 30
DELTA_YEAR  = 365

# 历史快照文件（每次构建写回，用于算增量与排名变化）。不进 git，由 workflow 用 gh 上传/下载。
HISTORY_PATH = env("HISTORY_PATH", "stars_history.json")
HISTORY_KEEP_DAYS = 400       # 只保留最近 400 天快照，控制文件体积

# 摘要阶段保护：总预算 + 单次超时，保证构建绝不卡在 LLM 上
SUMMARY_DEADLINE_SEC = 18 * 60   # 到点后剩余项直接走兜底摘要
SUMMARY_TIMEOUT      = 60        # 单次 LLM HTTP 超时（秒）
SUMMARY_MAX_TOKENS   = 800       # 生成上限（仅需一句 ≤40 字摘要）

# 复合动量分权重（参考 trending-repos：star 主导，fork 辅助，新鲜度加成）
W_STAR   = 1.0
W_FORK   = 0.3
FRESH_HALF_LIFE = 150.0          # 新鲜度半衰期（天）：新库加成更高
FRESH_MAX = 0.3                  # 最新鲜库额外加成上限（即最高 1.3×，避免过度放大新库）

# 分类体系（顺序即优先级：越具体的子类越靠前，避免被宽泛类吞掉）
CATEGORIES = ["大模型 / LLM", "智能体 / Agent", "多模态 / 视觉", "检索增强 / RAG",
              "训练 / 微调", "推理 / 部署", "语音 / 音频", "框架 / 工具", "其他"]

CAT_KEYWORDS = [
    ("智能体 / Agent", ["agent", "autonomous", "multi-agent", "ai-agent", "ai-agents",
                        "agentic", "tool-use", "tooluse", "workflow", "copilot", "mcp"]),
    ("多模态 / 视觉", ["computer-vision", "vision", "diffusion", "stable-diffusion",
                      "text-to-image", "text-to-video", "image-generation", "image-gen",
                      "video", "video-generation", "multimodal", "vlm", "ocr", "segment",
                      "face", "photoreal"]),
    ("检索增强 / RAG", ["rag", "retrieval-augmented", "vector-database", "vector-db",
                       "embedding", "semantic-search", "knowledge-base", "graphrag"]),
    ("训练 / 微调", ["training", "fine-tuning", "finetune", "lora", "qlora", "rlhf",
                    "distillation", "pretraining", "deepspeed", "accelerate", "sft"]),
    ("推理 / 部署", ["inference", "serving", "deployment", "onnx", "quantization",
                    "llama-cpp", "tensorrt", "triton", "vllm", "sglang", "ort"]),
    ("语音 / 音频", ["speech", "tts", "asr", "stt", "audio", "voice", "music", "sound", "sing"]),
    ("大模型 / LLM", ["llm", "large-language-models", "gpt", "transformer", "llama",
                     "chatgpt", "nlp", "language-model", "language-models", "prompt",
                     "chatbot", "moe", "reasoning", "mixture-of-experts", "generative"]),
    ("框架 / 工具", ["framework", "library", "toolkit", "pytorch", "tensorflow", "jax",
                    "api", "sdk", "benchmark", "dataset", "data", "cli", "gui", "scraper"]),
]

PRIMARY = {
    "base":  env("LLM_BASE_URL", "https://apihub.agnes-ai.com/v1"),
    "key":   env("LLM_API_KEY", ""),
    "model": env("LLM_MODEL", "agnes-2.0-flash"),
}
FALLBACK = {
    "base":  env("FALLBACK_BASE_URL", "https://apihub.agnes-ai.com/v1"),
    "key":   env("FALLBACK_API_KEY", ""),
    "model": env("FALLBACK_MODEL", "agnes-2.0-flash"),
}


def log(*a):
    print("[fetch]", *a, file=sys.stderr, flush=True)


# ---------------- 中文摘要缓存（离线兜底，保证每张卡片都是中文） ----------------
CACHE = {}

def load_cache():
    """加载手写/历史中文摘要缓存（按仓库 full_name 索引）。命中即免网络、必中文。"""
    global CACHE
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = ["summaries_cache.json",
                  os.path.join(here, "..", "summaries_cache.json")]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    CACHE = json.load(f)
                log("cache loaded:", len(CACHE), "entries from", p)
            except Exception as e:
                log("cache load failed:", e)
            break

load_cache()

# 离线兜底翻译词典：把英文描述翻译成「以中文为主、专有名词保留英文」的摘要。
# 分三块：① 多词短语/术语 ② 名词 ③ 动词/形容词；功能词单独在 FUNCTION_WORDS 处理。
# 中文技术写作中保留 Python/PyTorch/GPT/Claude/API 等专有名词是正常且专业的，
# 因此这里只翻译普通英文词，让结果读起来是中文而非「英文里夹几个中文词」。
TERM_ZH = {
    # ---- 多词短语 / 技术术语（长优先） ----
    "machine learning": "机器学习", "deep learning": "深度学习",
    "neural network": "神经网络", "neural networks": "神经网络",
    "natural language processing": "自然语言处理",
    "large language model": "大语言模型", "large language models": "大语言模型",
    "language model": "语言模型", "language models": "语言模型",
    "reinforcement learning": "强化学习", "transfer learning": "迁移学习",
    "fine-tuning": "微调", "fine tuning": "微调", "fine-tune": "微调",
    "finetune": "微调", "pretraining": "预训练", "pretrained": "预训练",
    "pre-trained": "预训练", "vector database": "向量数据库",
    "semantic search": "语义搜索", "text-to-image": "文生图",
    "text-to-video": "文生视频", "text to image": "文生图",
    "image generation": "图像生成", "image-generation": "图像生成",
    "image-to-3d": "图生3D", "object detection": "目标检测",
    "speech recognition": "语音识别", "real-time": "实时", "real time": "实时",
    "state-of-the-art": "最先进的", "open-source": "开源", "open source": "开源",
    "high-performance": "高性能", "low-latency": "低延迟",
    "end-to-end": "端到端", "out-of-the-box": "开箱即用",
    "production-ready": "可用于生产", "cutting-edge": "前沿的",
    "generative ai": "生成式 AI", "multi-agent": "多智能体",
    "model context protocol": "模型上下文协议",
    "retrieval augmented": "检索增强", "retrieval-augmented": "检索增强",
    "diffusion model": "扩散模型", "mixture of experts": "混合专家",
    "mixture-of-experts": "混合专家", "self-hosted": "可自托管",
    "self hosted": "可自托管", "knowledge graph": "知识图谱",
    "vector store": "向量库", "code-only": "纯代码",
    "quality-gated": "质量门控", "animation-ready": "可动画",
    "token-efficient": "省 token 的", "research-first": "研究优先",
    "landing pages": "落地页", "landing page": "落地页",
    "image-to-image": "图生图", "text-to-speech": "文生语音",
    "voice cloning": "声音克隆", "speech synthesis": "语音合成",

    # ---- 名词 ----
    "neural": "神经", "network": "网络", "networks": "网络",
    "algorithm": "算法", "algorithms": "算法",
    "model": "模型", "models": "模型", "framework": "框架", "library": "库",
    "libraries": "库", "toolkit": "工具包", "tool": "工具", "tools": "工具",
    "engine": "引擎", "agent": "智能体", "agents": "智能体",
    "application": "应用", "applications": "应用", "app": "应用", "apps": "应用",
    "system": "系统", "platform": "平台", "solution": "方案", "project": "项目",
    "code": "代码", "codebase": "代码库", "dataset": "数据集", "datasets": "数据集",
    "data": "数据", "image": "图像", "images": "图像", "video": "视频",
    "text": "文本", "audio": "音频", "speech": "语音", "voice": "语音",
    "language": "语言", "chat": "对话", "chatbot": "对话机器人",
    "document": "文档", "documents": "文档", "docs": "文档",
    "file": "文件", "files": "文件", "browser": "浏览器", "memory": "记忆",
    "cache": "缓存", "database": "数据库", "pipeline": "流水线",
    "pipelines": "流水线", "graph": "图", "node": "节点", "nodes": "节点",
    "service": "服务", "module": "模块", "modules": "模块",
    "plugin": "插件", "plugins": "插件", "extension": "扩展",
    "server": "服务端", "client": "客户端", "interface": "界面", "ui": "界面",
    "api": "API", "sdk": "SDK", "cli": "命令行", "web": "Web", "bot": "机器人",
    "notebook": "笔记本", "template": "模板", "templates": "模板",
    "example": "示例", "examples": "示例", "tutorial": "教程", "tutorials": "教程",
    "course": "课程", "benchmark": "基准测试", "benchmarks": "基准测试",
    "embedding": "嵌入", "embeddings": "嵌入向量", "vector": "向量",
    "retrieval": "检索", "search": "搜索", "semantic": "语义", "rag": "检索增强",
    "transformer": "Transformer", "gpt": "GPT", "llm": "大语言模型",
    "llms": "大语言模型", "moe": "混合专家", "pytorch": "PyTorch",
    "tensorflow": "TensorFlow", "jax": "JAX", "python": "Python",
    "gpu": "GPU", "cpu": "CPU", "cuda": "CUDA", "docker": "Docker",
    "kubernetes": "Kubernetes", "prompt": "提示词", "prompts": "提示词",
    "reasoning": "推理", "inference": "推理", "serving": "服务部署",
    "deployment": "部署", "quantization": "量化", "quant": "量化",
    "lora": "LoRA", "qlora": "QLoRA", "rlhf": "RLHF", "sft": "监督微调",
    "distillation": "蒸馏", "accelerate": "加速", "acceleration": "加速",
    "distributed": "分布式", "scalable": "可扩展", "monitoring": "监控",
    "observability": "可观测性", "management": "管理", "automation": "自动化",
    "generative": "生成式", "diffusion": "扩散", "segmentation": "分割",
    "detection": "检测", "ocr": "文字识别", "photo": "照片", "photos": "照片",
    "music": "音乐", "sound": "声音", "research": "研究", "finance": "金融",
    "financial": "金融", "agentic": "智能体化", "copilot": "编程助手",
    "mcp": "模型上下文协议", "workflow": "工作流", "workflows": "工作流",
    "documentation": "文档", "wrapper": "封装", "wrappers": "封装",
    "bridge": "桥接", "port": "移植", "portable": "可移植的",
    "schema": "模式", "schemas": "模式", "config": "配置", "configs": "配置",
    "skill": "技能", "skills": "技能", "edge": "关系边", "edges": "关系边",
    "ast": "AST", "sql": "SQL", "pdf": "PDF", "html": "HTML", "pptx": "PPTX",
    "mp4": "MP4", "prototype": "原型", "prototypes": "原型", "slide": "幻灯片",
    "slides": "幻灯片", "dashboard": "仪表盘", "dashboards": "仪表盘",
    "export": "导出", "reference": "参考", "object": "物体",
    "token": "token", "tokens": "token", "developer": "开发者", "dev": "开发者",
    "harness": "harness", "plugin": "插件", "alternative": "替代方案",
    "desktop": "桌面", "app": "应用", "room": "环境",

    # ---- 动词 ----
    "build": "构建", "building": "构建", "built": "构建",
    "train": "训练", "training": "训练", "trained": "训练",
    "deploy": "部署", "deploying": "部署", "deployed": "部署",
    "create": "创建", "creating": "创建", "created": "创建",
    "make": "制作", "making": "制作", "made": "制作",
    "generate": "生成", "generating": "生成", "generated": "生成",
    "use": "使用", "using": "使用", "used": "使用",
    "run": "运行", "running": "运行", "runs": "运行",
    "support": "支持", "supports": "支持", "supported": "支持",
    "provide": "提供", "provides": "提供", "provided": "提供",
    "manage": "管理", "managing": "管理", "managed": "管理",
    "optimize": "优化", "optimizing": "优化", "optimized": "优化",
    "implement": "实现", "implementing": "实现", "implemented": "实现",
    "develop": "开发", "developing": "开发", "developed": "开发",
    "process": "处理", "processing": "处理", "processed": "处理",
    "interact": "交互", "interactive": "交互式", "explore": "探索",
    "discover": "发现", "convert": "转换", "transform": "转换",
    "translate": "翻译", "find": "查找", "detect": "检测", "recognize": "识别",
    "classify": "分类", "segment": "分割", "edit": "编辑", "render": "渲染",
    "stream": "流式", "streaming": "流式", "download": "下载",
    "install": "安装", "configure": "配置", "connect": "连接",
    "integrate": "集成", "integrating": "集成", "extend": "扩展",
    "scale": "扩展", "monitor": "监控", "visualize": "可视化",
    "visualisation": "可视化", "automate": "自动化", "test": "测试",
    "debug": "调试", "learn": "学习", "learning": "学习",
    "think": "思考", "grows": "成长", "grow": "成长", "growing": "成长",
    "turn": "把…变成", "turns": "把…变成", "rebuild": "重建",
    "cut": "削减", "cuts": "削减", "cutting": "削减", "talking": "对话",
    "explain": "解释", "explained": "可解释", "query": "查询",
    "queryable": "可查询的", "parse": "解析", "parsing": "解析",
    "deterministic": "确定性的", "local": "本地", "becomes": "成为",

    # ---- 形容词 / 副词 ----
    "easy": "易用的", "simple": "简洁的", "simplest": "最简洁的",
    "powerful": "强大的", "fast": "快速的", "faster": "更快的",
    "fastest": "最快的", "efficient": "高效的", "efficiently": "高效地",
    "lightweight": "轻量的", "minimal": "极简的", "minimalist": "极简的",
    "high": "高", "higher": "更高", "highest": "最高", "low": "低",
    "lower": "更低", "modern": "现代的", "flexible": "灵活的",
    "robust": "鲁棒的", "free": "免费的", "advanced": "先进的",
    "native": "原生的", "cross-platform": "跨平台的", "modular": "模块化的",
    "customizable": "可定制的", "extensible": "可扩展的", "beautiful": "美观的",
    "clean": "简洁的", "popular": "热门的", "official": "官方的",
    "community": "社区", "unofficial": "非官方的", "experimental": "实验性的",
    "stable": "稳定的", "unified": "统一的", "complete": "完整的",
    "fully": "完全地", "production": "生产", "ready": "就绪",
    "open": "开源", "lazy": "偷懒的", "laziest": "最懒的", "senior": "资深",
    "best": "最佳", "never": "从未", "wrote": "写过", "every": "每个",
    "few": "少量", "many": "许多", "why": "为何", "trick": "奏效", "do": "做",
    "queryable": "可查询的", "local-first": "本地优先", "real": "真实的",
    "procedural": "程序化的", "quality": "质量", "animation": "动画",
    "reference": "参考", "deterministic": "确定性的",
}

# 功能词 / 介词 / 连词（带空格边界，逐词替换；首字母大写也能匹配）
FUNCTION_WORDS = {
    " a ": "一个", " an ": "一个", " the ": "",
    " and ": " 与 ", " or ": " 或 ", " for ": " 用于 ",
    " with ": " 带 ", " to ": " 以 ", " of ": " 的 ",
    " in ": " 在 ", " on ": " 于 ", " by ": " 通过 ",
    " from ": " 来自 ", " that ": " 该 ", " this ": " 此 ",
    " your ": "你的", " you ": "你", " you'll ": "你将", " you can ": "你可",
    " it ": "它", " its ": "其", " is ": " 是 ", " are ": " 是 ",
    " be ": " 是 ", " was ": " 曾 ", " will ": " 将 ", " can ": " 可 ",
    " can't ": " 无法 ", " cannot ": " 无法 ", " them ": "它们",
    " their ": "它们的", " they ": "它们", " we ": "我们", " our ": "我们的",
    " all ": "全部", " any ": "任意", " some ": "一些", " not ": "不",
    " no ": "无", " without ": "无需", " as ": "作为", " at ": "在",
    " into ": "成为", " onto ": "到", " more ": "更多", " most ": "最",
    " less ": "更少", " than ": "于", " if ": "若", " when ": "当",
    " while ": "同时", " you've ": "你已", " you're ": "你是",
    " i ": "我", " he ": "他", " she ": "她", " his ": "他的",
    " her ": "她的", " my ": "我的", " me ": "我", " do ": "做",
    " does ": "做", " did ": "做过", " have ": "有", " has ": "有",
    " had ": "曾", " like ": "像", " so ": "因此", " but ": "但",
    " because ": "因为", " which ": "其",     " who ": "其",
}

# 补充：常见剩余英文词（动词/形容词/名词/副词），进一步降低残留英文
TERM_ZH.update({
    "everything": "一切", "makes": "让", "gives": "给", "give": "给",
    "good": "好的", "taste": "品味", "world": "世界", "world's": "世界",
    "first": "首个", "global": "全球", "intelligence": "智能", "where": "在",
    "co-evolve": "协同进化", "job": "求职", "bash": "Bash", "need": "需要",
    "original": "原创的", "meta": "元", "curated": "精选的", "list": "清单",
    "lists": "清单", "resources": "资源", "customizing": "定制",
    "control plane": "控制平面", "catalog": "目录", "discovery": "发现",
    "owned": "自有", "selection": "选择", "persistent": "持久的",
    "context": "上下文", "across": "跨", "sessions": "会话", "captures": "捕获",
    "during": "期间", "clone": "克隆", "website": "网站", "websites": "网站",
    "command": "命令", "live": "实时", "runtime": "运行时", "stops": "停止",
    "never": "从不", "coding": "编程", "brings": "带来", "power": "能力",
    "powers": "驱动", "directly": "直接", "terminal": "终端",
    "accessible": "可访问", "scrape": "抓取", "interact": "交互",
    "extended": "扩展的", "long-horizon": "长程", "superagent": "超级智能体",
    "researches": "研究", "financial": "金融", "trading": "交易", "multi": "多",
    "powered": "驱动", "stock": "股票", "analysis": "分析",
    "multi-market": "多市场", "market": "市场", "news": "新闻",
    "decision": "决策", "dashboard": "看板", "push": "推送", "auto": "自动",
    "zero-cost": "零成本", "scheduled": "定时", "runs": "运行", "native": "原生",
    "personal": "个人", "webui": "Web 界面", "compress": "压缩", "outputs": "输出",
    "logs": "日志", "chunks": "分块", "before": "在…之前", "reach": "到达",
    "reduces": "降低", "consumption": "消耗", "common": "常见", "proxy": "代理",
    "fleet": "集群", "parallel": "并行", "working": "工作", "omni": "全能",
    "route": "路由", "monitor": "监控", "harness": "框架",
    "battle-tested": "久经考验", "best-benchmarked": "基准表现最佳",
    "world's first": "世界首个", "good taste": "好品味", "nano": "轻量",
    "curated list": "精选清单", "control-plane": "控制平面",
    "everything's": "一切", "ade": "ADE",
})

# 已知专有名词 / 缩写：翻译时保留英文（中文技术写作中保留这些是正常的）
PROPER_NOUNS = {
    "ai", "agi", "api", "sdk", "cli", "web", "sql", "pdf", "html", "pptx", "mp4",
    "llm", "moe", "rag", "gpt", "ast", "byok", "cuda", "cpu", "gpu", "docker",
    "kubernetes", "pytorch", "tensorflow", "jax", "claude", "codex", "cursor",
    "gemini", "openai", "anthropic", "meta", "google", "alibaba", "xiaomi",
    "huggingface", "deepseek", "three.js", "github", "orca", "mcp", "sft",
    "rlhf", "lora", "qlora", "tts", "asr", "ocr", "vlm", "ml", "dl", "nlp",
    "cv", "ide", "ui",
    # 编程语言 / 运行时（中文技术文保留英文是常态）
    "token", "python", "bash", "javascript", "typescript", "rust", "golang",
    "go", "java", "node", "nodejs", "c++", "cpp", "ruby", "php", "scala",
    "kotlin", "swift", "vue", "react", "angular", "linux", "windows", "macos",
}

# 单 token 功能词（无空格边界，逐词翻译；与 TERM_ZH 单词条合并成查表）
FUNCTION_TOK = {
    "a": "一个", "an": "一个", "the": "", "and": "与", "or": "或", "for": "用于",
    "with": "带", "to": "以", "of": "的", "in": "在", "on": "于", "by": "通过",
    "from": "来自", "that": "该", "this": "此", "your": "你的", "you": "你",
    "you'll": "你将", "you can": "你可", "it": "它", "its": "其", "is": "是",
    "are": "是", "be": "是", "was": "曾", "will": "将", "can": "可",
    "can't": "无法", "cannot": "无法", "them": "它们", "their": "它们的",
    "they": "它们", "we": "我们", "our": "我们的", "all": "全部", "any": "任意",
    "some": "一些", "not": "不", "no": "无", "without": "无需", "as": "作为",
    "at": "在", "into": "成为", "onto": "到", "more": "更多", "most": "最",
    "less": "更少", "than": "于", "if": "若", "when": "当", "while": "同时",
    "you've": "你已", "you're": "你是", "i": "我", "he": "他", "she": "她",
    "his": "他的", "her": "她的", "my": "我的", "me": "我", "do": "做",
    "does": "做", "did": "做过", "have": "有", "has": "有", "had": "曾",
    "like": "像", "so": "因此", "but": "但", "because": "因为",
    "which": "其", "who": "其",
}

# 合并单 token 翻译表：TERM_ZH 中无空格的条目 + 功能词
_TOKEN_ZH = {}
for _k, _v in TERM_ZH.items():
    if " " not in _k:
        _TOKEN_ZH[_k.lower()] = _v
_TOKEN_ZH.update(FUNCTION_TOK)


def _cjk_count(s):
    return len(re.findall(r'[一-鿿]', s))


def _cjk_ratio(s):
    if not s:
        return 0
    cjk = _cjk_count(s)
    alpha = sum(1 for ch in s if ch.isalpha() and ord(ch) < 128)
    return cjk / alpha if alpha else 1.0


def _strip_emoji(s):
    """去掉开头的装饰性 emoji / 符号，中文简介更干净。"""
    return re.sub(r'^[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\s]+', '', s).strip()


def offline_zh_summary(it):
    """LLM / 缓存都不可用时，用大词典把英文描述翻译为「以中文为主、专有名词保留英文」的摘要。

    做法：先替换含空格的多词短语，再把句子拆成 token 逐词翻译（避免英文词与中文粘连
    导致漏译），未知英文词直接丢弃（已知专有名词 Claude/GPU/API 与数字保留），
    从而彻底消除旧版「英文里夹几个中文词」的半翻译问题，中文模式读起来是完整中文。
    """
    title = it["title"]
    if title in CACHE:
        return CACHE[title]
    desc = (it.get("description") or "").strip()
    if not desc:
        return "（暂无简介）"
    seg = re.split(r'(?<=[.!?])\s', desc)[0]   # 取第一句
    # 1) 先替换含空格的多词短语（术语），避免被拆成单 token 漏译
    for term in sorted((k for k in TERM_ZH if " " in k), key=len, reverse=True):
        if re.search(r'(?i)\b' + re.escape(term) + r'\b', seg):
            seg = re.sub(r'(?i)\b' + re.escape(term) + r'\b', TERM_ZH[term], seg)
    # 2) 拆成 token（英文词 / 中文串 / 中文标点），逐 token 翻译
    toks = re.findall(r"[A-Za-z0-9_.+\-]+|[一-鿿]+|[，。、：；！？]", seg)
    out = []
    for w in toks:
        if re.fullmatch(r"[A-Za-z0-9_.+\-]+", w):
            wl = w.lower()
            if wl in _TOKEN_ZH:
                out.append(_TOKEN_ZH[wl])
            elif wl in PROPER_NOUNS or wl.rstrip("s") in PROPER_NOUNS \
                    or re.fullmatch(r'[0-9][0-9%.+\-]*', w):
                out.append(w)          # 已知专有名词 / 数字 → 保留
            # 其余未知英文词 → 丢弃，消除半英文
        else:
            out.append(w)              # 中文 / 标点 → 保留
    seg = "".join(out)
    # 3) 清理多余空格、中文标点前的空格，去开头 emoji
    seg = re.sub(r'\s+', ' ', seg).strip()
    seg = re.sub(r'\s+([，。、：；！？])', r'\1', seg)
    seg = _strip_emoji(seg)
    # 4) 翻译太稀疏 → 退回分类式中文简介
    if _cjk_count(seg) < 2:
        cat = it.get("category") or "其他"
        name = title.split("/")[-1]
        return f"{name}：一个 {cat} 方向的 GitHub 开源 AI 项目。"
    if len(seg) > 80:
        seg = seg[:80] + "…"
    return "（简介）" + seg



# ---------------- 分类 ----------------
def classify(repo):
    """依据 GitHub topics + 描述，把仓库归入某个 AI 子领域。确定性映射，稳定可筛。"""
    topics = [t.lower() for t in (repo.get("topics") or [])]
    desc = (repo.get("description") or "").lower()
    text = " ".join(topics) + " | " + desc
    for cat, kws in CAT_KEYWORDS:
        for kw in kws:
            if kw in text:
                return cat
    return "其他"


# ---------------- 抓取：GitHub（多查询扇出） ----------------
def _gh_token():
    return env("GH_TOKEN", "") or env("GITHUB_TOKEN", "")


def _gh_search_once(q, page, token):
    """对单个 GitHub Search 查询翻一页，返回原始 repo 列表（失败返回空）。"""
    url = ("https://api.github.com/search/repositories?q=" + urllib.parse.quote(q) +
           f"&sort=stars&order=desc&per_page={GH_PER_PAGE}&page={page}")
    headers = {"User-Agent": "ai-hotspot", "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        return data.get("items", [])
    except Exception as e:
        log("github search failed:", q[:60], "page", page, "-", e)
        return []


def _normalize_repo(repo):
    stars = repo.get("stargazers_count", 0) or 0
    return {
        "source": "github",
        "category": classify(repo),
        "title": repo.get("full_name", ""),
        "url": repo.get("html_url", ""),
        "description": (repo.get("description") or "")[:600],
        "published": (repo.get("pushed_at") or "")[:10],
        "raw_metric": float(stars),
        "extra": {"stars": stars,
                  "forks": repo.get("forks_count", 0) or 0,
                  "language": repo.get("language"),
                  "topics": repo.get("topics", []),
                  "created_at": (repo.get("created_at") or "")[:10],
                  "pushed_at": (repo.get("pushed_at") or "")[:10]},
    }


# 候选池 = 「多 AI topic + 关键词 + 上升新星」扇出，去重合并。
# 不同查询覆盖不同语义（llm / agent / diffusion / rag / cv / mlops …），避免只抓一个 topic 漏掉大量 AI 库。
def _build_queries():
    today = datetime.date.today()
    recent_created = (today - datetime.timedelta(days=730)).isoformat()   # 近 2 年新建
    recent_pushed = (today - datetime.timedelta(days=60)).isoformat()    # 近 60 天活跃
    return [
        # 稳定头部（宽泛 AI 领域，按总 star）
        ("topic:machine-learning stars:>200", 2),
        ("topic:deep-learning stars:>200", 1),
        ("topic:computer-vision stars:>100", 1),
        # 具体子领域（高信号，抓得起量就抓）
        ("topic:llm stars:>100", 1),
        ("topic:large-language-models stars:>50", 1),
        ("topic:agent stars:>100", 1),
        ("topic:ai-agents stars:>50", 1),
        ("topic:stable-diffusion stars:>50", 1),
        ("topic:diffusion stars:>50", 1),
        ("topic:rag stars:>50", 1),
        ("topic:mlops stars:>50", 1),
        # 关键词（捕捉没打 topic 标签、但名字/描述里带 ai/llm/agent 的库）
        ("ai in:name,description stars:>300", 1),
        ("llm in:name,description stars:>300", 1),
        ("agent in:name,description stars:>300", 1),
        # 上升新星（新建即爆红 / 近期活跃的高 star 库）
        (f"topic:machine-learning stars:>300 created:>{recent_created}", 1),
        (f"topic:machine-learning stars:>300 pushed:>={recent_pushed}", 1),
    ]


def fetch_github():
    """多查询扇出抓取候选池；受 API 预算与池上限约束，去重合并。"""
    token = _gh_token()
    queries = _build_queries()
    budget = GH_REQ_BUDGET
    items, seen = [], set()
    for q, pages in queries:
        if budget <= 0:
            break
        for page in range(1, pages + 1):
            if budget <= 0:
                break
            for repo in _gh_search_once(q, page, token):
                name = repo.get("full_name", "")
                if not name or name in seen:
                    continue
                seen.add(name)
                items.append(_normalize_repo(repo))
            budget -= 1
            if len(items) >= GH_POOL_CAP:
                break
            time.sleep(1)   # 放慢，避免触发 GitHub 搜索速率限制
        if len(items) >= GH_POOL_CAP:
            break
    items = items[:GH_POOL_CAP]
    log(f"github: {len(items)} items (multi-query pool incl. rising stars)")
    return items


# ---------------- 历史（stars + forks 快照 + 上次排名） ----------------
# history 结构：
# {
#   "snapshots": { "YYYY-MM-DD": { full_name: {"s": stars, "f": forks} }, ... },
#   "prev_ranks": { "day": {full_name: rank}, "month": {...}, "year": {...} }   # 上次运行写入
# }
def load_history(path):
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            # 兼容旧 schema：{ "YYYY-MM-DD": { full_name: stars } } → 新 schema
            if isinstance(d, dict) and "snapshots" not in d and "prev_ranks" not in d:
                snaps = {}
                for date_key, repos in d.items():
                    if not isinstance(repos, dict):
                        continue
                    # 旧 schema 只存了 star 数，fork 历史未知 → 仅迁移 "s"，fork 回退到估算
                    snaps[date_key] = {
                        fn: {"s": int(v) if isinstance(v, (int, float)) else 0}
                        for fn, v in repos.items() if isinstance(fn, str)
                    }
                d = {"snapshots": snaps, "prev_ranks": {}}
                log("migrated old star history schema:", len(snaps), "days")
            d.setdefault("snapshots", {})
            d.setdefault("prev_ranks", {})
            return d
    except Exception as e:
        log("history load failed:", e)
    return {"snapshots": {}, "prev_ranks": {}}


def upload_history_to_repo(path):
    """用 gh CLI 把本地 history 文件 PUT 回仓库（需 GH_TOKEN 且仓库可写）。失败仅告警。"""
    import subprocess, base64
    gh = os.environ.get("GH_CLI") or "gh"
    repo = os.environ.get("GITHUB_REPOSITORY") or "ao2176233569-ai/ai-hotspot"
    api = f"/repos/{repo}/contents/stars_history.json"
    try:
        sha = None
        p = subprocess.run([gh, "api", api, "--jq", ".sha"],
                           capture_output=True, text=True, timeout=30)
        if p.returncode == 0 and p.stdout.strip():
            sha = p.stdout.strip()
        with open(path, "rb") as f:
            content = base64.b64encode(f.read()).decode("ascii")
        cmd = [gh, "api", "--method", "PUT", api,
               "-f", f"message=chore: update star history ({datetime.date.today().isoformat()})",
               "-f", f"content={content}"]
        if sha:
            cmd += ["-f", f"sha={sha}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            log("history uploaded to repo")
        else:
            log("history upload failed:", r.stderr[:200])
    except Exception as e:
        log("history upload error:", e)


def save_history(history, path, today, current_map):
    """把今天的快照写入 history（保留最近 HISTORY_KEEP_DAYS 天），并视情况上传到仓库。"""
    history["snapshots"][today] = current_map
    dates = sorted(history["snapshots"].keys())
    if len(dates) > HISTORY_KEEP_DAYS:
        for d in dates[:-HISTORY_KEEP_DAYS]:
            history["snapshots"].pop(d, None)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=1, sort_keys=True)
    except Exception as e:
        log("history write local failed:", e)
    if os.environ.get("UPLOAD_HISTORY") == "1":
        upload_history_to_repo(path)


def delta_for(snapshots, full_name, current_stars, days_back):
    """返回 days_back 天前的 star 增量（current - 历史快照）。找不到更早快照返回 None。"""
    today = datetime.date.today()
    target = (today - datetime.timedelta(days=days_back)).isoformat()
    best = None
    for d in snapshots:
        if d <= target and (best is None or d > best):
            best = d
    if best is None:
        return None
    past = snapshots[best].get(full_name)
    if not isinstance(past, dict) or "s" not in past:
        return None
    return max(0, current_stars - past["s"])


def fork_delta_for(snapshots, full_name, current_forks, days_back):
    today = datetime.date.today()
    target = (today - datetime.timedelta(days=days_back)).isoformat()
    best = None
    for d in snapshots:
        if d <= target and (best is None or d > best):
            best = d
    if best is None:
        return None
    past = snapshots[best].get(full_name)
    # "f" 缺失或为 0（旧历史/迁移 artifact）→ 视为未知，回退到估算，避免把全部 fork 数当增量
    if not isinstance(past, dict) or "f" not in past or not past.get("f"):
        return None
    return max(0, current_forks - past["f"])


def compute_deltas(items, history):
    """计算各周期 star/fork 增量；有真实历史则取真实值，否则用库龄估算（标记 estimated）。"""
    snapshots = history.get("snapshots", {})
    today = datetime.date.today()
    for it in items:
        s = it["extra"]["stars"]
        f = it["extra"]["forks"]
        ca = (it["extra"].get("created_at") or "")[:10]
        try:
            age = max(1, (today - datetime.date.fromisoformat(ca)).days)
        except Exception:
            age = 3650
        it["extra"]["age_days"] = age
        it["deltas"] = {}
        for period, win in (("day", DELTA_DAY), ("month", DELTA_MONTH), ("year", DELTA_YEAR)):
            d_star = delta_for(snapshots, it["title"], s, win)
            d_fork = fork_delta_for(snapshots, it["title"], f, win)
            if d_star is not None and d_fork is not None:
                it["deltas"][period] = {"stars": d_star, "forks": d_fork, "est": False}
            else:
                # 首日无基线：用「总量 / 库龄 × 周期天数」估算，并封顶为总量（新库不会超过自身历史）
                est_star = min(int(s / age * win), s)
                est_fork = min(int(f / age * win), f) if f else 0
                it["deltas"][period] = {"stars": est_star, "forks": est_fork, "est": True}


def _freshness(age_days):
    """新鲜度乘数：新库加成高、老库趋近 1。"""
    return 1.0 + FRESH_MAX * math.exp(-age_days / FRESH_HALF_LIFE)


def _safe_log(x):
    return math.log(max(x, 1) + 10)   # ln(total+10)，避免 log(0)


def add_scores(items):
    """复合动量分 TrendScore = (规模归一 star 增量 + 0.3×规模归一 fork 增量) × 新鲜度。
    规模归一用 gain/ln(total+10)，让小体量暴涨库也能压过巨头（GitTrend 思路）。"""
    for it in items:
        s = it["extra"]["stars"]
        f = it["extra"]["forks"]
        age = it["extra"].get("age_days", 3650)
        fresh = _freshness(age)
        size_s = _safe_log(s)
        size_f = _safe_log(f if f else 1)
        it["scores"] = {}
        for period in ("day", "month", "year"):
            d = it["deltas"][period]
            m_star = d["stars"] / size_s
            m_fork = (d["forks"] / size_f) if f else 0
            it["scores"][period] = round((W_STAR * m_star + W_FORK * m_fork) * fresh, 4)
        it["freshness"] = round(fresh, 3)


def score_all(items):
    """给每个 item 算 log 热度分（0~1），用于「按 Star 总量」排序选项的辅助指标。"""
    if not items:
        return
    mx = max((it["raw_metric"] for it in items), default=1) or 1
    log_mx = math.log10(mx + 1)
    for it in items:
        it["heat_score"] = round(math.log10(it["raw_metric"] + 1) / log_mx, 4) if log_mx > 0 else 0


def compute_rank_changes(items, history):
    """对比上次运行的名次，给出排名变化（正值=上升）。首次运行无上次名次→None。"""
    prev = history.get("prev_ranks", {})
    # items 已分组进 ranges 后再调本函数；这里接收「按 period 排好序的列表」更合适。
    # 为简化：在 build_range 内完成 rank_change 计算。
    return prev


def build_range(items, period):
    """按某周期的复合动量分降序取 Top N；同时对比上次名次填充 rank_change。"""
    prev = None  # 由调用方注入
    def sort_key(it):
        sc = it["scores"].get(period, 0)
        return sc
    lst = sorted(items, key=sort_key, reverse=True)[:KEEP_TOP_N]
    out = []
    for i, it in enumerate(lst, 1):
        c = copy.deepcopy(it)
        c["rank"] = i
        out.append(c)
    return out


def build_ranges(items, history):
    score_all(items)
    compute_deltas(items, history)
    add_scores(items)
    ranges = {
        "day": build_range(items, "day"),
        "month": build_range(items, "month"),
        "year": build_range(items, "year"),
    }
    # 排名变化：用本次名次 vs 上次运行名次
    prev = history.get("prev_ranks", {})
    new_prev = {}
    for period in ("day", "month", "year"):
        cur_map = {it["title"]: it["rank"] for it in ranges[period]}
        new_prev[period] = cur_map
        prev_map = prev.get(period, {})
        for it in ranges[period]:
            p = prev_map.get(it["title"])
            it["rank_change"] = (p - it["rank"]) if p is not None else None
    # 写回本次名次，供下次对比
    history["prev_ranks"] = new_prev
    # is_new：以库龄为准（新建 <90 天视为新星），避免首日 prev_ranks 为空时全员误标
    for period in ranges:
        for it in ranges[period]:
            it["is_new"] = it["extra"].get("age_days", 9999) < 90
    return ranges


# ---------------- 大模型摘要 ----------------
SUMMARY_SYS = ("你是一个 AI 领域编辑。请用一句简洁的中文（专业术语保留英文）"
               "概括下面 GitHub 项目的核心用途或亮点，不超过 40 字，不要使用引号或编号。")


def call_llm(system, user):
    """调用主/备模型生成摘要。单次尝试、短超时；失败返回 None 由上层兜底。"""
    cfgs = [c for c in (PRIMARY, FALLBACK) if c["key"]]
    if not cfgs:
        return None
    payload = {
        "model": None,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.3,
        "max_tokens": SUMMARY_MAX_TOKENS,
    }
    for cfg in cfgs:  # PRIMARY -> FALLBACK，各最多尝试一次
        payload["model"] = cfg["model"]
        try:
            import requests
            r = requests.post(
                cfg["base"].rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + cfg["key"],
                          "Content-Type": "application/json"},
                json=payload, timeout=SUMMARY_TIMEOUT)
            if r.status_code == 200:
                msg = r.json()["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                if content:
                    return content
                log("llm empty content, next cfg")  # 推理模型额度占满，转下一配置
                continue
            if r.status_code == 429:
                time.sleep(3)
                continue
            log("llm err", r.status_code, r.text[:200])
        except Exception as e:
            log("llm call failed:", e)
        time.sleep(1)
    return None


def fallback_summary(it):
    """LLM / 缓存都不可用时的确定性兜底：返回离线中文摘要，保证卡片有中文内容。"""
    return offline_zh_summary(it)


def summarize(it, deadline):
    """单条摘要：命中缓存直接返回（免网络、必中文）；否则在截止前试一次 LLM。"""
    title = it["title"]
    if title in CACHE:
        return CACHE[title]
    if time.time() > deadline:
        return None
    user = f"项目：{title}\n描述：{(it.get('description') or '')[:800]}"
    return call_llm(SUMMARY_SYS, user)


def summarize_all(items, workers=6):
    """并发摘要 + 全局截止时间保护：到点/失败均走兜底，构建永不卡在 LLM 上。"""
    if not items:
        return
    start = time.time()
    deadline = start + SUMMARY_DEADLINE_SEC
    total = len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(summarize, it, deadline): it for it in items}
        for f in as_completed(futs):
            it = futs[f]
            try:
                s = f.result()
            except Exception:
                s = None
            it["summary"] = (s.strip() if s else fallback_summary(it))
            # 英文简介：默认用 GitHub 原始英文描述（无需 LLM，云端构建即可生效）；
            # 若原始描述为空则退回中文摘要，保证始终有内容可切。
            en = (it.get("description") or "").strip()
            it["summary_en"] = en or it["summary"]
            done += 1
            if done % 10 == 0 or done == total:
                log(f"summary {done}/{total}")


# ---------------- 示例数据（离线预览） ----------------
def DEMO_ITEMS():
    """内置示例，含 stars/forks/created_at，使 v2 打分引擎在预览中也能体现真实差异。"""
    today = datetime.date.today().isoformat()
    recent = (datetime.date.today() - datetime.timedelta(days=14)).isoformat()
    old = (datetime.date.today() - datetime.timedelta(days=1200)).isoformat()
    base = [
        ("deepseek-ai/DeepSeek-V3", "大模型 / LLM", 95000, 4200,
         "Official implementation of DeepSeek-V3, a strong Mixture-of-Experts language model with 671B total parameters.",
         "DeepSeek-V3 官方实现，671B 参数的 MoE 旗舰语言模型。",
         ["llm", "moe", "pytorch"], old, 320, 5200, 95000, 180, 2600, 42000),
        ("vllm-project/vllm", "推理 / 部署", 38000, 3600,
         "A high-throughput and memory-efficient inference and serving engine for LLMs.",
         "高吞吐、省显存的大模型推理与服务引擎。",
         ["llm", "inference", "serving"], old, 85, 1800, 38000, 60, 1200, 36000),
        ("comfyanonymous/ComfyUI", "多模态 / 视觉", 72000, 4100,
         "The most powerful and modular diffusion model GUI, API and backend with a graph/node based interface.",
         "基于节点图的最强模块化扩散模型可视化与推理后端。",
         ["diffusion", "stable-diffusion", "image-generation"], old, 150, 2400, 72000, 120, 2000, 4100),
        ("FareedKhan-dev/kimi-k3-in-c", "大模型 / LLM", 5701, 924,
         "Production-ready Kimi K3 inference in pure C with zero dependencies, blazing fast on CPU.",
         "零依赖纯 C 实现的 Kimi K3 推理，CPU 上极速。",
         ["llm", "c", "inference"], recent, 407*1, 407*30, 407*120, 70, 520, 850),
        ("langchain-ai/langchain", "智能体 / Agent", 95000, 16000,
         "Build context-aware reasoning applications with LLMs. Frameworks for agents, RAG and orchestration.",
         "面向 LLM 应用的开发框架，内置 Agent、RAG 与编排能力。",
         ["llm", "agents", "framework"], old, 40, 800, 95000, 30, 600, 16000),
        ("run-llama/llama_index", "检索增强 / RAG", 38000, 4200,
         "LlamaIndex is a data framework for your LLM applications to ingest, structure and retrieve private data.",
         "面向 LLM 的数据框架，专注私有数据的检索增强（RAG）。",
         ["rag", "llm", "retrieval"], old, 60, 900, 38000, 45, 700, 4200),
        ("openai/whisper", "语音 / 音频", 75000, 9000,
         "Robust speech recognition via large-scale weak supervision. Approach to multilingual ASR and translation.",
         "OpenAI 开源的强鲁棒性多语种语音识别（ASR）模型。",
         ["speech", "asr", "audio"], old, 20, 400, 75000, 15, 300, 9000),
        ("hiyouga/LLaMA-Factory", "训练 / 微调", 42000, 4900,
         "Easy and efficient LLM fine-tuning with LoRA, QLoRA and RLHF. Supports hundreds of models.",
         "易用的 LLM 微调框架，支持 LoRA / QLoRA / RLHF。",
         ["llm", "lora", "fine-tuning"], old, 110, 2600, 42000, 80, 1900, 4900),
        ("huggingface/transformers", "框架 / 工具", 140000, 28000,
         "State-of-the-art Machine Learning for PyTorch, JAX and TensorFlow. Thousands of pretrained models.",
         "最流行的深度学习模型库，集成数千个预训练模型。",
         ["pytorch", "transformers", "nlp"], old, 95, 1500, 140000, 70, 1100, 28000),
        ("facebookresearch/segment-anything", "多模态 / 视觉", 48000, 5200,
         "The Segment Anything Model (SAM): a foundation model for image segmentation with promptable masks.",
         "Meta 的 SAM：可提示驱动的通用图像分割基础模型。",
         ["computer-vision", "segmentation", "vision"], old, 15, 300, 48000, 12, 240, 5200),
        ("microsoft/autogen", "智能体 / Agent", 40000, 6100,
         "A framework that enables development of LLM applications using multiple agents that can converse.",
         "微软开源的多智能体对话框架，用于编排 LLM 应用。",
         ["llm", "multi-agent", "framework"], old, 70, 1100, 40000, 50, 800, 6100),
    ]
    items = []
    for (title, cat, stars, forks, desc, summ, topics, created, d_day, d_month, d_year,
         f_day, f_month, f_year) in base:
        it = {
            "source": "github", "category": cat, "title": title,
            "url": "https://github.com/" + title,
            "description": desc, "published": today, "raw_metric": float(stars),
            "summary": summ, "summary_en": desc,
            "extra": {"stars": stars, "forks": forks, "language": "Python",
                      "topics": topics, "created_at": created, "pushed_at": today},
            "deltas": {
                "day":   {"stars": d_day,   "forks": f_day,   "est": False},
                "month": {"stars": d_month, "forks": f_month, "est": False},
                "year":  {"stars": d_year,  "forks": f_year,  "est": False},
            },
        }
        items.append(it)
    return items


def DEMO_HISTORY():
    """预览模式用 DEMO_ITEMS 的增量反推快照，使三榜单在本地预览里有真实差异。"""
    today = datetime.date.today()
    d1 = (today - datetime.timedelta(days=1)).isoformat()
    d30 = (today - datetime.timedelta(days=30)).isoformat()
    d365 = (today - datetime.timedelta(days=365)).isoformat()
    h = {"snapshots": {d1: {}, d30: {}, d365: {}}, "prev_ranks": {}}
    for it in DEMO_ITEMS():
        s = it["extra"]["stars"]; f = it["extra"]["forks"]
        h["snapshots"][d1][it["title"]]   = {"s": s - it["deltas"]["day"]["stars"],   "f": f - it["deltas"]["day"]["forks"]}
        h["snapshots"][d30][it["title"]]  = {"s": s - it["deltas"]["month"]["stars"], "f": f - it["deltas"]["month"]["forks"]}
        h["snapshots"][d365][it["title"]] = {"s": s - it["deltas"]["year"]["stars"],  "f": f - it["deltas"]["year"]["forks"]}
    return h


# ---------------- 主流程 ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="使用内置示例数据，离线预览")
    args = p.parse_args()

    os.makedirs("public", exist_ok=True)

    if args.demo:
        items = DEMO_ITEMS()
        history = DEMO_HISTORY()      # 内置快照，让三榜单在预览里有真实差异
    else:
        items = fetch_github()
        history = load_history(HISTORY_PATH)
        summarize_all(items, workers=6)
        log("all summaries done")
        today = datetime.date.today().isoformat()
        current_map = {it["title"]: {"s": it["extra"]["stars"], "f": it["extra"]["forks"]} for it in items}
        save_history(history, HISTORY_PATH, today, current_map)

    ranges = build_ranges(items, history)

    if args.demo:
        for rng in ranges.values():
            for it in rng:
                it.setdefault("summary", "")

    out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": "github",
        "mode": "trend-score",
        "categories": CATEGORIES,
        "ranges": ranges,
        "count": {k: len(v) for k, v in ranges.items()},
    }
    with open("public/data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"wrote public/data.json: day={len(ranges['day'])} month={len(ranges['month'])} year={len(ranges['year'])}")


if __name__ == "__main__":
    main()
