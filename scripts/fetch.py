#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 热点聚合脚本（GitHub 版）
==========================
只聚合 GitHub 上最热门的 AI 项目，按 AI 子领域自动分类，
调用大模型为每条生成一句中文摘要，输出 public/data.json。

用法：
  python scripts/fetch.py            # 正式抓取（需联网 + LLM_API_KEY）
  python scripts/fetch.py --demo     # 使用内置示例数据，离线预览前端 UI

环境变量（正式模式）：
  LLM_API_KEY     主用模型 Key（默认平台 agnes，https://apihub.agnes-ai.com/v1）
  LLM_BASE_URL    主用模型 Base URL（默认 https://apihub.agnes-ai.com/v1）
  LLM_MODEL       主用模型名（默认 agnes-2.0-flash）
  FALLBACK_API_KEY / FALLBACK_BASE_URL / FALLBACK_MODEL  备用模型（可同平台不同 Key）
未设置任何 Key 时，摘要由离线中文词典兜底，脚本仍会产出全中文 data.json。
本地可用 .env 文件（参考 .env.example）放置以上变量，脚本启动时会自动加载。
"""
import os
import sys
import re
import json
import math
import time
import argparse
import datetime
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# GitHub 抓取：围绕一个宽泛的 AI topic，取近期活跃且 star 较高的项目
GH_TOPIC       = "machine-learning"
GH_PER_PAGE    = 30
GH_MIN_STARS   = 100
GH_RECENT_DAYS = 30

KEEP_TOP_N     = 30

# 摘要阶段保护：总预算 + 单次超时，保证构建绝不卡在 LLM 上
SUMMARY_DEADLINE_SEC = 18 * 60   # 到点后剩余项直接走兜底摘要
SUMMARY_TIMEOUT      = 60        # 单次 LLM HTTP 超时（秒）
SUMMARY_MAX_TOKENS   = 800       # 生成上限（仅需一句 ≤40 字摘要）

# 分类体系（顺序即优先级：越具体的子类越靠前，避免被宽泛类吞掉）
CATEGORIES = ["大模型 / LLM", "智能体 / Agent", "多模态 / 视觉", "检索增强 / RAG",
              "训练 / 微调", "推理 / 部署", "语音 / 音频", "框架 / 工具", "其他"]

CAT_KEYWORDS = [
    ("智能体 / Agent", ["agent", "autonomous", "multi-agent", "ai-agent", "workflow", "tool-use"]),
    ("多模态 / 视觉", ["computer-vision", "vision", "diffusion", "stable-diffusion",
                      "text-to-image", "image-generation", "video", "video-generation",
                      "multimodal", "vlm", "ocr", "segment"]),
    ("检索增强 / RAG", ["rag", "retrieval-augmented", "vector-database",
                       "embedding", "semantic-search", "knowledge-base"]),
    ("训练 / 微调", ["training", "fine-tuning", "finetune", "lora", "qlora", "rlhf",
                    "distillation", "pretraining", "deepseed", "deepspeed"]),
    ("推理 / 部署", ["inference", "serving", "deployment", "onnx", "quantization",
                    "llama-cpp", "tensorrt", "triton", "accelerat"]),
    ("语音 / 音频", ["speech", "tts", "asr", "audio", "voice", "music", "sound"]),
    ("大模型 / LLM", ["llm", "large-language-models", "gpt", "transformer", "llama",
                     "chatgpt", "nlp", "language-model", "prompt", "chatbot", "moe"]),
    ("框架 / 工具", ["framework", "library", "toolkit", "pytorch", "tensorflow", "jax",
                    "api", "sdk", "benchmark", "dataset", "data"]),
]


def env(key, default):
    """读环境变量；为空或仅空白时回退到默认值（避免空 Secret 覆盖默认配置）。"""
    v = os.environ.get(key, "")
    v = v.strip()
    return v if v else default

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

# 常用 AI/ML 术语 → 中文，用于离线兜底翻译（按长度降序匹配，避免短词误伤长词）
TERM_ZH = {
    "machine learning": "机器学习", "deep learning": "深度学习",
    "neural network": "神经网络", "neural networks": "神经网络",
    "natural language processing": "自然语言处理",
    "large language model": "大语言模型", "language model": "语言模型",
    "language models": "语言模型", "reinforcement learning": "强化学习",
    "transfer learning": "迁移学习", "fine-tuning": "微调", "fine tuning": "微调",
    "pretraining": "预训练", "pretrained": "预训练", "pre-trained": "预训练",
    "inference": "推理", "serving": "服务部署", "deployment": "部署",
    "framework": "框架", "library": "库", "libraries": "库", "toolkit": "工具包",
    "tool": "工具", "tools": "工具", "api": "API", "sdk": "SDK", "cli": "命令行工具",
    "model": "模型", "models": "模型", "dataset": "数据集", "datasets": "数据集",
    "training": "训练", "optimizer": "优化器", "quantization": "量化",
    "embedding": "嵌入", "embeddings": "嵌入向量", "vector database": "向量数据库",
    "vector": "向量", "retrieval": "检索", "search": "搜索", "semantic": "语义",
    "rag": "检索增强", "agent": "智能体", "agents": "智能体",
    "multi-agent": "多智能体", "workflow": "工作流", "diffusion model": "扩散模型",
    "diffusion": "扩散", "text-to-image": "文生图", "image generation": "图像生成",
    "image": "图像", "images": "图像", "video": "视频", "audio": "音频",
    "speech": "语音", "voice": "语音", "tts": "语音合成", "asr": "语音识别",
    "music": "音乐", "ocr": "文字识别", "segmentation": "分割", "detection": "检测",
    "object detection": "目标检测", "transformer": "Transformer", "gpt": "GPT",
    "llm": "大语言模型", "moe": "混合专家", "pytorch": "PyTorch",
    "tensorflow": "TensorFlow", "jax": "JAX", "python": "Python",
    "gpu": "GPU", "acceleration": "加速", "accelerate": "加速",
    "distributed": "分布式", "scalable": "可扩展", "open source": "开源",
    "open-source": "开源", "real-time": "实时", "real time": "实时",
    "benchmark": "基准测试", "benchmarks": "基准测试", "tutorial": "教程",
    "tutorials": "教程", "course": "课程", "courses": "课程", "examples": "示例",
    "chatbot": "对话机器人", "chat": "对话", "prompt": "提示词", "prompts": "提示词",
    "observability": "可观测性", "monitoring": "监控", "document": "文档",
    "documents": "文档", "documentation": "文档", "photo": "照片", "photos": "照片",
    "management": "管理", "platform": "平台", "quant": "量化",
    "quantitative": "量化", "finance": "金融", "financial": "金融",
    "research": "研究", "pipeline": "流水线", "pipelines": "流水线",
    "automation": "自动化", "generative": "生成式", "generative ai": "生成式 AI",
}


def offline_zh_summary(it):
    """LLM / 缓存都不可用时，用术语词典把英文描述翻译出一句可信的中文摘要。"""
    title = it["title"]
    if title in CACHE:
        return CACHE[title]
    desc = (it.get("description") or "").strip()
    if not desc:
        return "（暂无简介）"
    seg = re.split(r'(?<=[.!?])\s', desc)[0]   # 取第一句
    for term in sorted(TERM_ZH, key=len, reverse=True):
        if re.search(r'(?i)\b' + re.escape(term) + r'\b', seg):
            seg = re.sub(r'(?i)\b' + re.escape(term) + r'\b', TERM_ZH[term], seg)
    seg = seg.strip()
    if len(seg) > 70:
        seg = seg[:70] + "…"
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


# ---------------- 抓取：GitHub ----------------
def fetch_github():
    since = (datetime.date.today() - datetime.timedelta(days=GH_RECENT_DAYS)).isoformat()
    q = f"topic:{GH_TOPIC} stars:>{GH_MIN_STARS} pushed:>{since}"
    url = ("https://api.github.com/search/repositories?q=" + urllib.parse.quote(q) +
           f"&sort=stars&order=desc&per_page={GH_PER_PAGE}")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "ai-hotspot",
                          "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
    except Exception as e:
        log("github fetch failed:", e)
        return []
    items = []
    for repo in data.get("items", []):
        stars = repo.get("stargazers_count", 0) or 0
        items.append({
            "source": "github",
            "category": classify(repo),
            "title": repo.get("full_name", ""),
            "url": repo.get("html_url", ""),
            "description": (repo.get("description") or "")[:600],
            "published": (repo.get("pushed_at") or "")[:10],
            "raw_metric": float(stars),
            "extra": {"stars": stars,
                      "language": repo.get("language"),
                      "topics": repo.get("topics", [])},
        })
    log(f"github: {len(items)} items")
    return items


# ---------------- 打分 ----------------
def score(items):
    """仅 GitHub 单一来源：用 star 数的 log 缩放做热度分（0~1），更直观。"""
    if not items:
        return items
    mx = max((it["raw_metric"] for it in items), default=1) or 1
    log_mx = math.log10(mx + 1)
    for it in items:
        it["heat_score"] = round(math.log10(it["raw_metric"] + 1) / log_mx, 4) if log_mx > 0 else 0
    items.sort(key=lambda x: x["heat_score"], reverse=True)
    for i, it in enumerate(items, 1):
        it["rank"] = i
    return items[:KEEP_TOP_N]


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
            done += 1
            if done % 10 == 0 or done == total:
                log(f"summary {done}/{total}")


# ---------------- 示例数据（离线预览） ----------------
def DEMO_ITEMS():
    today = datetime.date.today().isoformat()
    return [
        {"source": "github", "category": "大模型 / LLM",
         "title": "deepseek-ai/DeepSeek-V3",
         "url": "https://github.com/deepseek-ai/DeepSeek-V3",
         "description": "Official implementation of DeepSeek-V3, a strong Mixture-of-Experts language model with 671B total parameters.",
         "published": today, "raw_metric": 95000.0,
         "summary": "DeepSeek-V3 官方实现，671B 参数的 MoE 旗舰语言模型。",
         "extra": {"stars": 95000, "language": "Python", "topics": ["llm", "moe", "pytorch"]}},
        {"source": "github", "category": "推理 / 部署",
         "title": "vllm-project/vllm",
         "url": "https://github.com/vllm-project/vllm",
         "description": "A high-throughput and memory-efficient inference and serving engine for LLMs.",
         "published": today, "raw_metric": 38000.0,
         "summary": "高吞吐、省显存的大模型推理与服务引擎。",
         "extra": {"stars": 38000, "language": "Python", "topics": ["llm", "inference", "serving"]}},
        {"source": "github", "category": "多模态 / 视觉",
         "title": "comfyanonymous/ComfyUI",
         "url": "https://github.com/comfyanonymous/ComfyUI",
         "description": "The most powerful and modular diffusion model GUI, API and backend with a graph/node based interface.",
         "published": today, "raw_metric": 72000.0,
         "summary": "基于节点图的最强模块化扩散模型可视化与推理后端。",
         "extra": {"stars": 72000, "language": "Python", "topics": ["diffusion", "stable-diffusion", "image-generation"]}},
        {"source": "github", "category": "智能体 / Agent",
         "title": "langchain-ai/langchain",
         "url": "https://github.com/langchain-ai/langchain",
         "description": "Build context-aware reasoning applications with LLMs. Frameworks for agents, RAG and orchestration.",
         "published": today, "raw_metric": 95000.0,
         "summary": "面向 LLM 应用的开发框架，内置 Agent、RAG 与编排能力。",
         "extra": {"stars": 95000, "language": "Python", "topics": ["llm", "agents", "framework"]}},
        {"source": "github", "category": "检索增强 / RAG",
         "title": "run-llama/llama_index",
         "url": "https://github.com/run-llama/llama_index",
         "description": "LlamaIndex is a data framework for your LLM applications to ingest, structure and retrieve private data.",
         "published": today, "raw_metric": 38000.0,
         "summary": "面向 LLM 的数据框架，专注私有数据的检索增强（RAG）。",
         "extra": {"stars": 38000, "language": "Python", "topics": ["rag", "llm", "retrieval"]}},
        {"source": "github", "category": "语音 / 音频",
         "title": "openai/whisper",
         "url": "https://github.com/openai/whisper",
         "description": "Robust speech recognition via large-scale weak supervision. Approach to multilingual ASR and translation.",
         "published": today, "raw_metric": 75000.0,
         "summary": "OpenAI 开源的强鲁棒性多语种语音识别（ASR）模型。",
         "extra": {"stars": 75000, "language": "Python", "topics": ["speech", "asr", "audio"]}},
        {"source": "github", "category": "训练 / 微调",
         "title": "hiyouga/LLaMA-Factory",
         "url": "https://github.com/hiyouga/LLaMA-Factory",
         "description": "Easy and efficient LLM fine-tuning with LoRA, QLoRA and RLHF. Supports hundreds of models.",
         "published": today, "raw_metric": 42000.0,
         "summary": "易用的 LLM 微调框架，支持 LoRA / QLoRA / RLHF。",
         "extra": {"stars": 42000, "language": "Python", "topics": ["llm", "lora", "fine-tuning"]}},
        {"source": "github", "category": "框架 / 工具",
         "title": "huggingface/transformers",
         "url": "https://github.com/huggingface/transformers",
         "description": "State-of-the-art Machine Learning for PyTorch, JAX and TensorFlow. Thousands of pretrained models.",
         "published": today, "raw_metric": 140000.0,
         "summary": "最流行的深度学习模型库，集成数千个预训练模型。",
         "extra": {"stars": 140000, "language": "Python", "topics": ["pytorch", "transformers", "nlp"]}},
        {"source": "github", "category": "多模态 / 视觉",
         "title": "facebookresearch/segment-anything",
         "url": "https://github.com/facebookresearch/segment-anything",
         "description": "The Segment Anything Model (SAM): a foundation model for image segmentation with promptable masks.",
         "published": today, "raw_metric": 48000.0,
         "summary": "Meta 的 SAM：可提示驱动的通用图像分割基础模型。",
         "extra": {"stars": 48000, "language": "Python", "topics": ["computer-vision", "segmentation", "vision"]}},
        {"source": "github", "category": "智能体 / Agent",
         "title": "microsoft/autogen",
         "url": "https://github.com/microsoft/autogen",
         "description": "A framework that enables development of LLM applications using multiple agents that can converse.",
         "published": today, "raw_metric": 40000.0,
         "summary": "微软开源的多智能体对话框架，用于编排 LLM 应用。",
         "extra": {"stars": 40000, "language": "Python", "topics": ["llm", "multi-agent", "framework"]}},
    ]


# ---------------- 主流程 ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="使用内置示例数据，离线预览")
    args = p.parse_args()

    os.makedirs("public", exist_ok=True)

    if args.demo:
        items = DEMO_ITEMS()
    else:
        items = fetch_github()

    items = score(items)

    if not args.demo:
        summarize_all(items, workers=6)
        log("all summaries done")
    else:
        for it in items:
            it.setdefault("summary", "")

    out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": "github",
        "categories": CATEGORIES,
        "count": len(items),
        "items": items,
    }
    with open("public/data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"wrote public/data.json with {len(items)} items")


if __name__ == "__main__":
    main()
