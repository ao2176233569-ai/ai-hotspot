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
    "agentic": "智能体化", "mcp": "模型上下文协议", "copilot": "编程助手",
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
            "summary": summ,
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
