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

# GitHub 抓取：围绕一个宽泛的 AI topic，取总 star 较高的项目组成"稳定候选池"。
# 用稳定池才能持续累积 star 历史、算出日/月/年增量（避免用 pushed 过滤把老牌库排除）。
GH_TOPIC       = "machine-learning"
GH_PER_PAGE    = 100
GH_MIN_STARS   = 200
GH_POOL_SIZE   = 500          # 稳定头部候选池上限（5 页 × 100）
GH_POOL_CAP    = 800          # 合并「稳定头部 + 上升新星」后的总候选池上限
GH_PAGES       = 5
GH_RISING_PAGES = 3           # 上升新星每个查询翻页数（控制 API 调用次数）

KEEP_TOP_N     = 30           # 每个时间窗（当天/当月/当年）各取 Top 30

# star 增量窗口（天）
DELTA_DAY   = 1
DELTA_MONTH = 30
DELTA_YEAR  = 365

# star 历史快照文件（每次构建写回，用于算增量）。不进 git，由 workflow 用 gh 上传/下载。
HISTORY_PATH = env("HISTORY_PATH", "stars_history.json")
HISTORY_KEEP_DAYS = 400       # 只保留最近 400 天快照，控制文件体积

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
                  "forks": repo.get("forks_count", 0),
                  "language": repo.get("language"),
                  "topics": repo.get("topics", []),
                  "created_at": repo.get("created_at") or ""},
    }


def fetch_github():
    """候选池 = 稳定头部(总 star 前 N) + 上升新星(近 2 年新建即爆红 / 近 60 天活跃的高 star 库)。
    目的是让榜单出现真正在涨、新冒头的热门项目，而不是只把所有时间的巨头按总 star 排。"""
    token = _gh_token()
    today = datetime.date.today()
    recent_created = (today - datetime.timedelta(days=730)).isoformat()   # 近 2 年新建
    recent_pushed = (today - datetime.timedelta(days=60)).isoformat()    # 近 60 天活跃
    queries = [
        (f"topic:{GH_TOPIC} stars:>{GH_MIN_STARS}", GH_PAGES),          # 稳定头部
        (f"topic:{GH_TOPIC} stars:>300 created:>{recent_created}", GH_RISING_PAGES),  # 新建即爆红
        (f"topic:{GH_TOPIC} stars:>300 pushed:>={recent_pushed}", GH_RISING_PAGES),  # 近期活跃热库
    ]
    items, seen = [], set()
    for q, pages in queries:
        for page in range(1, pages + 1):
            for repo in _gh_search_once(q, page, token):
                name = repo.get("full_name", "")
                if name in seen:
                    continue
                seen.add(name)
                items.append(_normalize_repo(repo))
            if len(items) >= GH_POOL_CAP:
                break
            time.sleep(1)   # 放慢，避免触发 GitHub 搜索速率限制
        if len(items) >= GH_POOL_CAP:
            break
    items = items[:GH_POOL_CAP]
    log(f"github: {len(items)} items (pool, incl. rising stars)")
    return items


# ---------------- star 历史（用于算增量） ----------------
def load_history(path):
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log("history load failed:", e)
    return {}


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
    history[today] = current_map
    dates = sorted(history.keys())
    if len(dates) > HISTORY_KEEP_DAYS:
        for d in dates[:-HISTORY_KEEP_DAYS]:
            history.pop(d, None)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=1, sort_keys=True)
    except Exception as e:
        log("history write local failed:", e)
    if os.environ.get("UPLOAD_HISTORY") == "1":
        upload_history_to_repo(path)


def delta_for(history, full_name, current_stars, days_back):
    """返回 days_back 天前的 star 增量（current - 历史快照）。找不到更早快照返回 None。"""
    today = datetime.date.today()
    target = (today - datetime.timedelta(days=days_back)).isoformat()
    best = None
    for d in history:                      # history 形如 {date: {full_name: stars}}
        if d <= target and (best is None or d > best):
            best = d
    if best is None:
        return None
    past = history[best].get(full_name)
    if past is None:
        return None
    return max(0, current_stars - past)


def compute_deltas(items, history):
    for it in items:
        s = it["extra"]["stars"]
        it["day_delta"]   = delta_for(history, it["title"], s, DELTA_DAY)
        it["month_delta"] = delta_for(history, it["title"], s, DELTA_MONTH)
        it["year_delta"]  = delta_for(history, it["title"], s, DELTA_YEAR)


def add_velocity(items, history):
    """给每个 item 算「日均涨星」(stars/day)：
    - 有真实增量窗口时：增量 / 天数（当日=day_delta，当月=month_delta/30，当年=year_delta/365）。
    - 首日无基线时：用 总star / 库龄 估算（估算值，非真实统计），让上升新星当天就能冒头。
    排序改用 velocity 而非绝对增量，避免巨头永远霸榜，使新冒头、涨得猛的库浮到前面。"""
    today = datetime.date.today()
    for it in items:
        s = it["extra"]["stars"]
        ca = (it["extra"].get("created_at") or "")[:10]
        try:
            age = max(1, (today - datetime.date.fromisoformat(ca)).days)
        except Exception:
            age = 3650
        it["extra"]["age_days"] = age
        it["day_velocity"]   = (it["day_delta"]   / 1.0)   if it.get("day_delta")   is not None else s / age
        it["month_velocity"] = (it["month_delta"] / 30.0)  if it.get("month_delta") is not None else s / age
        it["year_velocity"]  = (it["year_delta"]  / 365.0) if it.get("year_delta")  is not None else s / age


def score_all(items):
    """给每个 item 算 log 热度分（0~1），用于排序选项。"""
    if not items:
        return
    mx = max((it["raw_metric"] for it in items), default=1) or 1
    log_mx = math.log10(mx + 1)
    for it in items:
        it["heat_score"] = round(math.log10(it["raw_metric"] + 1) / log_mx, 4) if log_mx > 0 else 0


def build_range(items, vel_key):
    """按某时间窗的「日均涨星」(velocity, stars/day) 降序取 Top N；
    首日无基线时 velocity 由 总star/库龄 估算，仍能把上升新星排到前面。"""
    def sort_key(it):
        v = it.get(vel_key)
        return (v is not None, v if v is not None else 0, it["raw_metric"])
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
    add_velocity(items, history)
    return {
        "day":   build_range(items, "day_velocity"),
        "month": build_range(items, "month_velocity"),
        "year":  build_range(items, "year_velocity"),
    }


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
         "day_delta": 320, "month_delta": 5200, "year_delta": 95000,
         "extra": {"stars": 95000, "language": "Python", "topics": ["llm", "moe", "pytorch"]}},
        {"source": "github", "category": "推理 / 部署",
         "title": "vllm-project/vllm",
         "url": "https://github.com/vllm-project/vllm",
         "description": "A high-throughput and memory-efficient inference and serving engine for LLMs.",
         "published": today, "raw_metric": 38000.0,
         "summary": "高吞吐、省显存的大模型推理与服务引擎。",
         "day_delta": 85, "month_delta": 1800, "year_delta": 30000,
         "extra": {"stars": 38000, "language": "Python", "topics": ["llm", "inference", "serving"]}},
        {"source": "github", "category": "多模态 / 视觉",
         "title": "comfyanonymous/ComfyUI",
         "url": "https://github.com/comfyanonymous/ComfyUI",
         "description": "The most powerful and modular diffusion model GUI, API and backend with a graph/node based interface.",
         "published": today, "raw_metric": 72000.0,
         "summary": "基于节点图的最强模块化扩散模型可视化与推理后端。",
         "day_delta": 150, "month_delta": 2400, "year_delta": 60000,
         "extra": {"stars": 72000, "language": "Python", "topics": ["diffusion", "stable-diffusion", "image-generation"]}},
        {"source": "github", "category": "智能体 / Agent",
         "title": "langchain-ai/langchain",
         "url": "https://github.com/langchain-ai/langchain",
         "description": "Build context-aware reasoning applications with LLMs. Frameworks for agents, RAG and orchestration.",
         "published": today, "raw_metric": 95000.0,
         "summary": "面向 LLM 应用的开发框架，内置 Agent、RAG 与编排能力。",
         "day_delta": 40, "month_delta": 800, "year_delta": 90000,
         "extra": {"stars": 95000, "language": "Python", "topics": ["llm", "agents", "framework"]}},
        {"source": "github", "category": "检索增强 / RAG",
         "title": "run-llama/llama_index",
         "url": "https://github.com/run-llama/llama_index",
         "description": "LlamaIndex is a data framework for your LLM applications to ingest, structure and retrieve private data.",
         "published": today, "raw_metric": 38000.0,
         "summary": "面向 LLM 的数据框架，专注私有数据的检索增强（RAG）。",
         "day_delta": 60, "month_delta": 900, "year_delta": 35000,
         "extra": {"stars": 38000, "language": "Python", "topics": ["rag", "llm", "retrieval"]}},
        {"source": "github", "category": "语音 / 音频",
         "title": "openai/whisper",
         "url": "https://github.com/openai/whisper",
         "description": "Robust speech recognition via large-scale weak supervision. Approach to multilingual ASR and translation.",
         "published": today, "raw_metric": 75000.0,
         "summary": "OpenAI 开源的强鲁棒性多语种语音识别（ASR）模型。",
         "day_delta": 20, "month_delta": 400, "year_delta": 70000,
         "extra": {"stars": 75000, "language": "Python", "topics": ["speech", "asr", "audio"]}},
        {"source": "github", "category": "训练 / 微调",
         "title": "hiyouga/LLaMA-Factory",
         "url": "https://github.com/hiyouga/LLaMA-Factory",
         "description": "Easy and efficient LLM fine-tuning with LoRA, QLoRA and RLHF. Supports hundreds of models.",
         "published": today, "raw_metric": 42000.0,
         "summary": "易用的 LLM 微调框架，支持 LoRA / QLoRA / RLHF。",
         "day_delta": 110, "month_delta": 2600, "year_delta": 40000,
         "extra": {"stars": 42000, "language": "Python", "topics": ["llm", "lora", "fine-tuning"]}},
        {"source": "github", "category": "框架 / 工具",
         "title": "huggingface/transformers",
         "url": "https://github.com/huggingface/transformers",
         "description": "State-of-the-art Machine Learning for PyTorch, JAX and TensorFlow. Thousands of pretrained models.",
         "published": today, "raw_metric": 140000.0,
         "summary": "最流行的深度学习模型库，集成数千个预训练模型。",
         "day_delta": 95, "month_delta": 1500, "year_delta": 130000,
         "extra": {"stars": 140000, "language": "Python", "topics": ["pytorch", "transformers", "nlp"]}},
        {"source": "github", "category": "多模态 / 视觉",
         "title": "facebookresearch/segment-anything",
         "url": "https://github.com/facebookresearch/segment-anything",
         "description": "The Segment Anything Model (SAM): a foundation model for image segmentation with promptable masks.",
         "published": today, "raw_metric": 48000.0,
         "summary": "Meta 的 SAM：可提示驱动的通用图像分割基础模型。",
         "day_delta": 15, "month_delta": 300, "year_delta": 46000,
         "extra": {"stars": 48000, "language": "Python", "topics": ["computer-vision", "segmentation", "vision"]}},
        {"source": "github", "category": "智能体 / Agent",
         "title": "microsoft/autogen",
         "url": "https://github.com/microsoft/autogen",
         "description": "A framework that enables development of LLM applications using multiple agents that can converse.",
         "published": today, "raw_metric": 40000.0,
         "summary": "微软开源的多智能体对话框架，用于编排 LLM 应用。",
         "day_delta": 70, "month_delta": 1100, "year_delta": 38000,
         "extra": {"stars": 40000, "language": "Python", "topics": ["llm", "multi-agent", "framework"]}},
    ]


def DEMO_HISTORY():
    """用 DEMO_ITEMS 里预设的增量反推历史快照，使预览模式下三榜单呈现真实差异。"""
    today = datetime.date.today()
    d1 = (today - datetime.timedelta(days=1)).isoformat()
    d30 = (today - datetime.timedelta(days=30)).isoformat()
    d365 = (today - datetime.timedelta(days=365)).isoformat()
    h = {d1: {}, d30: {}, d365: {}}
    for it in DEMO_ITEMS():
        s = it["extra"]["stars"]
        h[d1][it["title"]]   = s - it["day_delta"]
        h[d30][it["title"]]  = s - it["month_delta"]
        h[d365][it["title"]] = s - it["year_delta"]
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
        current_map = {it["title"]: it["extra"]["stars"] for it in items}
        save_history(history, HISTORY_PATH, today, current_map)

    ranges = build_ranges(items, history)

    if args.demo:
        for rng in ranges.values():
            for it in rng:
                it.setdefault("summary", "")

    out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": "github",
        "mode": "star-delta",
        "categories": CATEGORIES,
        "ranges": ranges,
        "count": {k: len(v) for k, v in ranges.items()},
    }
    with open("public/data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"wrote public/data.json: day={len(ranges['day'])} month={len(ranges['month'])} year={len(ranges['year'])}")


if __name__ == "__main__":
    main()
