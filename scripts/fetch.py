#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 热点聚合脚本
==============
抓取三类来源：arXiv（论文）/ GitHub（代码）/ Hugging Face（模型），
计算统一热度分（GitHub 0.5 / Hugging Face 0.3 / arXiv 0.2），
调用大模型为每条生成一句中文摘要，输出 public/data.json。

用法：
  python scripts/fetch.py            # 正式抓取（需联网 + LLM_API_KEY）
  python scripts/fetch.py --demo     # 使用内置示例数据，离线预览前端 UI

环境变量（正式模式）：
  LLM_API_KEY     主用模型 Key（默认平台 agnes，https://apihub.agnes-ai.com/v1）
  LLM_BASE_URL    主用模型 Base URL（默认 https://apihub.agnes-ai.com/v1）
  LLM_MODEL       主用模型名（默认 agnes-2.0-flash）
  FALLBACK_API_KEY / FALLBACK_BASE_URL / FALLBACK_MODEL  备用模型（可同平台不同 Key）
未设置任何 Key 时，摘要字段留空，脚本仍会产出 data.json（用原文做兜底展示）。
本地可用 .env 文件（参考 .env.example）放置以上变量，脚本启动时会自动加载。
"""
import os
import sys
import json
import time
import argparse
import datetime
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict

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

ARXIV_CATS   = ["cs.AI", "cs.LG", "cs.CL"]
ARXIV_LIMIT  = 30

GH_TOPICS    = ["machine-learning", "deep-learning",
                "artificial-intelligence", "large-language-models", "computer-vision"]
GH_PER_TOPIC = 8
GH_MIN_STARS = 30

HF_LIMIT     = 30

# 统一热度分权重（用户自定义）
WEIGHTS      = {"github": 0.5, "huggingface": 0.3, "arxiv": 0.2}
KEEP_TOP_N   = 60

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


# ---------------- 抓取：arXiv ----------------
def fetch_arxiv(limit=ARXIV_LIMIT):
    cat_q = "+OR+".join("cat:" + c for c in ARXIV_CATS)
    url = (f"http://export.arxiv.org/api/query?search_query={cat_q}"
           f"&sortBy=submittedDate&sortOrder=descending&max_results={limit}")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
    except Exception as e:
        log("arxiv fetch failed:", e)
        return []
    ns = "{http://www.w3.org/2005/Atom}"
    root = ET.fromstring(data)
    items = []
    for e in root.findall(ns + "entry"):
        title = " ".join((e.find(ns + "title").text or "").split())
        summary = " ".join((e.find(ns + "summary").text or "").split())
        link = (e.find(ns + "id").text or "").strip()
        published = (e.find(ns + "published").text or "")[:10]
        items.append({
            "source": "arxiv",
            "title": title,
            "url": link,
            "description": summary[:600],
            "published": published,
            "raw_metric": _recency(published),
            "extra": {},
        })
    log(f"arxiv: {len(items)} items")
    return items


def _recency(date_str):
    """arXiv 没有热度字段，用新鲜度近似：30 天内线性衰减到 0。"""
    try:
        d = datetime.date.fromisoformat(date_str)
        age = (datetime.date.today() - d).days
        return max(0.0, min(1.0, 1.0 - age / 30.0))
    except Exception:
        return 0.5


# ---------------- 抓取：GitHub ----------------
def fetch_github():
    items = []
    seen = set()
    since = (datetime.date.today() - datetime.timedelta(days=90)).isoformat()
    for topic in GH_TOPICS:
        q = f"topic:{topic} stars:>{GH_MIN_STARS} pushed:>{since}"
        url = ("https://api.github.com/search/repositories?q=" +
               urllib.parse.quote(q) +
               f"&sort=stars&order=desc&per_page={GH_PER_TOPIC}")
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "ai-hotspot",
                              "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
        except Exception as e:
            log(f"github topic '{topic}' failed:", e)
            time.sleep(2)
            continue
        for repo in data.get("items", []):
            name = repo.get("full_name", "")
            if name in seen:
                continue
            seen.add(name)
            stars = repo.get("stargazers_count", 0) or 0
            items.append({
                "source": "github",
                "title": name,
                "url": repo.get("html_url", ""),
                "description": (repo.get("description") or "")[:600],
                "published": (repo.get("pushed_at") or "")[:10],
                "raw_metric": float(stars),
                "extra": {"stars": stars, "language": repo.get("language")},
            })
        time.sleep(1.5)  # 避免触发 GitHub 未认证限流（10 次/分钟）
    log(f"github: {len(items)} items")
    return items


# ---------------- 抓取：Hugging Face ----------------
def fetch_huggingface(limit=HF_LIMIT):
    url = (f"https://huggingface.co/api/models?sort=trendingScore"
           f"&direction=-1&limit={limit}&full=true&config=false")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ai-hotspot"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
    except Exception as e:
        log("huggingface fetch failed:", e)
        return []
    items = []
    for m in data:
        mid = m.get("id", "")
        likes = m.get("likes", 0) or 0
        downloads = m.get("downloads", 0) or 0
        card = m.get("cardData") or {}
        desc = card.get("description") or m.get("description") or ""
        # 热度信号优先用 likes，没有 likes 时用 downloads 近似
        metric = float(likes) if likes > 0 else float(downloads) / 1000.0
        items.append({
            "source": "huggingface",
            "title": mid,
            "url": "https://huggingface.co/" + mid,
            "description": str(desc)[:600],
            "published": (m.get("lastModified") or m.get("createdAt") or "")[:10],
            "raw_metric": metric,
            "extra": {"likes": likes, "downloads": downloads,
                      "pipeline_tag": m.get("pipeline_tag")},
        })
    log(f"huggingface: {len(items)} items")
    return items


# ---------------- 打分 ----------------
def score(items):
    by_src = defaultdict(list)
    for it in items:
        by_src[it["source"]].append(it)
    for src, lst in by_src.items():
        mx = max((it["raw_metric"] for it in lst), default=1) or 1
        for it in lst:
            norm = it["raw_metric"] / mx if mx > 0 else 0
            it["norm"] = round(norm, 4)
            it["heat_score"] = round(WEIGHTS.get(src, 0) * norm, 4)
    items.sort(key=lambda x: x["heat_score"], reverse=True)
    for i, it in enumerate(items, 1):
        it["rank"] = i
    return items[:KEEP_TOP_N]


# ---------------- 大模型摘要 ----------------
SUMMARY_SYS = ("你是一个 AI 领域编辑。请用一句简洁的中文（专业术语保留英文）"
               "概括下面内容的核心贡献或用途，不超过 40 字，不要使用引号或编号。")


def call_llm(system, user):
    if not PRIMARY["key"] and not FALLBACK["key"]:
        return None
    payload = {
        "model": None,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.3,
        "max_tokens": 400,
    }
    for cfg in (PRIMARY, FALLBACK):
        if not cfg["key"]:
            continue
        payload["model"] = cfg["model"]
        try:
            import requests
            r = requests.post(
                cfg["base"].rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + cfg["key"],
                         "Content-Type": "application/json"},
                json=payload, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            if r.status_code == 429:
                time.sleep(5)
                continue
            log("llm err", r.status_code, r.text[:200])
        except Exception as e:
            log("llm call failed:", e)
        time.sleep(2)
    return None


def summarize(it):
    text = (it.get("description") or "").strip()
    if not text:
        return ""
    user = f"标题：{it['title']}\n内容：{text[:800]}"
    s = call_llm(SUMMARY_SYS, user)
    return s or ""


# ---------------- 示例数据（离线预览） ----------------
def DEMO_ITEMS():
    today = datetime.date.today().isoformat()
    return [
        # arXiv
        {"source": "arxiv", "title": "LLaMA-Vision: Unified Multimodal Instruction Tuning at Scale",
         "url": "https://arxiv.org/abs/2408.00001",
         "description": "We present a vision-language model that unifies image and text understanding via a single instruction-tuning recipe, achieving strong zero-shot performance on OCR, chart and document tasks.",
         "published": today, "raw_metric": 1.0,
         "summary": "提出统一多模态指令微调方法，在 OCR 与文档理解上实现强零样本表现。", "extra": {}},
        {"source": "arxiv", "title": "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion",
         "url": "https://arxiv.org/abs/2303.04137",
         "description": "We propose Diffusion Policy, a new way to represent robot visuomotor policies by denoising stochastic temporal ensembles of actions, showing significant improvement on 15 tasks.",
         "published": today, "raw_metric": 0.8,
         "summary": "用扩散模型表征机器人视觉运动策略，在 15 项任务上明显优于基线。", "extra": {}},
        {"source": "arxiv", "title": "LoRA+: Efficient Low Rank Adaptation for Fine-Tuning",
         "url": "https://arxiv.org/abs/2402.12354",
         "description": "We show that the two matrices in LoRA should be trained with different learning rates and propose LoRA+ which converges faster with negligible extra cost.",
         "published": today, "raw_metric": 0.6,
         "summary": "指出 LoRA 双矩阵应分别设学习率，提出收敛更快的 LoRA+。", "extra": {}},
        # GitHub
        {"source": "github", "title": "deepseek-ai/DeepSeek-V3",
         "url": "https://github.com/deepseek-ai/DeepSeek-V3",
         "description": "Official implementation of DeepSeek-V3, a strong Mixture-of-Experts language model with 671B total parameters.",
         "published": today, "raw_metric": 95000.0,
         "summary": "DeepSeek-V3 官方实现，671B 参数的 MoE 旗舰语言模型。", "extra": {"stars": 95000, "language": "Python"}},
        {"source": "github", "title": "vllm-project/vllm",
         "url": "https://github.com/vllm-project/vllm",
         "description": "A high-throughput and memory-efficient inference and serving engine for LLMs.",
         "published": today, "raw_metric": 38000.0,
         "summary": "高吞吐、省显存的大模型推理与服务引擎。", "extra": {"stars": 38000, "language": "Python"}},
        {"source": "github", "title": "comfyanonymous/ComfyUI",
         "url": "https://github.com/comfyanonymous/ComfyUI",
         "description": "The most powerful and modular diffusion model GUI, API and backend with a graph/node based interface.",
         "published": today, "raw_metric": 72000.0,
         "summary": "基于节点图的最强模块化扩散模型可视化与推理后端。", "extra": {"stars": 72000, "language": "Python"}},
        # Hugging Face
        {"source": "huggingface", "title": "meta-llama/Llama-3.2-11B-Vision-Instruct",
         "url": "https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct",
         "description": "Instruction-tuned multimodal model from Meta, supporting image reasoning and chat.",
         "published": today, "raw_metric": 1200.0,
         "summary": "Meta 出品的指令微调多模态模型，支持图像推理与对话。", "extra": {"likes": 1200, "downloads": 500000, "pipeline_tag": "image-text-to-text"}},
        {"source": "huggingface", "title": "Qwen/Qwen2.5-72B-Instruct",
         "url": "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct",
         "description": "Qwen2.5 72B instruction-tuned model with strong coding and math ability.",
         "published": today, "raw_metric": 2100.0,
         "summary": "通义千问 72B 指令模型，代码与数学能力突出。", "extra": {"likes": 2100, "downloads": 900000, "pipeline_tag": "text-generation"}},
        {"source": "huggingface", "title": "stabilityai/stable-diffusion-3.5-large",
         "url": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
         "description": "The largest Stable Diffusion 3.5 model for high-quality text-to-image generation.",
         "published": today, "raw_metric": 850.0,
         "summary": "Stable Diffusion 3.5 最大版本，高质量文生图模型。", "extra": {"likes": 850, "downloads": 300000, "pipeline_tag": "text-to-image"}},
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
        items = fetch_arxiv() + fetch_github() + fetch_huggingface()

    items = score(items)

    use_llm = bool(PRIMARY["key"] or FALLBACK["key"])
    if not args.demo and use_llm:
        for it in items:
            it["summary"] = summarize(it)
            time.sleep(0.5)  # 温和限速，避免触发模型限流
    else:
        for it in items:
            it.setdefault("summary", "")

    out = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "weights": WEIGHTS,
        "count": len(items),
        "items": items,
    }
    with open("public/data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"wrote public/data.json with {len(items)} items")


if __name__ == "__main__":
    main()
