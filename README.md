# AI 热点聚合 · ai-hotspot

自动收集 **GitHub 上最热门的 AI 项目**，按 AI 子领域自动分类，并为每条生成一句**中文摘要**，以静态卡片流网页呈现。每天北京时间 08:00 / 20:00 自动更新，零服务器成本。

> 在线示例：https://ao2176233569-ai.github.io/ai-hotspot/

---

## ✨ 特性

- **单一可信数据源**：只聚合 GitHub（Search API），围绕 `machine-learning` topic，筛选近 30 天活跃、star > 100 的项目，取热度 Top 30。
- **自动分类**：基于 GitHub topics + 描述，确定性地归入 9 个 AI 子领域，前端支持按分类筛选。
- **全中文摘要**：优先调用大模型（agnes-2.0-flash）；命中本地缓存即用缓存；都不行则用离线中文词典兜底——**每张卡片永远有中文内容，绝不会出现空白或纯英文**。
- **静态托管 + 定时构建**：GitHub Actions 抓取并构建，GitHub Pages 部署，无需任何后端。
- **中文界面**：界面全中文，专业术语（LLM、MoE、Agent、RAG、diffusion 等）保留英文。

---

## 🗂 分类体系（9 类）

大模型 / LLM、`智能体 / Agent`、`多模态 / 视觉`、`检索增强 / RAG`、
`训练 / 微调`、`推理 / 部署`、`语音 / 音频`、`框架 / 工具`、`其他`

分类规则在 `scripts/fetch.py` 的 `CATEGORIES` 与 `CAT_KEYWORDS` 中维护，顺序即优先级（越具体的子类越靠前）。

---

## 🏗 工作原理

```
GitHub Search API
      │  抓取近30天活跃、star>100 的 machine-learning 项目
      ▼
   打分（star 数 log 缩放 → 0~1 热度分）
      ▼
   分类（topics + 描述 → 9 类之一）
      ▼
   摘要（缓存命中 → 直接中文；否则试大模型；失败 → 离线中文兜底）
      ▼
   输出 public/data.json  ──▶  GitHub Actions 构建 ──▶  GitHub Pages 部署
```

目录结构：

```
.
├── .github/workflows/build.yml   # 定时构建 + 部署（cron 0 0,12 * * *）
├── public/
│   ├── index.html                # 前端卡片流（分类筛选 / 搜索 / 排序）
│   └── data.json                 # 由脚本生成的榜单数据（被 .gitignore 忽略）
├── scripts/
│   ├── fetch.py                  # 抓取 / 分类 / 摘要 主脚本
│   └── requirements.txt          # 依赖（仅 requests）
├── summaries_cache.json          # 手写中文摘要缓存（按仓库 full_name 索引）
├── .env.example                  # 大模型密钥样例
└── README.md
```

---

## 💻 本地运行

需要 Python 3.11+。

```bash
# 1. 安装依赖
pip install -r scripts/requirements.txt

# 2a. 离线预览前端 UI（使用内置示例数据，无需联网/密钥）
python scripts/fetch.py --demo
# 然后起一个本地静态服务器查看：
python -m http.server 8000 --directory public
# 浏览器打开 http://localhost:8000

# 2b. 真实抓取并生成榜单（需联网；可选配置大模型密钥）
export LLM_API_KEY=sk-xxxx          # 主用模型 Key
export LLM_BASE_URL=https://apihub.agnes-ai.com/v1
export LLM_MODEL=agnes-2.0-flash
python scripts/fetch.py             # 输出 public/data.json
```

> 不配置任何密钥也能跑：脚本会自动走 `summaries_cache.json` + 离线中文词典兜底，产出**全中文**的 `data.json`。

---

## 🚀 部署到 GitHub Pages

1. **创建仓库**：新建一个公开仓库，把本项目文件推上去（`.github/`、`public/`、`scripts/`、`summaries_cache.json`、`.env.example`）。
2. **（可选）配置大模型密钥**：仓库 `Settings → Secrets and variables → Actions` 中添加以下 Secrets（不配置则用缓存+离线兜底，依旧全中文）：
   - `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`（主用）
   - `FALLBACK_API_KEY` / `FALLBACK_BASE_URL` / `FALLBACK_MODEL`（备用）
3. **开启 Pages**：`Settings → Pages → Source` 选择 **GitHub Actions**。
4. **触发构建**：
   - 每次推送 `main` 分支会自动构建；
   - 也可在 `Actions → Build AI Hotspot → Run workflow` 手动触发；
   - 定时任务每天 **北京时间 08:00 与 20:00** 自动刷新（`cron: "0 0,12 * * *"`，UTC）。

部署完成后，Pages 地址即为你的站点 URL。

---

## 🛠 自定义

| 想改什么 | 改哪里 |
| --- | --- |
| 抓取的主题 / 最低 star / 时间窗口 / 榜单数量 | `scripts/fetch.py` 顶部：`GH_TOPIC`、`GH_MIN_STARS`、`GH_RECENT_DAYS`、`KEEP_TOP_N` |
| 分类类别与关键词映射 | `scripts/fetch.py` 的 `CATEGORIES` 与 `CAT_KEYWORDS` |
| 给某个项目定制地道中文摘要 | 在 `summaries_cache.json` 增加 `"仓库full_name": "中文摘要"` 条目 |
| 析出英文描述的中文术语表 | `scripts/fetch.py` 的 `TERM_ZH` 词典 |
| 定时频率 | `.github/workflows/build.yml` 里的 `cron` 表达式 |

---

## 📌 说明与免责

- 数据来自 GitHub 公开 API，热门度以 star 为主要信号，仅供参考。
- 摘要由 AI / 缓存 / 离线词典生成，可能不完全准确，请以项目原仓库为准。
- 本项目仅作技术演示，不对榜单完整性或摘要准确性作保证。
