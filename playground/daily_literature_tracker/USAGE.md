# Daily Literature Tracker Usage (Minimal)

## 当前默认（来自 config.yaml）
- `category_list`: `cs.CL,cs.AI,cs.CV,cs.CR,cs.LG`
- `keyword_list`: 安全相关 15 个关键词
- `days: 1`（日报，默认只看最近一天）
- `scan_limit: 100`（每个分类默认抓取 100 篇，和 ArXivToday-Lark 对齐）
- `max_results: 0`（all，不设返回上限）

## 参数语义
- `days=1`：包含今天与昨天（日报默认）
- `days>1`：包含今天与最近 N 天
- `days=0`：不按日期过滤
- `scan_limit=100`：每个分类默认抓 100 篇（推荐，开源同款）
- `scan_limit=0`：每个分类不设上限抓取（全量模式）
- `scan_limit>0`：每个分类最多抓 N 篇
- `max_results=0`：返回全部相关论文

## 优先级
- 对话显式参数 > `config.yaml` > 工具内置兜底

## 最简调用
```text
@Magiclaw daily_literature_tracker 请做今日论文早报：先调用 arxiv_raw_check，参数 profile=alice；按“今日论文早报”模板输出。
```

## 日志
- 默认开启，包含：
  - `Total papers`
  - `Deduplicated papers across categories`
  - `Filtered papers by Keyword`
  - `LLM response for paper "...": <raw-response>`
  - `Filtered papers by LLM`
  - `Deduplicated papers`
  - `Translating Abstracts: xx%|####...|`
  - `Translated Abstracts into Chinese`
- 关闭：`DAILY_ARXIV_VERBOSE_LOG=0`
