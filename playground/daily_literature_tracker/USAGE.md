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

## 追踪键与 Profile
- 默认追踪键：`{profile_or_default}`
- 指定 `profile=alice` -> `alice`
- 不指定 `profile` -> `default`
- 如需强制指定，可传 `track_key=...`

## LLM 判定提示词来源
- 指定 `profile` 且存在 `playground/daily_literature_tracker/hunts/{profile}.md`：优先使用该文件
- 否则使用默认 `playground/daily_literature_tracker/paper_to_hunt.md`
- 即：不指定 profile 时，默认就是 `paper_to_hunt.md`

## 过滤流程（重要）
- 流程顺序：`抓取 -> 时间窗 -> keyword 预过滤 -> LLM 判定`
- `Filtered papers by Keyword` 不是 LLM 语义判断，而是字符串命中：
- 单词关键词（如 `security`）按摘要分词命中
- 短语关键词（如 `"intellectual property"`）按摘要子串命中
- 只要命中任意一个关键词就进入下一步 LLM 判定
- 当前默认只看摘要，不看标题
- 如果希望“不过 keyword、全部交给 LLM 判断”，可将 `keyword_list` 设为空列表 `[]`

## 配置覆盖（config.yaml）
- 你可以直接改 `playground/daily_literature_tracker/config.yaml` 覆盖默认分类、关键词和参数
- 示例：
```yaml
category_list:
  - cs.AI
  - cs.LG

keyword_list: []   # 为空表示关闭 keyword 预过滤，直接走 LLM

defaults:
  days: 1
  scan_limit: 100
  max_results: 0
  include_seen: false
  update_tracker: true
```
- 仍然遵循优先级：对话显式参数 > `config.yaml` > 工具内置兜底

## 最简调用
```text
@Magiclaw daily_literature_tracker 请做今日论文早报：先调用 arxiv_raw_check，参数 profile=alice；按“今日论文早报”模板输出。
```

## 日志
- 默认开启，包含：
  - `run` 标签（用于区分并发任务日志，避免串台误判）
  - `Total papers`
  - `Deduplicated papers across categories`
  - `Filtered papers by Keyword`
  - `LLM response for paper "...": <raw-response>`
  - `Filtered papers by LLM`
  - `Deduplicated papers`
  - `Translating Abstracts: xx%|####...|`
  - `Translated Abstracts into Chinese`
- 关闭：`DAILY_ARXIV_VERBOSE_LOG=0`

## 并发行为
- `arxiv_raw_check` 已启用单实例锁：同一时刻只允许一个任务运行，后触发的任务会等待前一个任务结束。
