# Daily Literature Tracker Usage (Minimal)

## 默认行为（极简日报）
- 默认分类：`cs.CL,cs.AI,cs.CV,cs.CR,cs.LG`
- 默认时间窗：`days=1`（日报）
- 默认抓取：`scan_limit=auto`（自动翻页到时间窗边界）
- 默认返回：`max_results=all`（相关论文全发，不截断）
- 通过 `profile` 对应 hunts 提示词 + LLM 对每篇论文判定相关性
- 用 `tracker_state.json` 做增量去重

## 你通常只需要传这两个参数
- `profile`
- `category`（可不传，不传就用默认 5 个分类）

## 可选参数
- `days`：默认 `1`
- `keyword`：默认空（不做关键词预筛）
- `scan_limit`：默认 `0/auto`
- `max_results`：默认 `0/all`
- `include_seen`：默认 `false`
- `update_tracker`：默认 `true`

## 推荐调用（日报）
```text
@Magiclaw daily_literature_tracker 请做今日论文早报：先调用 arxiv_raw_check，参数 profile=alice；按“今日论文早报”模板输出。
```

## 运行日志（默认开启）
- 会输出类似 ArXivToday-Lark 的日志：
  - `Task / Params`
  - `Total papers fetched`
  - `Filtered papers by keyword / LLM`
  - `LLM response for paper "...": Yes/No`
  - `Deduplicated papers`
  - `Translating Abstracts: 63%|████...| 22/35 [05:20<02:55, 13.54s/it]`（进度条）
  - `Run summary`
- 关闭详细日志：设置环境变量 `DAILY_ARXIV_VERBOSE_LOG=0`

## 状态文件
- `G:\xhe\MagiClaw\data\daily_literature_tracker\tracker_state.json`
- 结构：`track_key -> papers[]`
- 去重：按 `paper_id`，新论文在前
- 保留：每个 track 最多 `2000` 条
