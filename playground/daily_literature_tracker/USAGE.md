# Daily Literature Tracker Usage (Minimal)

## 1) 流程
- 按 `category` 逐类抓取 arXiv 最新论文（`submittedDate desc`）。
- 跨分类按去版本号后的论文 ID 去重（与 ArXivToday-Lark 一致）。
- 关键词做 token-set 初筛（`abstract.lower().split()` 交集逻辑）。
- 每篇论文固定执行 LLM Yes/No 相关性筛选（fail-open）。
- 固定执行摘要中译（先 EN，再 ZH）。
- 用 `profile + category + keyword` 生成 `track_key`，做增量去重。
- 输出包含 `daily_arxiv_digest_ready`，可被 Feishu 文献卡片解析。

## 2) 支持参数（仅保留这些）
- `profile`: 用户标识，建议始终传。
- `category`: 学科分类，支持逗号分隔多类。
- `keyword`: 关键词表达式。
- `days`: 时间窗口天数。
- `scan_limit`: 每个分类抓取上限。
- `max_results`: 最终返回上限。
- `include_seen`: 是否包含历史已见论文。
- `update_tracker`: 是否写回追踪状态。

说明：
- `use_llm_for_filtering` 与 `use_llm_for_translation` 已移除，不再需要手动传参。

## 3) 默认值（prompt 会自动补齐）
- `category=cs.AI`
- `keyword=agent`
- `days=1`
- `scan_limit=100`
- `max_results=8`
- `include_seen=false`
- `update_tracker=true`

## 4) 常用调用
### 生产模式
```text
@Magiclaw daily_literature_tracker 请做论文早报：先调用 arxiv_raw_check，参数 profile=alice, category=cs.AI, keyword=agent, include_seen=false, update_tracker=true；按“今日论文早报”模板输出。
```

### 测试模式（不污染状态）
```text
@Magiclaw daily_literature_tracker 请做论文早报测试：先调用 arxiv_raw_check，参数 profile=alice, category=cs.AI,cs.LG, keyword=agent, days=3, scan_limit=40, max_results=6, include_seen=true, update_tracker=false；按“今日论文早报”模板输出。
```

## 5) 筛选提示词
- 优先读取：`playground/daily_literature_tracker/hunts/{profile}.md`
- 回退读取：`playground/daily_literature_tracker/paper_to_hunt.md`

## 6) 状态文件
- `G:\xhe\MagiClaw\data\daily_literature_tracker\tracker_state.json`
- 测试建议：`include_seen=true, update_tracker=false`
- 生产建议：`include_seen=false, update_tracker=true`
