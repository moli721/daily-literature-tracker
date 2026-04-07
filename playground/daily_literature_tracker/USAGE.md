# Daily Literature Tracker Usage

## 1) 核心流程
- 抓取 arXiv 最新论文（支持多分类）。
- 关键词初筛（默认 `keyword_mode=token_set`）。
- LLM 语义筛选（默认 fail-open，避免因 LLM 故障误报“无新增”）。
- 去重（按 `track_key` / `profile` 隔离）。
- 中英摘要输出（可关闭翻译）。

## 2) 多人共用（推荐）
多人使用同一个 agent 时，建议统一传 `profile`。

效果：
- 去重状态隔离：自动生成 profile 专属 `track_key`（当未手动指定 `track_key` 时）。
- 筛选规则隔离：自动读取 `hunts/{profile}.md` 作为该用户筛选规则（当未手动传 `llm_filter_prompt_file` 时）。

目录示例：
- `playground/daily_literature_tracker/hunts/alice.md`
- `playground/daily_literature_tracker/hunts/bob.md`
- 模板：`playground/daily_literature_tracker/hunts/_template.md`

## 3) 常用调用
### 生产模式
```text
@Magiclaw daily_literature_tracker 请做论文早报：先调用 arxiv_raw_check，参数 profile=alice, category=cs.AI, keyword=agent, include_seen=false, update_tracker=true；按“今日论文早报”模板输出。
```

### 测试模式（不污染状态）
```text
@Magiclaw daily_literature_tracker 请做论文早报测试：先调用 arxiv_raw_check，参数 profile=alice, category=cs.AI, keyword=agent, include_seen=true, update_tracker=false；按“今日论文早报”模板输出。
```

### 扩大扫描范围
```text
@Magiclaw daily_literature_tracker 请做论文早报测试：先调用 arxiv_raw_check，参数 profile=alice, category=cs.AI,cs.LG,cs.CL,cs.CR, keyword=agent, days=2, scan_limit=300, max_results=12, include_seen=true, update_tracker=false；按“今日论文早报”模板输出。
```

## 4) 参数说明（重点）
- `profile`: 用户标识，建议必传。
- `category`: 学科分类，可多项（逗号分隔）。
- `keyword`: 关键词表达式（支持 `OR` / `AND`）。
- `keyword_mode`: `token_set`（默认）或 `expression`。
- `token_set_scope`: `abstract`（默认，贴近 ArXivToday）或 `title_abstract`。
- `keyword_list`: 逗号分隔关键词列表（优先于 `keyword` 的分词结果）。
- `days`: 时间窗口，默认 `1`。
- `scan_limit`: 抓取上限，默认 `100`。
- `max_results`: 返回上限，默认 `8`。
- `include_seen`: 是否包含历史已推送论文。
- `update_tracker`: 是否写回去重状态。
- `use_llm_for_filtering`: 是否启用 LLM 语义筛选。
- `llm_filter_fail_open`: LLM 筛选失败时是否放行（默认 `true`）。
- `strict_llm_filter`: 是否严格执行 fail-close。默认 `false`；为 `false` 时会强制 fail-open（更稳妥）。
- `use_llm_for_translation`: 是否启用摘要翻译。

## 5) 要“专门找某类论文”改哪里
优先修改：
- `playground/daily_literature_tracker/hunts/{profile}.md`

全局默认规则：
- `playground/daily_literature_tracker/paper_to_hunt.md`

## 6) 追踪状态文件
- 文件：`G:\xhe\MagiClaw\data\daily_literature_tracker\tracker_state.json`
- 调试建议：`include_seen=true, update_tracker=false`
- 生产建议：`include_seen=false, update_tracker=true`

## 7) 与 ArXivToday 对齐点
- `token_set` 默认使用 `abstract.lower().split()` 做关键词交集（可选范围）。
- LLM 筛选默认 fail-open（网络故障/鉴权失败不会把论文全过滤掉）。
- 先关键词后 LLM，再做去重与输出。
