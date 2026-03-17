# OpenAlex Avatar Pipeline

本项目已完全切换到当前新链路：

`DB input -> Qwen web_search_image tool output -> candidate normalization/dedupe -> first candidate -> image validation -> OSS upload -> DB upsert -> local run audit`

## 输入源与输入字段

作者输入来自两张表：

- `public.authors_analysis`
- `public.author_last_known_institution`

输入字段（进入 pipeline 的 `AuthorRecord`）：

- `author_id`
- `display_name`
- `orcid`（在代码中由 `orcid_url` property 归一化得到）
- `institution_name`

## 搜图链路（当前实现）

- 使用 Qwen Responses API
- 工具：`web_search_image`
- 单次请求调用
- 只消费 tool output 构建候选
- 不做“tool output 回灌模型”
- 不依赖模型文本输出构建候选（`response_text` 仅用于审计）

## 后处理链路

1. 从 tool output 提取图片 URL 与 source URL
2. 候选标准化与去重（按 URL 去重、内部排序分值）
3. 取首个候选进入后续处理
4. 下载图片并读取基础元数据
5. MIME / 尺寸校验
6. 计算 `sha256`
7. 上传 OSS
8. upsert `public.authors_avatars`

## Source Pages 语义

- `source_pages` 仅是调试/审计透传信息（由 tool output 的 source URL 推导）
- 不参与 HTML 抓取
- 不存在 profile page 二次抽图逻辑

## 本地运行输出（runs 审计）

每次运行目录：

```text
runs/<date>/<run_id>/
  summary.json
  planned_authors.jsonl
  author_runs.jsonl
  successes.jsonl
  failures.jsonl
```

文件说明：

- `summary.json`：运行总体状态、进度、计数、配置快照、输入摘要
- `planned_authors.jsonl`：本次计划处理作者列表
- `author_runs.jsonl`：每个作者完整处理结果
- `successes.jsonl`：成功子集
- `failures.jsonl`：失败子集

`author_runs.jsonl` 中常见审计字段：

- `source_pages`（调试透传）
- `image_candidates`
- `filtered_candidates`
- `selected_candidate`
- `raw_content`
- `response_text`（audit-only）
- `usage_total_tokens`

## 环境变量（仅当前有效项）

### Postgres

- `PGHOST`
- `PGPORT`（默认 `5432`）
- `PGDATABASE`
- `PGUSER`
- `PGPASSWORD`
- `PGSSLMODE`（可选）

### OSS

- `ALIYUN_OSS_ACCESS_KEY_ID`
- `ALIYUN_OSS_ACCESS_KEY_SECRET`
- `ALIYUN_OSS_BUCKET`
- `ALIYUN_OSS_ENDPOINT`
- `ALIYUN_OSS_PUBLIC_BASE_URL`
- `ALIYUN_OSS_KEY_PREFIX`（默认 `openalex`）
- `ALIYUN_OSS_CACHE_CONTROL`（可选）

### HTTP / Retry / Rate Limit

- `REQUEST_TIMEOUT_SECONDS`（默认 `20`）
- `MAX_RETRIES`（默认 `3`）
- `GLOBAL_QPS_LIMIT`（默认 `2`）
- `RETRY_BASE_SECONDS`（默认 `1.5`）
- `RETRY_MAX_SECONDS`（默认 `60`）
- `RETRY_JITTER_RATIO`（默认 `0.25`）
- `RETRY_429_MIN_DELAY_SECONDS`（默认 `8`）

### Qwen

- `LLM_API_KEY` 或 `QWEN_API_KEY`
- `LLM_BASE_URL` 或 `QWEN_BASE_URL`
- `LLM_MODEL` 或 `QWEN_MODEL`
- `QWEN_ENABLE_WEB_SEARCH`（默认 `true`）
- `QWEN_MAX_CANDIDATES`（默认 `3`）
- `QWEN_MIN_CONFIDENCE`（默认 `0.55`）
- `QWEN_MAX_OUTPUT_TOKENS`（默认 `256`）
- `QWEN_TIMEOUT_SECONDS`（默认 `120`）
- `QWEN_MIN_CALL_INTERVAL_SECONDS`（默认 `0`）
- `QWEN_SDK_MAX_RETRIES`（默认 `0`）

### 图片校验

- `ALLOWED_MIME`（默认 `image/jpeg,image/png,image/webp`）
- `MIN_IMAGE_EDGE_PX`（默认 `96`）

## CLI 运行

处理指定作者：

```bash
python3 main.py --author-id A5038153411
```

从文件批量处理：

```bash
python3 main.py --author-ids-file author_ids.json --workers 4
```

扫描数据库作者输入集合：

```bash
python3 main.py --author-limit 1000 --author-offset 0 --workers 4
```

## 设计原则与限制

- ORCID 是当前身份锚点；无 ORCID 时 Qwen 搜图阶段直接失败
- 默认只取首个候选进入图片验证与上传
- `source_pages` 仅用于审计透传，不参与抓取
- 项目不包含 profile page 二次抽图流程
- 主链候选仅来自 `web_search_image` tool output

