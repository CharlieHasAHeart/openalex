# Avatar Pipeline README

This package contains the author-avatar pipeline and the benchmark toolchain around it.

It currently supports:
- running the avatar pipeline against `authors_analysis`
- writing production avatar results to `public.authors_avatars`
- writing run/process metadata to `openalex.*` pipeline tables
- building stratified sampling sets for benchmark design
- building benchmark annotation packages
- generating rule-based pre-annotations for human review
- exporting evaluation summaries and benchmark reports

## 1. High-Level Architecture

Core runtime flow:
1. Load author rows from `authors_analysis`.
2. Discover author image candidates through provider orchestration (`qwen` / `legacy` / `hybrid`).
3. In `qwen`/`hybrid`, use qwen3.5-plus with `web_search + t2i_search` as primary candidate discovery.
4. In `legacy`/fallback, use DuckDuckGo HTML + profile-page extraction.
5. Enrich candidates (page context, image metadata), cluster/dedupe, pre-rank.
6. Use `LlmMatcher` to choose candidate.
7. Apply final decision gating and review metadata.
8. Upload accepted image to OSS and upsert production result.
9. Persist run, candidate, and decision metadata to `openalex` schema.

Data/benchmark flow:
1. Build stratified sampling sets (`development`, `frozen_eval`, `shadow`).
2. Build annotation-ready benchmark packages from sampling sets.
3. Generate pre-annotations (assistant suggestions, not final truth).
4. Join results and human labels to compute benchmark metrics.

## 2. Module Map

- `config.py`: env loading and `PipelineConfig`.
- `http.py`: retry-aware HTTP client and global QPS limiter.
- `web_search_client.py`: search orchestrator + provider dispatch + candidate enrich + clustering.
- `qwen_tools.py`: qwen3.5-plus tool call wrapper (`web_search`, `t2i_search`, structured JSON parsing).
- `llm_matcher.py`: heuristic/LLM candidate selection.
- `pipeline_runner.py`: per-author orchestration, gating, persistence calls.
- `avatar_gate.py`: image mime validation and extension mapping.
- `oss_uploader.py`: OSS key generation and upload.
- `pg_repository.py`: database read/write repository.
- `stratified_sampling.py`: author sampling label/bucket generation and stratified set export.
- `benchmark_package.py`: package/template/review-sheet generation and annotation validation.
- `preannotation.py`: evidence-aware rule-based pre-annotation helper.
- `benchmark.py`: benchmark join, metrics, error buckets, sampling utilities.
- `evaluation.py`: summary and stratified sampling over recent decision rows.

## 3. Database Contracts

Production result table:
- `public.authors_avatars`
  - final avatar artifacts only (`author_id`, `oss_url`, `oss_object_key`, `content_sha256`, etc.)

Pipeline run/process tables:
- `openalex.avatar_pipeline_runs`
- `openalex.avatar_pipeline_author_runs`
- `openalex.avatar_candidate_images`
- `openalex.avatar_candidate_decisions`

Author/topic source tables used by sampler:
- `public.authors_analysis`
- `public.author_topic`
- `public.topics`

## 4. Required Environment Variables

Minimum required to run `main.py`:
- Postgres: `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`, `PGSSLMODE`
- OSS: `ALIYUN_OSS_ACCESS_KEY_ID`, `ALIYUN_OSS_ACCESS_KEY_SECRET`, `ALIYUN_OSS_BUCKET`, `ALIYUN_OSS_ENDPOINT`, `ALIYUN_OSS_PUBLIC_BASE_URL`
- LLM (optional but recommended): `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`

Important defaults in `config.py`:
- default proxy fallback: `http://127.0.0.1:7890` if no proxy env exists
- allowed mime: `image/jpeg,image/png,image/webp`
- web search max: `WEBSEARCH_MAX_RESULTS` (default `8`)
- search provider mode: `SEARCH_PROVIDER` (`qwen` / `legacy` / `hybrid`, default `hybrid`)
- qwen compatible-mode base URL: `QWEN_BASE_URL` (default `https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1`)
- qwen response endpoint path: `QWEN_RESPONSE_PATH` (default `/responses`)
- qwen model: `QWEN_MODEL` (default `qwen3.5-plus`)
- qwen `web_search` switch: `QWEN_ENABLE_WEB_SEARCH` (default `true`)
- qwen `t2i_search` switch: `QWEN_ENABLE_T2I_SEARCH` (default `true`)
- qwen candidate cap: `QWEN_MAX_CANDIDATES` (default `8`)
- qwen confidence floor: `QWEN_MIN_CONFIDENCE` (default `0.55`)
- qwen timeout: `QWEN_TIMEOUT_SECONDS` (default `30`)
- person-page discovery queries per author: `PERSON_PAGE_QUERY_MAX` (default `7`)
- per-query search result cap: `PERSON_PAGE_PER_QUERY_RESULTS` (default `5`)
- max profile pages fetched per author: `PERSON_PAGE_MAX_FETCH` (default `12`)
- structured image precheck threshold: `PROFILE_IMAGE_SCORE_THRESHOLD` (default `0.35`)

## 5. CLI Usage

All commands are executed through `main.py`.

### 5.1 Run avatar pipeline

```bash
python3 main.py --author-limit 100 --workers 4
python3 main.py --author-id https://openalex.org/A1234567890
python3 main.py --author-ids-file reports/sampling/development_set.json --workers 4
```

### 5.2 Export review summaries

```bash
python3 main.py --export-review-summaries --review-summary-limit 200
python3 main.py --export-review-summaries --review-recommendation needs_review --sample-review
```

### 5.3 Build benchmark report from existing decision rows

```bash
python3 main.py \
  --export-benchmark-report \
  --benchmark-authors-file reports/sampling/development_set.json \
  --annotations-file reports/benchmark_packages/development_annotations_template.jsonl
```

### 5.4 Build stratified sampling sets

```bash
python3 main.py \
  --build-sampling-sets \
  --development-size 2000 \
  --frozen-eval-size 2000 \
  --shadow-size 10000 \
  --sampling-seed 42
```

Outputs:
- `reports/sampling/development_set.json`
- `reports/sampling/frozen_eval_set.json`
- `reports/sampling/shadow_set.json`
- `reports/sampling/sampling_summary.json`

### 5.5 Build benchmark package from sampling set

```bash
python3 main.py \
  --build-benchmark-package \
  --sampling-set-file reports/sampling/development_set.json \
  --benchmark-package-name development \
  --package-output-dir reports/benchmark_packages
```

Outputs:
- `<name>_package.json`
- `<name>_annotations_template.jsonl`
- `<name>_review_sheet.csv`

### 5.6 Validate annotation template/result file

```bash
python3 main.py \
  --validate-annotations-file reports/benchmark_packages/development_annotations_template.jsonl \
  --sampling-set-file reports/sampling/development_set.json
```

### 5.7 Build pre-annotations (assistant suggestions)

```bash
python3 main.py \
  --build-preannotations \
  --benchmark-package-file reports/benchmark_packages/development_package.json \
  --preannotation-output-dir reports/benchmark_packages
```

Outputs:
- `<name>_preannotations.json`
- `<name>_preannotation_review_sheet.csv`

Note:
- Pre-annotation is an assistant output only.
- Human labels remain final truth for benchmark.

## 6. Label and Annotation Conventions

Canonical annotation labels:
- `correct`
- `incorrect`
- `uncertain`
- `no_avatar_available`

Pre-annotation fields:
- `suggested_label`
- `suggested_reason`
- `prelabel_confidence`
- `needs_human_review`
- `prelabel_source`

## 7. Current Behavioral Notes

- `upsert_result()` writes production table only for `status == "ok"`.
- Skip logic uses latest run state from `openalex.avatar_pipeline_author_runs`, not production table.
- If benchmark package authors have no matching pipeline run evidence, pre-annotation will fall back to uncertain and mark low evidence.
- `avatar_gate.validate_image_candidate()` currently only validates mime type against allowed set.
- Web search path does not depend on SerpAPI. It uses DuckDuckGo HTML, then a person-page-first extraction strategy.
- Structured image extraction priority is: `JSON-LD Person.image` -> `og:image` -> profile/people/faculty blocks -> generic `<img>` fallback.
- Search/extraction emits staged logs for query counts, profile-page recall, per-page extraction buckets, and zero-candidate reasons (`no_search_results`, `no_profile_pages`, `no_structured_images`, `all_images_filtered`, `page_fetch_failed`, `parse_failed`).
- Candidate discovery now supports provider modes:
  - `qwen`: qwen3.5-plus `web_search + t2i_search` only
  - `legacy`: DuckDuckGo/profile extraction only
  - `hybrid`: qwen first, auto-fallback to legacy when candidate recall is insufficient
- Qwen provider now calls DashScope compatible-mode Response API (`/responses`) and applies schema validation on model output (`profile_pages`, `image_candidates`, `filtered_candidates`, `failure_reason`).
- Key diagnostic reasons include: `qwen_api_key_missing`, `qwen_request_failed`, `qwen_http_error`, `qwen_response_decode_failed`, `qwen_output_missing`, `qwen_output_not_json`, `qwen_schema_invalid`, `qwen_no_profile_pages`, `qwen_no_image_candidates`, `qwen_low_confidence_only`, `qwen_empty_filtered_candidates`, `fallback_to_legacy`, `legacy_no_candidates`.

## 9. Development Verification

```bash
# 1) Qwen only
SEARCH_PROVIDER=qwen python3 main.py --author-ids-file reports/sampling/development_set.json --author-limit 20 --workers 1 --log-level INFO

# 2) Hybrid (recommended)
SEARCH_PROVIDER=hybrid python3 main.py --author-ids-file reports/sampling/development_set.json --author-limit 20 --workers 1 --log-level INFO

# 3) Legacy baseline
SEARCH_PROVIDER=legacy python3 main.py --author-ids-file reports/sampling/development_set.json --author-limit 20 --workers 1 --log-level INFO
```

## 8. Typical End-to-End Benchmark Workflow

1. Build sampling sets.
2. Build benchmark packages (`development`, `frozen_eval`).
3. Run pipeline on benchmark authors to accumulate decision evidence.
4. Build pre-annotations.
5. Human reviewers fill `final_label`.
6. Export benchmark report and compare runs by stable metrics.
