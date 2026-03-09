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
2. Search and extract image candidates from web pages.
3. Enrich candidates (page context, image metadata), cluster/dedupe, pre-rank.
4. Use `LlmMatcher` to choose candidate.
5. Apply final decision gating and review metadata.
6. Upload accepted image to OSS and upsert production result.
7. Persist run, candidate, and decision metadata to `openalex` schema.

Data/benchmark flow:
1. Build stratified sampling sets (`development`, `frozen_eval`, `shadow`).
2. Build annotation-ready benchmark packages from sampling sets.
3. Generate pre-annotations (assistant suggestions, not final truth).
4. Join results and human labels to compute benchmark metrics.

## 2. Module Map

- `config.py`: env loading and `PipelineConfig`.
- `http.py`: retry-aware HTTP client and global QPS limiter.
- `web_search_client.py`: profile-page discovery, structured image extraction, candidate enrich + clustering.
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

## 5. CLI Usage

All commands are executed through `main.py`.

### 5.1 Run avatar pipeline

```bash
python3 main.py --author-limit 100 --workers 4
python3 main.py --author-id https://openalex.org/A1234567890
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

## 8. Typical End-to-End Benchmark Workflow

1. Build sampling sets.
2. Build benchmark packages (`development`, `frozen_eval`).
3. Run pipeline on benchmark authors to accumulate decision evidence.
4. Build pre-annotations.
5. Human reviewers fill `final_label`.
6. Export benchmark report and compare runs by stable metrics.

