# OpenAlex Avatar Pipeline

根据 `avatar_pipeline_spec.md` 实现的可运行版本：  
Top Works（时间窗 + 引用排序）-> 作者候选池去重/跳过 -> Wikidata(QID/P18) -> Commons imageinfo -> OSS -> PostgreSQL `openalex.authors_avatars` UPSERT。

## 1. 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

## 2. 必需环境变量

PostgreSQL:
- `PGHOST`
- `PGPORT`
- `PGDATABASE`
- `PGUSER`
- `PGPASSWORD`
- `PGSSLMODE`（可选）

OSS:
- `ALIYUN_OSS_ACCESS_KEY_ID`
- `ALIYUN_OSS_ACCESS_KEY_SECRET`
- `ALIYUN_OSS_BUCKET`
- `ALIYUN_OSS_ENDPOINT`
- `ALIYUN_OSS_PUBLIC_BASE_URL`
- `ALIYUN_OSS_KEY_PREFIX`
- `ALIYUN_OSS_CACHE_CONTROL`（可选）

图片/任务策略（可选，带默认值）:
- `AVATAR_THUMB_WIDTH=512`
- `ALLOWED_MIME=image/jpeg,image/png,image/webp`
- `MIN_IMAGE_EDGE_PX=200`（当前版本已不用于拦截，仅保留兼容配置）
- `REQUEST_TIMEOUT_SECONDS=20`
- `MAX_RETRIES=3`
- `GLOBAL_QPS_LIMIT=2`
- `ALLOW_NAME_FALLBACK=false`
- `OPENALEX_MAILTO`（建议填）
- `WORKS_WINDOW_YEARS=5`
- `WORKS_TOP_N=2000`
- `WORKS_PER_PAGE=200`
- `REFRESH_OK_DAYS=90`
- `REFRESH_NO_IMAGE_DAYS=30`
- `REFRESH_ERROR_DAYS=1`
- `REFRESH_AMBIGUOUS_DAYS=90`
- `REFRESH_NO_MATCH_DAYS=90`

## 3. 运行

先做依赖连通性检查（OSS/OpenAlex/PostgreSQL/Wikidata）。
其中 OSS 会执行 `list + 临时上传 + 删除`，可提前发现“只读可用、写入不可用”：

```bash
python3 scripts/check_connectivity.py
```

如果要补录历史 `invalid_image_too_small*` 作者（取消尺寸限制后重跑）：

```bash
python3 scripts/recover_invalid_image_too_small.py --source csv --csv-path logs/status_reason_details_full.csv
```

也可直接从日志提取全部 `status=invalid_image` 作者重跑：

```bash
python3 scripts/recover_invalid_image_too_small.py --source log --log-path logs/systemd_pipeline.log
```

单个作者：

```bash
python3 main.py --author-id https://openalex.org/A1969205038
```

批量默认模式（Top works -> 作者候选池）：

```bash
python3 main.py
```

批量覆盖参数：

```bash
python3 main.py --top-n 2000 --window-years 5 --per-page 200
python3 main.py --window-start-date 2021-01-01 --window-end-date 2026-01-01 --top-n 2000
python3 main.py --window-years 5 --top-offset 2000 --top-n 2000 --per-page 200
```

## 4. 状态码

- `ok`
- `no_match`
- `ambiguous`
- `no_image`
- `invalid_image`
- `error`

UPSERT 规则：`status != ok` 时不覆盖已有 `oss_object_key/oss_url/content_sha256`。

跳过规则（基于 `openalex.authors_avatars.status` + `updated_at`）：
- `ok`：90 天内跳过
- `no_image`：30 天内跳过
- `error`：1 天内跳过
- `ambiguous` / `no_match`：90 天内跳过
