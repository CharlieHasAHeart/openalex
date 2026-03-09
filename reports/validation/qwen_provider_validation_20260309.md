# Qwen Provider Validation Report (2026-03-09)

## Scope
Validate post-`1fcaf6c` qwen-provider chain in `qwen/legacy/hybrid` modes, then run confidence-threshold experiments.

## Key finding
Primary blocker is API call layer: every qwen request failed with
`requests.sessions.Session.request() got multiple values for keyword argument 'timeout'`.

This happened before any response payload arrived, so schema validation and confidence threshold did not get a chance to execute.

## Three-mode summary (active runs)
| mode | authors | ok | no_image | ambiguous | invalid_image | error | fallback_count | qwen_schema_invalid | qwen_no_profile_pages | qwen_no_image_candidates | qwen_empty_filtered_candidates |
| ---- | ------- | -- | -------- | --------- | ------------- | ----- | -------------- | ------------------- | --------------------- | ------------------------ | ------------------------------ |
| qwen | 20 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 20 | 20 | 20 |
| hybrid | 20 | 0 | 20 | 0 | 0 | 0 | 20 | 0 | 20 | 20 | 20 |
| legacy | 20 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Additional observations:
- `qwen_request_started`: seen for all qwen/hybrid authors.
- `qwen_response_received`: 0
- `qwen_schema_validation_summary`: 0
- `avg final_candidate_count`: 0.0 for all three modes.
- Legacy produced `legacy_no_candidates` + `page_fetch_failed`.

## Threshold experiments (qwen-only, active 10-author runs)
| QWEN_MIN_CONFIDENCE | authors | ok | no_image | ambiguous | qwen_low_confidence_only | qwen_empty_filtered_candidates | avg final_candidate_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.35 | 10 | 0 | 10 | 0 | 0 | 10 | 0.0 |
| 0.45 | 10 | 0 | 10 | 0 | 0 | 10 | 0.0 |
| 0.55 | 10 | 0 | 10 | 0 | 0 | 10 | 0.0 |
| 0.65 | 10 | 0 | 10 | 0 | 0 | 10 | 0.0 |

Conclusion: lowering/raising confidence had no effect under current failure mode.

## Decision (rule-based)
Matched **Rule A**:
- Dominant reason: `qwen_request_failed`
- No successful qwen response decode, no schema-validation execution

Therefore next priority is **protocol/API call-layer fix**, not threshold/fallback tuning.

## Recommended next single theme
Fix qwen request path to remove duplicate timeout passing in `QwenToolsClient.search_author` and verify first successful `qwen_response_received` before any prompt/schema tuning.
