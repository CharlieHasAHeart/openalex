# Authors & Topics Data Profile

## 1. Overview Summary
- `authors_analysis`: 309,858 rows (partitioned hash table on `id`); every row currently has `display_name`, `orcid`, and almost every row has `affiliations` and `last_known_institutions` metadata.
- `author_topic`: 1,548,680 associations covering exactly 309,858 authors and all 4,346 topics; nearly every author is mapped to five topics (min 1 / max 5, median/p75/p90/p99 all 5).
- `topics`: 4,346 entries with hierarchical labels (`domain`, `field`, `subfield`) and a `count` column that ranges from 1 to 1,785.
- Dominant research domains split roughly: Physical Sciences (104K authors), Health Sciences (104K), Life Sciences (75K), Social Sciences (26K).
- Landmark opportunities: domain buckets, institution metadata, and display name scripts offer immediate stratification signals; topic-count-per-author is uniform so it does not differentiate difficulty.

## 2. Table Structure Analysis
### `public.authors_analysis`
- Columns: `id (PK, bigint)`, `orcid`, `display_name`, author energetics (`works_count`, `cited_by_count`, h-index, i10-index, 2yr mean), `counts_by_year` (jsonb), `affiliations` (jsonb array), `last_known_institutions` (jsonb object), `updated_date`.
- Indexes: primary key on `id`, btree indexes on `display_name` and `orcid`.
- Partitioned with 128 hash partitions; each row is one author’s cleaned profile.

### `public.author_topic`
- Columns: `id (PK)`, `author_id`, `topic_id`; btree indexes on `author_id` and `topic_id` to support lookups.
- Partitioned hash table like `authors_analysis`; there are no weight/rank fields, only relation presence.
- Each author is tied to 1–5 topic rows (almost all 5).

### `public.topics`
- Columns: `id (PK)`, `count`, `display_name`, `field_display_name`, `domain_display_name`, `subfield_display_name`.
- Indexes: primary key on `id`, prefix index on `display_name` (`left(display_name,10)`).
- No explicit parent/child columns, but domain→field→subfield gives an implicit hierarchy; there are 26 distinct `field_display_name` values and 243 `subfield_display_name` entries.
- The `count` column provides a ready-made popularity proxy (min=1, max=1785, avg≈27).

## 3. `authors_analysis` Data Profile
- SQL snippet used for the coverage numbers:
```sql
SELECT COUNT(*) AS total, COUNT(display_name), COUNT(orcid), COUNT(affiliations), COUNT(last_known_institutions)
FROM public.authors_analysis;
```
- Query used: `SELECT COUNT(*) AS total, COUNT(display_name), COUNT(orcid), COUNT(affiliations), COUNT(last_known_institutions) FROM public.authors_analysis;`
  - Result: total=309,858; `display_name` and `orcid` cover 100%; `affiliations` non-null for 309,713 (~99.95%); `last_known_institutions` filled in 282,095 (~91%).
- Display-name quality:
  - Average length 14; min 3, max 65 characters.
  - ~39,057 names contain non-ASCII characters (~12.6%), so we can stratify by ASCII vs non-ASCII scripts.
- Sample `last_known_institutions` entries (from `SELECT id, last_known_institutions FROM public.authors_analysis LIMIT 3;`):
  - `{"display_name": "University of Vienna", "country_code": "AT", "type": "education", "ror": "..."}`
  - `{"display_name": "Trinity University", "country_code": "US", "type": "education", ...}`
  - `{"display_name": "Centre National de la Recherche Scientifique", "country_code": "FR", "type": "funder", ...}`
- `affiliations` is an array of institution history entries (typical fields: `years`, `institution_type`, `institution_country_code`, `institution_display_name`). Distribution of `institution_type`: education (3.7M entries), healthcare (1.4M), facility (900K), company (551K), nonprofit (348K).
- `affiliations` length percentiles (non-null rows): p50=16, p90=53, p99=145 entries, so some authors have hundreds of affiliations; these arrays often feed institutional sampling but need normalization.
- Country coverage (from `last_known_institutions->>'country_code'`): US (78,041), CN (41,039), blank/missing (28,265), GB (17,875), JP (14,038). A non-trivial slice (~9%) lacks country codes.

- SQL snippet for topic coverage:
```sql
SELECT COUNT(*) AS total, COUNT(DISTINCT author_id), COUNT(DISTINCT topic_id)
FROM public.author_topic;
```
- Query used: `SELECT COUNT(*) AS total, COUNT(DISTINCT author_id), COUNT(DISTINCT topic_id) FROM public.author_topic;` → (1,548,680 rows, 309,858 authors, 4,346 topics). There are zero authors without topic links.
- Topic-per-author distribution: 1 topic for 10 authors, 2 for 11, 3 for 200, 4 for 137, 5 for 309,500 → almost every author has 5 topics (avg ≈5, percentiles all 5).
- Distribution snippet:
```sql
SELECT cnt AS topics_per_author, COUNT(*) AS author_count
FROM (
  SELECT author_id, COUNT(*) AS cnt
  FROM public.author_topic
  GROUP BY author_id
) t
GROUP BY cnt
ORDER BY cnt;
```
- Sample author entry (SQL above) shows topics from the same domain (e.g., full Physical Sciences cluster) or mixed Health/Life Sciences.
- There is no notion of timestamp or confidence per association; the available leverage is topic identity and its domain/field.
- Considering this uniformity, “number of topics” is not a discriminating stratifier; instead, use the actual topic identities or domain labels.

- SQL snippet for domain breakdown:
```sql
SELECT domain_display_name, COUNT(*) AS topic_count
FROM public.topics
GROUP BY domain_display_name
ORDER BY topic_count DESC;
```
- Query used: `SELECT COUNT(*), MIN(count), MAX(count), AVG(count) FROM public.topics;` → (4,346 topics, count ∈ [1, 1,785], avg=27.7).
- Domain breakdown (`domain_display_name` counts): Physical Sciences 1,543 topics, Social Sciences 1,353, Health Sciences 841, Life Sciences 609.
- Field/Subfield cardinality: 26 unique `field_display_name` values, 243 `subfield_display_name` values.
- Top topics by `count` (popularity proxy): e.g., “Synthesis of heterocyclic compounds” (1,785), “Linguistic, Cultural, and Literary Studies” (929), “Intellectual Property and Patents” (797); these counts roughly mirror author coverage (e.g., “Genomics and Phylogenetic Studies” covers ~8,986 authors).
- No duplicate `display_name` values, so the table is clean; the implicit hierarchy is defined through the repeated use of domain/field/subfield columns, not via foreign keys.
- `count` is a lightweight indicator for rarity/abundance and can be used to create “common vs niche topic” buckets for sampling.

- Query that drives dominant-domain metrics:
```sql
WITH domain_counts AS (
  SELECT at.author_id, t.domain_display_name, COUNT(*) AS cnt
  FROM public.author_topic at
  JOIN public.topics t ON at.topic_id = t.id
  GROUP BY at.author_id, t.domain_display_name
),
best_domain AS (
  SELECT author_id, domain_display_name,
         ROW_NUMBER() OVER (PARTITION BY author_id ORDER BY cnt DESC, domain_display_name) AS rn
  FROM domain_counts
)
SELECT domain_display_name, COUNT(*) AS authors
FROM best_domain
WHERE rn = 1
GROUP BY domain_display_name
ORDER BY authors DESC;
```
- Coverage: Every `authors_analysis.id` has at least one `author_topic` row, so topic coverage is complete (zero authors without topics).
- Dominant domains (per-author majority topic domain): Physical Sciences (104,440 authors), Health Sciences (104,108), Life Sciences (75,010), Social Sciences (26,300). Query: `ROW_NUMBER()` over (domain counts) picks the top domain per author.
- Topic popularity (top 10 by author count) read from `author_topic`/`topics` join: e.g., “Genomics and Phylogenetic Studies” (8,986 authors), “Cancer Genomics and Diagnostics” (6,600), etc.
- Institutions vs topics: Top institutions by `last_known_institutions->>'display_name'` include CNRS (3,427 authors), UNC Chapel Hill (1,947), Inserm (1,113), Harvard (1,110). Domain breakdown for the top three institutions shows mixed domain presence (e.g., CNRS has about 2,218 Physical, 1,974 Life, 1,125 Health, 261 Social Science authors; “unknown” institutions map to ~14K Physical, 13K Health, 11K Life, 7.5K Social authors).
- Topic coverage by domain (join): Health (165,696 authors), Physical (154,736), Life (149,081), Social (61,966). This reinforces the four domain buckets for stratified sampling.

## 7. Testing/Stratification Recommendations
- **Primary strata (high confidence):** Dominant domain (`Physical`, `Health`, `Life`, `Social`) via per-author majority domain (counts above). This splits authors into four buckets of roughly 100K / 26K sizes and maps directly to `topics.domain_display_name`.
- **Secondary strata ideas:**
  1. `last_known_institutions->>'institution_type'` or the `affiliations` entries (`education`, `healthcare`, `facility`, etc.) because these fields are populated for nearly every author and cluster well.
  2. Institutional `country_code` (US, CN, GB, JP, and the large blank bucket) to ensure geographic diversity; fill-in rate ~90%.
  3. Display name script (ASCII vs non-ASCII) to surface language/script diversity (`~12.6% non-ASCII names`).
  4. Topic rarity via `topics.count`: define buckets such as common (`count > 500`, ~30 topics), medium (`count 50–500`, ~200 topics), rare (`count < 50`, remainder). Since `author_topic` is curated, mapping authors to rare vs common topics encourages coverage of niche profiles.
  5. Institution-affiliation length (e.g., authors with very long `affiliations` arrays likely have long careers/multiple roles) using percentiles (p50=16, p90=53, p99=145). Use these to isolate more senior/multi-affiliation authors from early-career ones.
- **Why ignore certain fields:**
  - Topic count per author is uniform (almost every author has 5 entries) so not useful for stratification.
  - ORCID presence is 100% so offers no stratification variance.
  - `counts_by_year` is a complex JSON requiring normalization; postpone until needed.
  - `author_topic` contains no score or timeframe; the `topic_id`s are static, so difficulty must be inferred through the topic's domain/popularity or by cross-joining with author metadata.
- **Proposed sampling buckets (example):**
  1. Domain bucket: {Physical, Health, Life, Social} (counts as above).
  2. Institution bucket: {education/healthcare vs facility/company} from affiliation types (tight coverage and easy to extract via `jsonb_array_elements`).
  3. Geography bucket: {US, CN, GB, others, unknown} using `last_known_institutions->>'country_code'` (covering ~90% of authors).
  4. Topic rarity bucket: {common, medium, rare} via `topics.count` thresholds. Use authors’ highest-count topic to assign them.
  5. Name script bucket: {ASCII-only, non-ASCII} to ensure page layout coverage for different scripts.
  Combined, a stratified design could draw e.g., 50% from the three largest domain buckets, 25% from less common domains, etc.; actual proportions depend on evaluation needs but have a strong empirical grounding.

## 8. Risks & Uncertainties
- `last_known_institutions` has ~28K blank entries for `country_code` and ~27.7K blank `display_name`s (empty string in the top-institution query). Any geography-based stratification must handle this “unknown” bucket explicitly.
- `affiliations` arrays can be extremely long (p99=145 entries), so normalizing institution/country/type requires aggregating or picking the most recent entry; otherwise, counting the same author multiple times is easy.
- `author_topic` lacks weights and is capped at five topics; thus, it cannot express confidence levels or emerging research directions without additional signals.
- `topics.count` is a popularity proxy; if it is not updated frequently, “rare” topics might be artificially inflated or deflated, so combine it with author-level data (e.g., how many unique authors currently share that topic) for robustness.
- About 12.6% of names contain non-ASCII characters; any downstream string matching or sampling should treat these as a separate bucket to avoid bias.

## 9. Suggested Next Steps
1. Build utility queries (or a Jupyter notebook) that extract the “most recent institution” from `affiliations` (e.g., the entry with the highest `years` value) so that institutions/ countries can be normalized for sampling.
2. Precompute per-author “topic rarity score” by joining `author_topic.topic_id` with `topics.count` and taking the median for the five topics; use this to label authors as “niche” vs “core”.
3. Enhance the benchmark harness to annotate each author with the proposed buckets (domain, institution type, country, topic rarity, name script) so that sampling can be stratified programmatically.
4. Monitor the ~9% of authors missing `country_code` and consider fallback attributes (e.g., using `affiliations->>'institution_country_code'` from the first non-null entry) to avoid losing them in geography-based samples.

*All queries listed above were executed against `public.authors_analysis`, `public.author_topic`, and `public.topics` on `openalex_fetch` (PostgreSQL). Please rerun or adapt them if the schema changes.*
