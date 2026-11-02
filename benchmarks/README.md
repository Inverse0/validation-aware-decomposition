# Benchmark inputs

`validation/dataset.jsonl` is the fixed 305-candidate validation benchmark: 270 BEAVER Neutron candidates and 35 Spider 2.0 local-database candidates. The 19 schemas are under `validation/schemas/`. There are 62 officially correct and 243 incorrect candidates. The reference SQL and correctness labels are input ground truth, not validator outputs.

Each row contains `id`, `benchmark`, `source_split`, `benchmark_id`, `database`, `schema_file`, `question`, `evidence`, `generated_sql`, `gold_sql`, `official_correct`, and `official_measure`. `schema_file` is a portable basename. `reproduction.experiment.inputs` gives the pipeline only `key`, `sql`, `question`, and `evidence`. The schemas retain declared keys and nullability. The packaging strips 37 historical generation/review/runtime fields without changing retained field values or schema bytes.

Evaluation uses only the boolean `official_correct` ground-truth label. Missing or non-boolean official labels are errors; no alternative label source is supported.

The candidates were generated once with the Spider 2.0 DIN-SQL adaptation using GPT-6 Luna at max effort, against the full PostgreSQL schema, with evidence where provided. Re-generating SQL would produce a different benchmark; the validation experiments therefore start with these fixed candidates. Gold SQL is the PostgreSQL form used during dataset construction.

## Original data, if needed

The original database contents are unnecessary for the bundled validation experiments and are not redistributed. They are needed only to independently regenerate execution labels or construct new candidate datasets.

### BEAVER

Obtain the queries from [beaverbench/beaver-query](https://huggingface.co/datasets/beaverbench/beaver-query) and the table metadata/database archive from [beaverbench/beaver-table](https://huggingface.co/datasets/beaverbench/beaver-table). Accept the access conditions on Hugging Face and authenticate using your own account. The upstream [BEAVER repository](https://github.com/beaverbench/beaver) documents database setup and evaluation.

Pinned source revisions used to construct this benchmark:

- Query dataset: `3c8eb607ee2b7709ad4b1d53243e04af1b9eda4b`, `data/neutron-00000-of-00001.parquet`.
- Table dataset: `86f2c4ec45990ac30677dfdfec9826f28171000a`, `data/neutron-00000-of-00001.parquet` and `beaver_db.zip` (use its `neutron.sql` member).

For example, after installing `huggingface_hub` and authenticating outside this repository:

```bash
hf download beaverbench/beaver-query --repo-type dataset \
  --revision 3c8eb607ee2b7709ad4b1d53243e04af1b9eda4b \
  --include data/neutron-00000-of-00001.parquet --local-dir downloads/beaver-query
hf download beaverbench/beaver-table --repo-type dataset \
  --revision 86f2c4ec45990ac30677dfdfec9826f28171000a \
  --include beaver_db.zip data/neutron-00000-of-00001.parquet --local-dir downloads/beaver-table
```

BEAVER's source database is MySQL. The included schema/reference SQL represent the experiment's PostgreSQL port; they should not be executed against the original MySQL dump without dialect adaptation. BEAVER correctness is set equality between candidate and gold result rows.

### Spider 2.0

Clone [xlang-ai/Spider2](https://github.com/xlang-ai/Spider2) and select the source revision `cafb867` recorded by the experiment. Follow the [Spider 2.0-Lite local-database instructions](https://github.com/xlang-ai/Spider2/tree/main/spider2-lite): download the linked local database archive and place its SQLite files in `spider2-lite/resource/databases/spider2-localdb/`. Use the official evaluator in `spider2-lite/evaluation_suite/`; this artifact uses only local-database tasks, so BigQuery/Snowflake credentials are unnecessary. The source IDs in each bundled row map to the upstream tasks.

The released candidate SQL is PostgreSQL, so independent evaluation on upstream SQLite files also needs dialect adaptation. Do not replace the official Spider evaluator with naive row equality: task-specific result comparison settings are part of its ground truth.

## Other files

- `toy/`: synthetic schema and query for an offline mechanical smoke test.

BEAVER and Spider 2.0 inputs retain their upstream MIT terms. See [third-party notices](../THIRD_PARTY_NOTICES.md).
