# Benchmark

`validation/dataset.jsonl` holds the 305 candidate queries of the paper, one JSON object per line: 270 from BEAVER (the Neutron split) and 35 from Spider 2.0 (tasks on local databases). DIN-SQL with GPT-6 Luna generated each candidate from the question, its evidence and the database schema. By their benchmark's official evaluation, 62 candidates are correct and 243 incorrect.

| Field | Contents |
| --- | --- |
| `id` | The candidate's ID, such as `beaver:neutron_494` |
| `benchmark`, `source_split`, `benchmark_id` | The source benchmark, its split, and the task's ID there |
| `database`, `schema_file` | The database, and its schema in `validation/schemas/` |
| `question`, `evidence` | The question and the information given with it (empty when there is none) |
| `generated_sql` | The candidate query |
| `gold_sql` | The reference query |
| `official_correct` | Whether the candidate passes the official evaluation |
| `official_measure` | How the official evaluation decides it |

`validation/schemas/` holds the schemas of the 19 databases: each table with its columns, their types and nullability, and its primary, unique and foreign keys. All SQL is in PostgreSQL. `toy/` holds the query and schema of the smoke test.

## Sources

The BEAVER rows come from [beaverbench/beaver-query](https://huggingface.co/datasets/beaverbench/beaver-query) and [beaverbench/beaver-table](https://huggingface.co/datasets/beaverbench/beaver-table); a BEAVER candidate is correct if it returns the same set of rows as the gold query. The Spider 2.0 rows come from [xlang-ai/Spider2](https://github.com/xlang-ai/Spider2) at revision `cafb867`; a Spider 2.0 candidate is correct if the benchmark's evaluator accepts its result. Both benchmarks are under the MIT license; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

The experiments do not need the original databases, so their contents are not included. To run the queries on the original data, download BEAVER's datasets (accept their conditions on Hugging Face and log in first; the database is `neutron.sql` in `beaver_db.zip`):

```bash
hf download beaverbench/beaver-query --repo-type dataset \
  --revision 3c8eb607ee2b7709ad4b1d53243e04af1b9eda4b \
  --include data/neutron-00000-of-00001.parquet --local-dir downloads/beaver-query
hf download beaverbench/beaver-table --repo-type dataset \
  --revision 86f2c4ec45990ac30677dfdfec9826f28171000a \
  --include beaver_db.zip data/neutron-00000-of-00001.parquet --local-dir downloads/beaver-table
```

For Spider 2.0, follow the local-database instructions of [Spider 2.0-Lite](https://github.com/xlang-ai/Spider2/tree/main/spider2-lite). The original databases are MySQL (BEAVER) and SQLite (Spider 2.0).
