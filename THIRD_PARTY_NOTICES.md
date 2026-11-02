# Third-party notices

## BEAVER inputs

Questions, evidence, reference SQL and schema information in the BEAVER rows derive from [beaverbench/beaver-query](https://huggingface.co/datasets/beaverbench/beaver-query) and [beaverbench/beaver-table](https://huggingface.co/datasets/beaverbench/beaver-table). The query dataset declares the MIT license. Source revisions and acquisition steps are in `benchmarks/README.md`. Credit: Peter Baile Chen and the BEAVER authors, *BEAVER: An Enterprise Benchmark for Text-to-SQL*, arXiv:2409.02038. The BEAVER source's copyright/permission notice is retained in `licenses/BEAVER-MIT.txt`.

## Spider 2.0 inputs

The Spider 2.0 rows derive from [xlang-ai/Spider2](https://github.com/xlang-ai/Spider2), source revision `cafb867`, under the MIT license. Its notice is in `licenses/Spider2-MIT.txt`. Credit: Fangyu Lei and the Spider 2.0 authors, *Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows*. The dataset has been subsetted, its schema/reference SQL ported to PostgreSQL, and candidate SQL added by this project.

## RT/SE baseline code and prompts

`nl2sql_verification/` and `prompts/alrashed/` derive from the artifact *Text-to-SQL Verification Methods and Benchmark*, by Tarfah Alrashed, Madhup Sukoon, David R. Karger and Natasha Noy, accompanying *Developing and Benchmarking Verification Algorithms to Improve Text-to-SQL Generation*, PVLDB 19(11), 3732–3744 (2026), DOI [10.14778/3836663.3836721](https://doi.org/10.14778/3836663.3836721).

Source artifact: [figshare](https://figshare.com/s/c88a751bc29d9751816b), file `source_code_and_benchmark.zip`, SHA-256 `31c537f18cd95648be0d202b6ff2a7979bca92475b8e377e7216f004d22f6e14`. The source repository records its license as [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/); the full [legal code](https://creativecommons.org/licenses/by/4.0/legalcode) applies to these files.

Modifications for this release: only RT/SE and their dependencies are retained; templates were moved to `prompts/alrashed/` and re-exported at the original import locations; unused few-shot examples were replaced by empty registries because the reported settings are zero-shot. The project adapters replace the model client with the sandboxed CLI client, convert schema/prompt inputs, execute candidate SQL in DuckDB, and aggregate block verdicts. No upstream benchmark results or settings file is included. These modifications do not imply endorsement by the baseline authors.

## Project code

No new project-wide open-source license is assigned by this packaging step. The third-party permissions above apply to their respective materials. Dependencies installed from Python packages retain their own licenses; they are not vendored here.
