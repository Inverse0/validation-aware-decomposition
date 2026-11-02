# DecoVal

DecoVal validates SQL queries generated from natural-language questions. It decomposes a candidate query into smaller queries, or blocks, that recompose to the original, and a validator (here, an LLM) checks each block against its own sub-question. Just as a query optimizer chooses among equivalent plans by estimated execution cost, DecoVal chooses among decompositions by how reliably the validator is expected to check them.

This repository contains the code and data for the paper *Optimizing Query Decomposition for Reliable Validation*.

The benchmark consists of **305 candidate queries** that DIN-SQL generated for questions from BEAVER (270) and Spider 2.0 (35). Each candidate is labeled by its benchmark's official evaluation: **62 are correct and 243 incorrect**.

## Repository layout

| Path | Contents |
| --- | --- |
| [`sqldecomp/`](sqldecomp) | DecoVal: query import, decomposition space, reliability model, search, block printing, sub-questions and validation. |
| [`reproduction/`](reproduction) | Experiment runner, metrics and analysis. |
| [`nl2sql_verification/`](nl2sql_verification) | The RT and SE baselines of Alrashed et al. |
| [`prompts/`](prompts) | All LLM prompts, with a [prompt map](prompts/README.md). |
| [`benchmarks/`](benchmarks) | The 305 candidates with their schemas and labels ([benchmark guide](benchmarks/README.md)), and a toy example. |
| [`scripts/`](scripts) | Setup and run scripts. |
| [`experiment.py`](experiment.py) | Entry point for the experiments. |

## Setup

DecoVal needs Python 3.11 or later. The setup script creates a virtual environment in `.venv/`, installs DecoVal and the RT/SE baselines, and runs an offline smoke test:

```bash
bash scripts/setup.sh --baselines
source .venv/bin/activate
```

LLM calls go through the [Codex CLI](https://github.com/openai/codex). On Linux, each call runs in a [bubblewrap](https://github.com/containers/bubblewrap) sandbox that holds only its prompt, and the experiment commands require it. Install both, log in to Codex, and check the setup:

```bash
sudo apt-get install bubblewrap        # Debian/Ubuntu
codex login
export SQLDECOMP_CODEX="$(command -v codex)"
python experiment.py doctor --baselines
```

Without Linux (for example, on macOS), DecoVal still validates queries without the sandbox: give `python -m sqldecomp.pipeline` the `--no-sandbox` option (see [Trying DecoVal on your own queries](#trying-decoval-on-your-own-queries)).

## Checking the installation

```bash
python experiment.py smoke
```

This decomposes a toy query without any LLM call and checks that the printed blocks recompose to the original query.

## Reproducing the experiments

| Experiment | Command |
| --- | --- |
| RQ1: Comparison with validation baselines | `python experiment.py run --suite rq1 --out out/rq1` |
| RQ2: Comparison of decompositions | `python experiment.py run --suite rq2 --out out/rq2` |
| RQ3: Effect of granularity across validators | `python experiment.py run --suite rq3 --models v0 v1 v2 --out out/rq3` |

The validators are `v0` (GPT-5.5), `v1` (GPT-5.6 Sol, the default) and `v2` (GPT-6 Astra); `--models` also accepts any other model that Codex serves. `--jobs 8` validates eight candidates at a time, as in the paper. `--dry-run` prints the steps of a run without executing them, and `--resume` continues an interrupted run.

Results are in `tables/` under the output directory:

- `<model>_a<alpha>/tables.md`: TP, FN, FP, TN and the measures of each method;
- `granularity/tables.md`: DecoVal and Whole for every validator and α (RQ3);
- `structure/structure.json`: blocks per candidate and structural risk of each decomposition (RQ2);
- `cost-and-runtime/runs.md`: LLM calls and running time of each run.

The tables name the methods as follows:

| Tables | Paper |
| --- | --- |
| `opt` | DecoVal |
| `whole`, `whole_k` | Whole, Whole ×k |
| `rt`, `se` | RT, SE |
| `opt_rt`, `opt_se` | DecoVal + RT, DecoVal + SE |
| `structural` | DecoVal with γ = 0 |
| `syntactic`, `random_k` | Syntactic, Random-k |

The RQ1 analysis by quartile of structural risk, and DecoVal combined with Whole, RT or SE, come from the finished RQ1 run:

```bash
python experiment.py diagnostics --run out/rq1/runs/gpt-5.6-sol_a1/opt \
  --arm rt=out/rq1/runs/gpt-5.6-sol_a1/rt-se-arms/rt@direct \
  --arm se=out/rq1/runs/gpt-5.6-sol_a1/rt-se-arms/se@direct \
  --out out/rq1-diagnostics
```

## Trying DecoVal on your own queries

DecoVal reads a schema as JSON: each table with its columns, their types and nullability, and its primary, unique and foreign keys, as in [`benchmarks/toy/schema.json`](benchmarks/toy/schema.json) and the schemas in [`benchmarks/validation/schemas/`](benchmarks/validation/schemas).

To see how DecoVal decomposes a query (no LLM calls):

```bash
python -m sqldecomp.optimize --sql @benchmarks/toy/query.sql --schema benchmarks/toy/schema.json
```

It prints the blocks of the optimal decomposition under the structural risk model. `--alpha` scales the thresholds: `--alpha 0.5` gives smaller blocks and `--alpha 2` larger ones.

The parameters θ of the reliability model are options of this command and of the one below. `--tau` sets thresholds by feature, with the names of the paper's table of block features; for example, `--tau user-study,pred=4` changes one threshold of the paper's τ*. `--beta` and `--mu` set β and μ, `--gamma` sets γ when the semantic score is on (`--semantic judge`), and `--theta-json` reads all of them from a file. The experiment commands keep the paper's θ and scale its thresholds with `--alpha` (RQ1, RQ2) or `--alphas` (RQ3).

To validate a query against a question:

```bash
python -m sqldecomp.pipeline --verify --direct --semantic judge --model gpt-5.6-sol \
  --sql @benchmarks/toy/query.sql --schema benchmarks/toy/schema.json \
  --question "Which customers placed more than two orders, and how many orders did each place?" \
  --cache cache
```

It prints each block with its sub-question and verdict, DecoVal's verdict on the query, and the verdict of Whole (`--direct`). Without Linux, add `--no-sandbox`. For many queries, replace `--sql` and `--question` with `--queries queries.jsonl --out results`, where each line of `queries.jsonl` has `key`, `sql`, `question` and optionally `evidence`.

To run the paper's methods on your own labeled candidates, write them in the format of [`benchmarks/validation/dataset.jsonl`](benchmarks/validation/dataset.jsonl), put the schemas in a `schemas/` folder next to the file, and run, for example:

```bash
python experiment.py run --suite custom --method opt --dataset my_benchmark/dataset.jsonl --out out/mine
```

## Third-party code and data

BEAVER, Spider 2.0 and the RT/SE baselines keep their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
