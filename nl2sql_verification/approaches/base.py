"""
Implements the base class for approaches
"""

# pylint: disable=broad-exception-caught

import os
import json
import asyncio
import traceback
from uuid import uuid4
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Self, Type, TypeVar, Generic, Optional, Tuple, Literal, get_args

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm
from smart_open import smart_open
from cloudpathlib import CloudPath
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ..config import BASE_CONFIG


class ColumnDict(TypedDict):
    """
    Define the type for an object in the 'columns' list
    """

    name: str
    type: str


class ForeignKeyDict(TypedDict):
    """
    Define the type for an object in the 'foreign_keys' list
    """

    column: str
    references_table: str
    references_column: str


class TableDict(TypedDict):
    """
    Define the type for an object in the 'tables' list
    """

    table_name: str
    columns: list[ColumnDict]
    primary_keys: list[str]
    foreign_keys: list[ForeignKeyDict]


class DatabaseSchemaDict(TypedDict):
    """
    Define the top-level type for the entire JSON object
    """

    db_id: str
    tables: list[TableDict]


class BaseQuestion(BaseModel, ABC):
    """
    Schema of a question
    """

    uuid: str = Field(default_factory=lambda: str(uuid4()))
    additional_information: Optional[str] = None
    db_schema: Optional[DatabaseSchemaDict] = None
    db_create_statements: Optional[list[str]] = None


class BaseAnswer(BaseModel, ABC):
    """
    Schema of an Answer
    """

    uuid: str = Field(default_factory=lambda: str(uuid4()))


Q = TypeVar("Q", bound=BaseQuestion)
A = TypeVar("A", bound=BaseAnswer)


class BaseExample(BaseModel, Generic[Q, A], ABC):
    """
    Base Schema of a single example
    """

    question: Q
    answers: list[A]


# Generation Question and Answer
GQ = TypeVar("GQ", bound=BaseQuestion)
GA = TypeVar("GA", bound=BaseAnswer)

# Evaluation Question and Answer
EQ = TypeVar("EQ", bound=BaseQuestion)
EA = TypeVar("EA", bound=BaseAnswer)


class BaseExperimentSpec(BaseModel, ABC):
    """
    Schema of an experiment spec
    """

    uuid: str = Field(default_factory=lambda: str(uuid4()))
    data_sample_rows: int = 0
    description: str

    @classmethod
    @abstractmethod
    def parser(cls, exp_params: dict) -> Self:
        """
        Parse an experiment spec
        """
        raise NotImplementedError


EXP = TypeVar("EXP", bound=BaseExperimentSpec)


class BaseMetadata(TypedDict):
    """
    A TypedDict for the generate / evaluate operation's Metadata
    """


GENMETA = TypeVar("GENMETA", bound=BaseMetadata)
EVALMETA = TypeVar("EVALMETA", bound=BaseMetadata)


class BaseApproach(BaseModel, ABC, Generic[GQ, GA, GENMETA, EQ, EA, EVALMETA, EXP]):
    """
    Base class for NL2SQL Verification Approaches
    """

    uuid: str = Field(default_factory=lambda: str(uuid4()))
    implementation_version: str = "v5"
    exp_class_placeholder: Optional[Type[EXP]] = Field(exclude=True, default=None)

    @classmethod
    def parse_experiment_spec(cls, exp_spec: dict) -> EXP:
        """
        Fetches the parser from the experiment spec class
        """
        type_arg, _ = get_args(cls.model_fields["exp_class_placeholder"].annotation)
        (exp_arg,) = get_args(type_arg)
        return exp_arg.parser(exp_spec)

    @abstractmethod
    async def generate(
        self,
        input_row: pd.Series,
        experiment_spec: EXP,
        *args,
        **kwargs,
    ) -> list[Tuple[GQ, GA, GENMETA]]:
        """
        Generate potential answer(s)
        """
        raise NotImplementedError

    @abstractmethod
    async def evaluate(
        self,
        input_row: pd.Series,
        experiment_spec: EXP,
        generate_question: GQ,
        generate_answer: GA,
        *args,
        **kwargs,
    ) -> list[Tuple[EQ, EA, EVALMETA]]:
        """
        Evaluate a generated answer
        """
        raise NotImplementedError

    async def experiment(
        self,
        input_row: pd.Series,
        experiment_spec: EXP,
        *args,
        **kwargs,
    ) -> list[
        Tuple[
            pd.Series,
            Optional[GQ],
            Optional[GA],
            Optional[GENMETA],
            Optional[EQ],
            Optional[EA],
            Optional[EVALMETA],
            Optional[str],
        ]
    ]:
        """
        Run (Generate & Evaluate) a single experiment
        """

        results: list[
            Tuple[
                pd.Series,
                Optional[GQ],
                Optional[GA],
                Optional[GENMETA],
                Optional[EQ],
                Optional[EA],
                Optional[EVALMETA],
                Optional[str],
            ]
        ] = []
        try:
            logger.trace(
                f"Awaiting Generator {input_row['uuid']} for experiment {experiment_spec.uuid}"
            )
            generations = await self.generate(
                input_row, experiment_spec, *args, **kwargs
            )
        except Exception as e1:
            tb = traceback.format_exc()
            logger.exception(f"[Row {input_row['uuid']}] Generation Failure : {e1}")
            results.append(
                (
                    input_row,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    f"Generation Failure : {e1}\n{tb}",
                )
            )
        else:
            logger.trace(
                f"Received Generator output for row {input_row['uuid']}"
                f" for experiment {experiment_spec.uuid}"
            )
            for gen_q, gen_a, gen_metadata in generations:
                try:
                    logger.trace(
                        f"Awaiting evaluator for row {input_row['uuid']}"
                        f" for experiment {experiment_spec.uuid}"
                    )
                    evaluations = await self.evaluate(
                        input_row, experiment_spec, gen_q, gen_a, *args, **kwargs
                    )
                except Exception as e2:
                    tb = traceback.format_exc()
                    logger.exception(
                        f"[Row {input_row['uuid']}] Evaluation Failure : {e2}"
                    )
                    results.append(
                        (
                            input_row,
                            gen_q,
                            gen_a,
                            gen_metadata,
                            None,
                            None,
                            None,
                            f"Evaluation Failure : {e2}\n{tb}",
                        )
                    )
                else:
                    logger.trace(
                        f"Received evaluator output for row {input_row['uuid']}"
                        f" for experiment {experiment_spec.uuid}"
                    )
                    for eval_q, eval_a, eval_metadata in evaluations:
                        results.append(
                            (
                                input_row,
                                gen_q,
                                gen_a,
                                gen_metadata,
                                eval_q,
                                eval_a,
                                eval_metadata,
                                None,
                            )
                        )

        return results

    async def run_pipeline(
        self,
        input_paths: list[str],
        output_path: str,
        experiment_specs: Optional[list[dict]] = None,
        experiment_specs_path: Optional[str] = None,
        ignore_invalid_experiments: bool = False,
        input_type: Literal["csv", "json"] = "json",
    ) -> Tuple[str, str]:
        """
        Runs a set of experiments in parallel
        """

        logged_errors = []

        if experiment_specs is None:
            if experiment_specs_path is None:
                raise ValueError(
                    "Either `experiment_specs` or `experiment_specs_path` must be provided."
                )
            logger.info(f"Reading experiment specs from {experiment_specs_path}")
            experiment_specs = json.loads(
                "\n".join(smart_open(experiment_specs_path, "r"))
            )

        assert experiment_specs, "No experiment specs provided."

        validated_experiments: list[EXP] = []
        for exp_spec in experiment_specs:
            try:
                es = self.parse_experiment_spec(exp_spec)
            except Exception as e:
                if ignore_invalid_experiments:
                    logger.warning(f"Skipping Experiment : {e}")
                else:
                    raise e
            else:
                validated_experiments.append(es)

        if not validated_experiments:
            raise RuntimeError("No Valid Experiments to run.")

        reader = pd.read_csv if input_type == "csv" else pd.read_json
        full_dataset = pd.concat(
            [reader(smart_open(path)).assign(input_path=path) for path in input_paths]
        ).replace({np.nan: None})

        pipeline_id = os.path.join(
            self.__class__.__name__,
            self.implementation_version,
        )

        if self.uuid:
            pipeline_id = os.path.join(pipeline_id, self.uuid)

        output_path = os.path.join(
            output_path,
            pipeline_id,
        )

        logger.info(
            f"Building Pipeline : {pipeline_id} with {len(validated_experiments)} experiment(s)."
        )

        for experiment_spec in tqdm(validated_experiments, desc="Running Experiments"):
            results = []
            experiment_id = experiment_spec.uuid
            sampled_dataset = (
                full_dataset.sample(experiment_spec.data_sample_rows, random_state=42)
                if experiment_spec.data_sample_rows > 0
                else full_dataset
            ).reset_index(drop=True)
            if BASE_CONFIG.DATASET_BATCHING_ROWS > 0:
                datasets = [
                    sampled_dataset.iloc[i : i + BASE_CONFIG.DATASET_BATCHING_ROWS]
                    for i in range(
                        0, len(sampled_dataset), BASE_CONFIG.DATASET_BATCHING_ROWS
                    )
                ]
            else:
                datasets = [sampled_dataset]
            for idx, dataset in enumerate(tqdm(datasets)):
                tasks = []
                batch_results = []
                for _, row in dataset.iterrows():
                    output_file = CloudPath(  # type: ignore[abstract]
                        os.path.join(
                            output_path,
                            "raw",
                            f"experiment-{experiment_id}",
                            f"batch-{idx}",
                            f"row-{row['uuid']}.jsonl",
                        )
                    )

                    if output_file.exists():
                        batch_results.extend(
                            [
                                json.loads(fileline)
                                for fileline in output_file.read_text().splitlines()
                            ]
                        )
                    else:
                        exp = self.experiment(
                            row,
                            experiment_spec,
                        )
                        tasks.append(
                            (
                                asyncio.create_task(exp)
                                if BASE_CONFIG.EXPERIMENT_EXECUTION_MODE == "PARALLEL"
                                else exp
                            )
                        )
                logger.info(
                    f"{len(batch_results)} experiments recovered, "
                    f"scheduling {len(tasks)} new experiments."
                )
                for task_id, future in enumerate(
                    tqdm(
                        (
                            asyncio.as_completed(tasks)
                            if BASE_CONFIG.EXPERIMENT_EXECUTION_MODE == "PARALLEL"
                            else tasks
                        ),
                        total=len(tasks),
                        desc=f"{experiment_spec.description} - Batch {idx + 1}/{len(datasets)}",
                    )
                ):
                    experiment_result_dict = {
                        "pipeline__uuid": pipeline_id,
                        "experiment__uuid": experiment_id,
                        "experiment__task_uuid": task_id,
                        **{
                            f"experiment__spec__{k}": v
                            for k, v in experiment_spec.model_dump().items()
                        },
                    }

                    result = defaultdict(list)
                    try:
                        logger.trace(
                            f"Awaiting Task {task_id} for experiment {experiment_spec.uuid}"
                        )
                        outputs = await future
                    except Exception as e:
                        logger.warning(
                            f"Task {task_id} failed for {experiment_spec} : {e}"
                        )
                        result["experiment_errors"].append(
                            {**experiment_result_dict, "error_message": str(e)}
                        )
                    else:
                        logger.trace(
                            f"Received Task {task_id} for experiment"
                            f" {experiment_spec.uuid}. Parsing"
                        )
                        for (
                            input_row,
                            gen_q,
                            gen_a,
                            gen_metadata,
                            eval_q,
                            eval_a,
                            eval_metadata,
                            error_msg,
                        ) in outputs:
                            flattened_results = {
                                **{
                                    f"input__{k}": v
                                    for k, v in input_row.to_dict().items()
                                },
                                **{
                                    f"output__{res_key}__{k}": v
                                    for res_key, res_obj in [
                                        ("generation_question", gen_q),
                                        ("generation_answer", gen_a),
                                        ("evaluation_question", eval_q),
                                        ("evaluation_answer", eval_a),
                                    ]
                                    for k, v in (
                                        res_obj.model_dump() if res_obj else {}
                                    ).items()
                                },
                            }
                            if gen_metadata:
                                flattened_results.update(
                                    {
                                        f"output__generation_metadata__{k}": v
                                        for k, v in gen_metadata.items()
                                    }
                                )
                            if eval_metadata:
                                flattened_results.update(
                                    {
                                        f"output__evaluation_metadata__{k}": v
                                        for k, v in eval_metadata.items()
                                    }
                                )
                            flattened_results["error_message"] = error_msg
                            result[input_row["uuid"]].append(
                                {**experiment_result_dict, **flattened_results}
                            )
                            if error_msg:
                                logged_errors.append(
                                    {
                                        "input": input_row.to_dict(),
                                        "experiment_spec": experiment_spec.model_dump(),
                                        "error_message": error_msg,
                                    }
                                )
                    finally:
                        for res_key, res_val in result.items():
                            batch_results.extend(res_val)
                            CloudPath(  # type: ignore[abstract]
                                os.path.join(
                                    output_path,
                                    "raw",
                                    f"experiment-{experiment_id}",
                                    f"batch-{idx}",
                                    f"row-{res_key}.jsonl",
                                )
                            ).write_text(
                                (
                                    "\n".join(
                                        [json.dumps(rv, default=str) for rv in res_val]
                                    )
                                )
                                + "\n"
                            )
                results.extend(batch_results)
                CloudPath(  # type: ignore[abstract]
                    os.path.join(
                        output_path,
                        "batched",
                        f"experiment-{experiment_id}",
                        f"batch-{idx}.jsonl",
                    )
                ).write_text(
                    ("\n".join([json.dumps(br, default=str) for br in batch_results]))
                    + "\n"
                )

            results_df = pd.DataFrame(results)
            results_df.to_csv(
                os.path.join(
                    output_path, "collated", f"experiment-{experiment_id}.csv"
                ),
                index=False,
            )
            results_df.to_json(
                os.path.join(
                    output_path, "collated", f"experiment-{experiment_id}.json"
                ),
                orient="records",
                indent=4,
                default_handler=str,
            )
            results_df.to_json(
                os.path.join(
                    output_path, "collated", f"experiment-{experiment_id}.jsonl"
                ),
                orient="records",
                lines=True,
                default_handler=str,
            )

        return output_path, json.dumps(logged_errors, default=str, indent=4)
