"""
Implements the DataGen approach to NL2SQL Verification
"""

# pylint: disable=unused-argument

import json
import time
from typing import Tuple, Optional, cast

import pandas as pd
from google import genai
from pandasql import sqldf
from jinja2 import Environment
from tenacity import (
    AsyncRetrying,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_not_exception_type,
)

from ...config import BASE_CONFIG
from ..base import BaseApproach
from .assets.schema import (
    DATATYPE_MAP,
    GenQuestion,
    GenAnswer,
    GenMetadata,
    DataGenQuestion,
    DataGenAnswer,
    DataGenMetadata,
    AnswerGenQuestion,
    AnswerGenAnswer,
    AnswerGenMetadata,
    EvalQuestion,
    EvalAnswer,
    EvalMetadata,
    ExperimentSpec,
)

jinja_env = Environment()
jinja_env.policies["json.dumps_kwargs"]["sort_keys"] = False


class NonRetryableError(Exception):
    """
    Raised when the error should not be retried
    """


class DataGen(
    BaseApproach[
        GenQuestion,
        GenAnswer,
        GenMetadata,
        EvalQuestion,
        EvalAnswer,
        EvalMetadata,
        ExperimentSpec,
    ]
):
    """
    DataGen Verification
    """

    async def data_generator(
        self,
        question: DataGenQuestion,
        experiment_spec: ExperimentSpec,
    ) -> list[tuple[DataGenAnswer, DataGenMetadata]]:
        """
        Tries to generate synthetic data
        """

        data_gen_schema = {
            "type": "object",
            "title": "Generated Data",
            "properties": {
                "analyze_nlq": {
                    "type": "string",
                },
                "plan": {
                    "type": "string",
                },
                "synthesize": {
                    "type": "string",
                },
                "focus_area_description": {
                    "type": "string",
                },
                "generated_data": {
                    "type": "object",
                    "title": "Generated Data",
                    "properties": {
                        t1["table_name"]: {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    c1["name"]: DATATYPE_MAP.get(
                                        c1["type"], {"type": "string"}
                                    )
                                    for c1 in t1["columns"]
                                },
                                "required": [c2["name"] for c2 in t1["columns"]],
                            },
                        }
                        for t1 in question.db_schema["tables"]
                    },
                    "required": [
                        t2["table_name"] for t2 in question.db_schema["tables"]
                    ],
                },
                "expected_answer": {
                    "type": "string",
                },
            },
            "required": [
                "analyze_nlq",
                "plan",
                "synthesize",
                "focus_area_description",
                "generated_data",
                "expected_answer",
            ],
        }

        output_schema = (
            {
                "type": "ARRAY",
                "items": data_gen_schema,
            }
            if experiment_spec.data_gen_scaling_factor != 1
            else data_gen_schema
        )

        data_gen_render_vars = {
            "exp_params": experiment_spec.model_dump(),
            "examples": [e.model_dump() for e in experiment_spec.data_gen_examples],
            "question": question.model_dump(),
            "output_schema": json.dumps(output_schema, indent=4),
        }

        data_gen_sys_instructions = jinja_env.from_string(
            experiment_spec.data_gen_system_instructions_template
        ).render(**data_gen_render_vars)

        data_gen_prompt = jinja_env.from_string(
            experiment_spec.data_gen_prompt_template
        ).render(**data_gen_render_vars)

        client = genai.Client(
            http_options=genai.types.HttpOptions(
                api_version=BASE_CONFIG.GENAI_HTTP_API_VERSION
            ),
            vertexai=True,
            project=BASE_CONFIG.GOOGLE_CLOUD_PROJECT,
            location=BASE_CONFIG.GOOGLE_CLOUD_LOCATION,
        )

        retry_attempt = 0
        retry_errors = []
        start_time = time.perf_counter()

        async for attempt in AsyncRetrying(
            wait=wait_random_exponential(
                multiplier=BASE_CONFIG.GENAI_RETRY_DELAY_MULTIPLIER,
                max=BASE_CONFIG.GENAI_RETRY_DELAY_MAX,
            ),
            stop=stop_after_attempt(BASE_CONFIG.GENAI_RETRY_MAX_ATTEMPTS),
            retry=retry_if_not_exception_type(NonRetryableError),
        ):
            with attempt:
                retry_attempt += 1
                try:
                    response = await client.aio.models.generate_content(
                        model=experiment_spec.data_gen_llm,
                        contents=data_gen_prompt,
                        config=genai.types.GenerateContentConfig(
                            system_instruction=data_gen_sys_instructions,
                            temperature=experiment_spec.data_gen_llm_temperature,
                            response_mime_type="application/json",
                            response_schema=output_schema,
                        ),
                    )
                    llm_call_duration = time.perf_counter() - start_time
                except genai.errors.ClientError as e0:
                    error_message = (
                        f"[Attempt {retry_attempt}] Data Generation LLM Call failed for "
                        f"{question.model_dump()} : {e0}"
                    )
                    retry_errors.append(error_message)

                    if (getattr(e0, "code", None) == 400) and (
                        getattr(e0, "status", None) == "INVALID_ARGUMENT"
                    ):
                        raise NonRetryableError(
                            "[NonRetryableError] " + error_message
                        ) from e0
                    else:
                        raise e0

                except Exception as e1:
                    retry_errors.append(
                        f"[Attempt {retry_attempt}] Data Generation LLM Call failed for "
                        f"{question.model_dump()} : {e1}"
                    )
                    raise e1
                else:
                    try:
                        return [
                            (
                                DataGenAnswer.model_validate(r),
                                {
                                    "prompt": data_gen_prompt,
                                    "sys_instructions": data_gen_sys_instructions,
                                    "output_schema": json.dumps(
                                        output_schema, default=str
                                    ),
                                    "token_count": (
                                        response.usage_metadata.total_token_count
                                        if response.usage_metadata is not None
                                        else None
                                    ),
                                    "time_taken": llm_call_duration,
                                    "attempt_errors": retry_errors,
                                },
                            )
                            for r in (
                                response.parsed
                                if isinstance(response.parsed, list)
                                else [response.parsed]
                            )
                        ]
                    except Exception as e2:
                        retry_errors.append(
                            f"[Attempt {retry_attempt}] Data Generation LLM output parsing failed "
                            f"for {response} : {e2}"
                        )
                        raise e2
        raise RuntimeError(f"Failed to generate data : {json.dumps(retry_errors)}")

    async def answer_generator(
        self,
        question: AnswerGenQuestion,
        experiment_spec: ExperimentSpec,
    ) -> list[tuple[AnswerGenAnswer, AnswerGenMetadata]]:
        """
        Tries to generate synthetic answer(s)
        """

        answer_gen_schema = AnswerGenAnswer.model_json_schema()
        genai._transformers.process_schema(  # pylint: disable=protected-access
            answer_gen_schema, client=None
        )

        output_schema = (
            {
                "type": "ARRAY",
                "items": answer_gen_schema,
            }
            if experiment_spec.answer_gen_scaling_factor != 1
            else answer_gen_schema
        )

        answer_gen_render_vars = {
            "exp_params": experiment_spec.model_dump(),
            "examples": [e.model_dump() for e in experiment_spec.answer_gen_examples],
            "question": question.model_dump(),
            "output_schema": json.dumps(output_schema, indent=4),
        }

        answer_gen_sys_instructions = jinja_env.from_string(
            experiment_spec.answer_gen_system_instructions_template
        ).render(**answer_gen_render_vars)

        answer_gen_prompt = jinja_env.from_string(
            experiment_spec.answer_gen_prompt_template
        ).render(**answer_gen_render_vars)

        client = genai.Client(
            http_options=genai.types.HttpOptions(
                api_version=BASE_CONFIG.GENAI_HTTP_API_VERSION
            ),
            vertexai=True,
            project=BASE_CONFIG.GOOGLE_CLOUD_PROJECT,
            location=BASE_CONFIG.GOOGLE_CLOUD_LOCATION,
        )

        retry_attempt = 0
        retry_errors = []
        start_time = time.perf_counter()

        async for attempt in AsyncRetrying(
            wait=wait_random_exponential(
                multiplier=BASE_CONFIG.GENAI_RETRY_DELAY_MULTIPLIER,
                max=BASE_CONFIG.GENAI_RETRY_DELAY_MAX,
            ),
            stop=stop_after_attempt(BASE_CONFIG.GENAI_RETRY_MAX_ATTEMPTS),
        ):
            with attempt:
                retry_attempt += 1
                try:
                    response = await client.aio.models.generate_content(
                        model=experiment_spec.answer_gen_llm,
                        contents=answer_gen_prompt,
                        config=genai.types.GenerateContentConfig(
                            system_instruction=answer_gen_sys_instructions,
                            temperature=experiment_spec.answer_gen_llm_temperature,
                            response_mime_type="application/json",
                            response_schema=output_schema,
                        ),
                    )
                    llm_call_duration = time.perf_counter() - start_time
                except Exception as e1:
                    retry_errors.append(
                        f"[Attempt {retry_attempt}] Answer Generation LLM Call failed for "
                        f"{question.model_dump()} : {e1}"
                    )
                    raise e1
                else:
                    try:
                        return [
                            (
                                AnswerGenAnswer.model_validate(r),
                                {
                                    "prompt": answer_gen_prompt,
                                    "sys_instructions": answer_gen_sys_instructions,
                                    "output_schema": json.dumps(
                                        output_schema, default=str
                                    ),
                                    "token_count": (
                                        response.usage_metadata.total_token_count
                                        if response.usage_metadata is not None
                                        else None
                                    ),
                                    "time_taken": llm_call_duration,
                                    "attempt_errors": retry_errors,
                                },
                            )
                            for r in (
                                response.parsed
                                if isinstance(response.parsed, list)
                                else [response.parsed]
                            )
                        ]
                    except Exception as e2:
                        retry_errors.append(
                            f"[Attempt {retry_attempt}] Answer Generation LLM output parsing "
                            f"failed for {response} : {e2}"
                        )
                        raise e2
        raise RuntimeError(f"Failed to generate answers : {json.dumps(retry_errors)}")

    async def generate(
        self,
        input_row: pd.Series,
        experiment_spec: ExperimentSpec,
        *args,
        **kwargs,
    ) -> list[Tuple[GenQuestion, GenAnswer, GenMetadata]]:
        """
        Generate potential answer(s)
        """

        question = GenQuestion(
            db_schema=input_row["db_schema"],
            additional_information=input_row["additional_information"],
            db_create_statements=input_row["db_create_statements"],
            original_nlq=input_row["original_nlq"],
        )

        results: list[Tuple[GenQuestion, GenAnswer, GenMetadata]] = []

        if experiment_spec.strategy == "ANSWER_TO_DATA":
            answer_gen_results = await self.answer_generator(
                AnswerGenQuestion(
                    db_schema=question.db_schema,
                    original_nlq=question.original_nlq,
                    additional_information=input_row["additional_information"],
                    db_create_statements=question.db_create_statements,
                    target_dataset=None,
                ),
                experiment_spec,
            )
            for answer_gen_answer, answer_gen_metadata in answer_gen_results:
                data_gen_results = await self.data_generator(
                    DataGenQuestion(
                        db_schema=question.db_schema,
                        original_nlq=question.original_nlq,
                        additional_information=input_row["additional_information"],
                        db_create_statements=question.db_create_statements,
                        target_answer=answer_gen_answer,
                    ),
                    experiment_spec,
                )
                for data_gen_answer, data_gen_metadata in data_gen_results:
                    results.append(
                        (
                            question,
                            GenAnswer(
                                data_generation=data_gen_answer,
                                answer_generation=answer_gen_answer,
                            ),
                            {
                                "data_gen_metadata": data_gen_metadata,
                                "answer_gen_metadata": answer_gen_metadata,
                            },
                        )
                    )

        elif experiment_spec.strategy == "DATA_TO_ANSWER":
            data_gen_results = await self.data_generator(
                DataGenQuestion(
                    db_schema=question.db_schema,
                    original_nlq=question.original_nlq,
                    db_create_statements=question.db_create_statements,
                    additional_information=input_row["additional_information"],
                    target_answer=None,
                ),
                experiment_spec,
            )

            for data_gen_answer, data_gen_metadata in data_gen_results:
                answer_gen_results = await self.answer_generator(
                    AnswerGenQuestion(
                        db_schema=question.db_schema,
                        original_nlq=question.original_nlq,
                        additional_information=input_row["additional_information"],
                        db_create_statements=question.db_create_statements,
                        target_dataset=data_gen_answer,
                    ),
                    experiment_spec,
                )

                for answer_gen_answer, answer_gen_metadata in answer_gen_results:
                    results.append(
                        (
                            question,
                            GenAnswer(
                                data_generation=data_gen_answer,
                                answer_generation=answer_gen_answer,
                            ),
                            {
                                "data_gen_metadata": data_gen_metadata,
                                "answer_gen_metadata": answer_gen_metadata,
                            },
                        )
                    )

        return results

    async def evaluate(
        self,
        input_row: pd.Series,
        experiment_spec: ExperimentSpec,
        generate_question: GenQuestion,
        generate_answer: GenAnswer,
        *args,
        **kwargs,
    ) -> list[Tuple[EvalQuestion, EvalAnswer, EvalMetadata]]:
        """
        Evaluate a generated answer
        """

        generated_data_lower = json.loads(
            json.dumps(
                generate_answer.data_generation.generated_data, default=str
            ).lower()
        )

        tables = {
            t["table_name"].lower(): pd.DataFrame(
                generated_data_lower.get(t["table_name"].lower(), []),
                columns=[c["name"].lower() for c in t["columns"]],
            )
            for t in input_row["db_schema"]["tables"]
        }

        answer = {}
        metadata: dict[str, Optional[str]] = {
            "generated_sql_result_json_records_error": None,
            "original_sql_result_json_records_error": None,
            "generated_sql_result_json_values_error": None,
            "original_sql_result_json_values_error": None,
            "generated_sql_result_json_table_error": None,
            "original_sql_result_json_table_error": None,
        }

        for sql_key in ["generated_sql", "original_sql"]:
            # pylint: disable=broad-exception-caught
            try:
                result = sqldf(input_row[sql_key].lower(), tables)
            except Exception as e1:
                answer[f"{sql_key}_execution_error"] = str(e1)
            else:
                result_sorted = [
                    "-<colsep>-".join(sorted([str(c) for c in r]))
                    for r in result.to_numpy()
                ]
                answer[f"{sql_key}_result_sorted_inner"] = "-<rowsep>-".join(
                    result_sorted
                )
                answer[f"{sql_key}_result_sorted_innerouter"] = "-<rowsep>-".join(
                    sorted(result_sorted)
                )

                for json_mode in ["records", "values", "table"]:
                    try:
                        answer[f"{sql_key}_result_json_{json_mode}"] = result.to_json(
                            orient=json_mode,
                            default_handler=str,
                        )
                    except Exception as e2:
                        metadata[f"{sql_key}_result_json_{json_mode}_error"] = str(e2)

        return [
            (
                EvalQuestion(),
                EvalAnswer(**answer),
                cast(EvalMetadata, metadata),
            )
        ]
