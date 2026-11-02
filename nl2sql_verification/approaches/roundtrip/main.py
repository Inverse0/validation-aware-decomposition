"""
Implements the Round-Trip approach to NL2SQL Verification
"""

# pylint: disable=unused-argument

import json
import time
from typing import Tuple

import pandas as pd
from google import genai
from jinja2 import Environment
from tenacity import AsyncRetrying, wait_random_exponential, stop_after_attempt

from ...config import BASE_CONFIG
from ..base import BaseApproach
from .assets.schema import (
    GenQuestion,
    GenAnswer,
    EvalQuestion,
    EvalAnswer,
    ExperimentSpec,
    GenMetadata,
    EvalMetadata,
)

jinja_env = Environment()
jinja_env.policies["json.dumps_kwargs"]["sort_keys"] = False


class RoundTrip(
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
    RoundTrip Verification
    """

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

        output_schema = GenAnswer.model_json_schema()
        genai._transformers.process_schema(  # pylint: disable=protected-access
            output_schema, client=None
        )
        if experiment_spec.scaling_factor != 1:
            output_schema = {
                "type": "ARRAY",
                "items": output_schema,
            }

        question = GenQuestion(
            db_schema=input_row["db_schema"],
            db_create_statements=input_row["db_create_statements"],
            additional_information=input_row["additional_information"],
            generated_sql=input_row["generated_sql"],
        )
        sys_instructions = jinja_env.from_string(
            experiment_spec.generator_system_instruction_template
        ).render(
            exp_params=experiment_spec.model_dump(),
            examples=[e.model_dump() for e in experiment_spec.generator_examples],
            question=question.model_dump(),
            output_schema=json.dumps(output_schema, indent=4),
        )

        prompt = jinja_env.from_string(
            experiment_spec.generator_prompt_template
        ).render(
            exp_params=experiment_spec.model_dump(),
            examples=[e.model_dump() for e in experiment_spec.generator_examples],
            question=question.model_dump(),
            output_schema=output_schema,
        )

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
            wait=wait_random_exponential(multiplier=0.01, max=5),
            stop=stop_after_attempt(50),
        ):
            with attempt:
                retry_attempt += 1
                try:
                    response = await client.aio.models.generate_content(
                        model=experiment_spec.generator_llm,
                        contents=prompt,
                        config=genai.types.GenerateContentConfig(
                            system_instruction=sys_instructions,
                            temperature=experiment_spec.generator_llm_temperature,
                            response_mime_type="application/json",
                            response_schema=output_schema,
                        ),
                    )
                except Exception as e1:
                    retry_errors.append(
                        f"[Attempt {retry_attempt}] LLM Call failed for {question.uuid} : {e1}"
                    )
                    raise e1
                else:
                    llm_call_duration = time.perf_counter() - start_time
                    try:
                        results: list[tuple[GenQuestion, GenAnswer, GenMetadata]] = [
                            (
                                question,
                                GenAnswer.model_validate(r),
                                {
                                    "prompt": prompt,
                                    "sys_instructions": sys_instructions,
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
                            f"[Attempt {retry_attempt}] LLM output parsing failed for {response} : {e2}"
                        )
                        raise e2
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

        question = EvalQuestion(
            db_schema=input_row["db_schema"],
            db_create_statements=input_row["db_create_statements"],
            additional_information=input_row["additional_information"],
            nlq_1=input_row["original_nlq"],
            nlq_2=generate_answer.final_output,
        )

        output_schema = EvalAnswer.model_json_schema()
        genai._transformers.process_schema(  # pylint: disable=protected-access
            output_schema, client=None
        )

        sys_instructions = jinja_env.from_string(
            experiment_spec.evaluator_system_instruction_template
        ).render(
            exp_params=experiment_spec.model_dump(),
            examples=[e.model_dump() for e in experiment_spec.evaluator_examples],
            question=question.model_dump(),
            output_schema=json.dumps(output_schema, indent=4),
        )

        prompt = jinja_env.from_string(
            experiment_spec.evaluator_prompt_template
        ).render(
            exp_params=experiment_spec.model_dump(),
            examples=[e.model_dump() for e in experiment_spec.evaluator_examples],
            question=question.model_dump(),
            output_schema=output_schema,
        )

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
            wait=wait_random_exponential(multiplier=0.01, max=5),
            stop=stop_after_attempt(50),
        ):
            with attempt:
                retry_attempt += 1
                try:
                    response = await client.aio.models.generate_content(
                        model=experiment_spec.evaluator_llm,
                        contents=prompt,
                        config=genai.types.GenerateContentConfig(
                            system_instruction=sys_instructions,
                            temperature=experiment_spec.evaluator_llm_temperature,
                            response_mime_type="application/json",
                            response_schema=output_schema,
                        ),
                    )
                except Exception as e1:
                    retry_errors.append(
                        f"[Attempt {retry_attempt}] LLM Call failed for {question.uuid} : {e1}"
                    )
                    raise e1
                else:
                    try:
                        result = EvalAnswer.model_validate(response.parsed)
                    except Exception as e2:
                        retry_errors.append(
                            f"[Attempt {retry_attempt}] LLM output parsing failed for {response} : {e2}"
                        )
                        raise e2

        return [
            (
                question,
                result,
                {
                    "prompt": prompt,
                    "sys_instructions": sys_instructions,
                    "output_schema": json.dumps(output_schema, default=str),
                    "token_count": (
                        response.usage_metadata.total_token_count
                        if response.usage_metadata is not None
                        else None
                    ),
                    "time_taken": time.perf_counter() - start_time,
                    "attempt_errors": retry_errors,
                },
            )
        ]
