"""
Provides the schemas for the Data Gen Approacch
"""

from typing import Type, Literal, Optional, Any, Union, Self

from pydantic import computed_field

from ....config import BASE_CONFIG
from ...base import (
    BaseQuestion,
    BaseAnswer,
    BaseExample,
    BaseExperimentSpec,
    BaseMetadata,
    DatabaseSchemaDict,
)
from . import answer_gen, data_gen


DATATYPE_MAP = {
    "number": {"type": "number"},
    "text": {"type": "string"},
    "others": {"type": "string"},
    "integer": {"type": "integer"},
    "real": {"type": "number"},
    # "datetime": {"type": "string", "format": "date-time"},
    # "date": {"type": "string", "format": "date-time"},
    # "time": {"type": "string", "format": "date-time"},
    "datetime": {"type": "string"},
    "date": {"type": "string"},
    "time": {"type": "string"},
}


class GenQuestion(BaseQuestion):
    """
    Schema of a generation-question
    """

    original_nlq: str
    db_schema: DatabaseSchemaDict
    db_create_statements: list[str]


# Answer


class DataGenAnswer(BaseAnswer):
    """
    Schema of a data generation answer
    """

    analyze_nlq: str
    plan: str
    synthesize: str
    focus_area_description: str
    generated_data: dict[str, list[dict[str, Any]]]
    expected_answer: str


class AnswerGenAnswer(BaseAnswer):
    """
    Schema of a data generation answer
    """

    analyze_nlq: str
    plan: str
    answer_reasoning: str
    answer_sentence: str
    answer_dictionary: list[dict[str, Any]]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def answer_dictionary_sorted(self) -> dict[str, Union[str, None]]:
        """
        Sorts and serializes the answer dictionary
        """
        try:
            datasorted = [
                "-<colsep>-".join(sorted([str(c) for c in r.values()]))
                for r in self.answer_dictionary
            ]

            innersorted = "-<rowsep>-".join(datasorted)

            inneroutersorted = "-<rowsep>-".join(sorted(datasorted))

            return {
                "innersorted": innersorted,
                "inneroutersorted": inneroutersorted,
                "innersortedlower": innersorted.lower(),
                "inneroutersortedlower": inneroutersorted.lower(),
                "sortingerror": None,
            }
        except Exception as e:  # pylint: disable=broad-exception-caught
            return {
                "innersorted": None,
                "inneroutersorted": None,
                "innersortedlower": None,
                "inneroutersortedlower": None,
                "sortingerror": str(e),
            }


# Metadata


class DataGenMetadata(BaseMetadata):
    """
    Metadata for the generation operation
    """

    prompt: str
    sys_instructions: str
    output_schema: str
    token_count: Optional[int]
    time_taken: float
    attempt_errors: list[str]


class AnswerGenMetadata(BaseMetadata):
    """
    Metadata for the generation operation
    """

    prompt: str
    sys_instructions: str
    output_schema: str
    token_count: Optional[int]
    time_taken: float
    attempt_errors: list[str]


# Question


class DataGenQuestion(GenQuestion):
    """
    Schema of a generation-question
    """

    target_answer: Optional[AnswerGenAnswer] = None


class AnswerGenQuestion(GenQuestion):
    """
    Schema of a generation-question
    """

    target_dataset: Optional[DataGenAnswer] = None


# Example


class DataGenExample(BaseExample[DataGenQuestion, DataGenAnswer]):
    """
    Implements a single example for generation
    """


class AnswerGenExample(BaseExample[AnswerGenQuestion, AnswerGenAnswer]):
    """
    Implements a single example for generation
    """


class GenAnswer(BaseAnswer):
    """
    Schema of a generation-question
    """

    data_generation: DataGenAnswer
    answer_generation: AnswerGenAnswer


class GenMetadata(BaseMetadata):
    """
    Metadata for the generation operation
    """

    data_gen_metadata: DataGenMetadata
    answer_gen_metadata: AnswerGenMetadata


class EvalQuestion(BaseQuestion):
    """
    Schema of an evaluation-question
    """


class EvalAnswer(BaseAnswer):
    """
    Schema of an evaluation-question
    """

    generated_sql_execution_error: Optional[str] = None
    original_sql_execution_error: Optional[str] = None

    generated_sql_result_sorted_inner: Optional[str] = None
    original_sql_result_sorted_inner: Optional[str] = None
    generated_sql_result_sorted_innerouter: Optional[str] = None
    original_sql_result_sorted_innerouter: Optional[str] = None

    generated_sql_result_json_records: Optional[str] = None
    original_sql_result_json_records: Optional[str] = None
    generated_sql_result_json_values: Optional[str] = None
    original_sql_result_json_values: Optional[str] = None
    generated_sql_result_json_table: Optional[str] = None
    original_sql_result_json_table: Optional[str] = None


class EvalMetadata(BaseMetadata):
    """
    Metadata for the evaluation operation
    """

    generated_sql_result_json_records_error: Optional[str]
    original_sql_result_json_records_error: Optional[str]
    generated_sql_result_json_values_error: Optional[str]
    original_sql_result_json_values_error: Optional[str]
    generated_sql_result_json_table_error: Optional[str]
    original_sql_result_json_table_error: Optional[str]


class ExperimentSpec(BaseExperimentSpec):
    """
    Implements a single experiment spec
    """

    strategy: Literal["DATA_TO_ANSWER", "ANSWER_TO_DATA"]

    data_gen_examples: list[DataGenExample]
    answer_gen_examples: list[AnswerGenExample]

    data_gen_prompt_template: str = data_gen.prompts.PROMPT_TEMPLATE
    data_gen_system_instructions_template: str = (
        data_gen.prompts.SYSTEM_INSTRUCTIONS_TEMPLATE
    )
    data_gen_scaling_factor: int = 0
    data_gen_llm: str = BASE_CONFIG.GENAI_DEFAULT_LLM
    data_gen_llm_temperature: float = BASE_CONFIG.GENAI_DEFAULT_TEMPERATURE
    data_gen_hide_db_create_statements: bool = False
    data_gen_hide_additional_information: bool
    data_gen_hide_db_schema: bool = False
    data_gen_hide_target_answer_analyze_nlq: bool = False
    data_gen_hide_target_answer_plan: bool = False
    data_gen_hide_target_answer_reasoning: bool = False

    answer_gen_prompt_template: str = answer_gen.prompts.PROMPT_TEMPLATE
    answer_gen_system_instructions_template: str = (
        answer_gen.prompts.SYSTEM_INSTRUCTIONS_TEMPLATE
    )
    answer_gen_scaling_factor: int = 0
    answer_gen_llm: str = BASE_CONFIG.GENAI_DEFAULT_LLM
    answer_gen_llm_temperature: float = BASE_CONFIG.GENAI_DEFAULT_TEMPERATURE
    answer_gen_hide_db_create_statements: bool = False
    answer_gen_hide_additional_information: bool
    answer_gen_hide_db_schema: bool = False
    answer_gen_hide_target_dataset_analyze_nlq: bool = False
    answer_gen_hide_target_dataset_plan: bool = False
    answer_gen_hide_target_dataset_synthesize: bool = False
    answer_gen_hide_target_dataset_focus_area_description: bool = False
    answer_gen_hide_target_dataset_expected_answer: bool = True

    @classmethod
    def parser(cls, exp_params: dict) -> Self:
        """
        Parse an experiment spec
        """

        if "strategy" not in exp_params:
            return cls(**exp_params)

        exp_params["data_gen_hide_additional_information"] = exp_params.get(
            "data_gen_hide_additional_information", True
        )
        exp_params["answer_gen_hide_additional_information"] = exp_params.get(
            "answer_gen_hide_additional_information", True
        )

        example_class: Type[BaseExample[Any, Any]]
        for keyword, example_store, example_class in [  # type: ignore[assignment]
            (
                "data_gen_examples",
                {
                    True: {
                        "DATA_TO_ANSWER": data_gen.examples.DATA_TO_ANSWER_EXAMPLES,
                        "ANSWER_TO_DATA": data_gen.examples.ANSWER_TO_DATA_EXAMPLES,
                    },
                    False: {
                        "DATA_TO_ANSWER": data_gen.examples.DATA_TO_ANSWER_EXAMPLES_WITH_ADDITIONAL_INFORMATION,
                        "ANSWER_TO_DATA": data_gen.examples.ANSWER_TO_DATA_EXAMPLES_WITH_ADDITIONAL_INFORMATION,
                    },
                }[exp_params["data_gen_hide_additional_information"]][
                    exp_params["strategy"]
                ],
                DataGenExample,
            ),
            (
                "answer_gen_examples",
                {
                    True: {
                        "DATA_TO_ANSWER": answer_gen.examples.DATA_TO_ANSWER_EXAMPLES,
                        "ANSWER_TO_DATA": answer_gen.examples.ANSWER_TO_DATA_EXAMPLES,
                    },
                    False: {
                        "DATA_TO_ANSWER": answer_gen.examples.DATA_TO_ANSWER_EXAMPLES_WITH_ADDITIONAL_INFORMATION,
                        "ANSWER_TO_DATA": answer_gen.examples.ANSWER_TO_DATA_EXAMPLES_WITH_ADDITIONAL_INFORMATION,
                    },
                }[exp_params["answer_gen_hide_additional_information"]][
                    exp_params["strategy"]
                ],
                AnswerGenExample,
            ),
        ]:
            if exp_params.get(keyword) is None:
                exp_params[keyword] = [
                    example_class.model_validate(v) for v in example_store.values()
                ]

            elif isinstance(exp_params[keyword], list):
                exp_params[keyword] = [
                    (
                        example_class.model_validate(example_store[example])
                        if isinstance(example, str) and (example in example_store)
                        else (
                            example_class.model_validate(example)
                            if isinstance(example, dict)
                            else example
                        )
                    )
                    for example in exp_params[keyword]
                ]

        return cls(**exp_params)
