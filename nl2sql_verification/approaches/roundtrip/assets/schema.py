"""
Provides the schemas for the Round-Trip Approacch
"""

from typing import Optional, Self, Any

from ....config import BASE_CONFIG
from ...base import (
    BaseQuestion,
    BaseAnswer,
    BaseExample,
    BaseExperimentSpec,
    BaseMetadata,
)
from .examples import (
    GEN_EXAMPLES,
    EVAL_EXAMPLES,
    GEN_EXAMPLES_WITH_ADDITIONAL_INFORMATION,
    EVAL_EXAMPLES_WITH_ADDITIONAL_INFORMATION,
)

from .prompts import (
    GENERATOR_SYSTEM_INSTRUCTION,
    GENERATOR_CONTENT,
    EVALUATOR_SYSTEM_INSTRUCTION,
    EVALUATOR_CONTENT,
)


class GenQuestion(BaseQuestion):
    """
    Schema of a generation-question
    """

    generated_sql: str


class GenAnswer(BaseAnswer):
    """
    Schema of a a generation-answer
    """

    divide: list[str]
    conquer: list[str]
    combine: list[str]
    final_output: str


class GenExample(BaseExample[GenQuestion, GenAnswer]):
    """
    Implements a single example for generation
    """


ALL_GEN_EXAMPLES = [
    GenExample.model_validate(example) for example in GEN_EXAMPLES.values()
]


class EvalQuestion(BaseQuestion):
    """
    Schema of an eval-question
    """

    nlq_1: str  # original
    nlq_2: str  # generated


class EvalAnswer(BaseAnswer):
    """
    Schema of a a generation-answer
    """

    nlq_1_core_intent: str
    nlq_2_core_intent: str
    intent_match: str
    nlq_1_entities_and_attributes: str
    nlq_2_entities_and_attributes: str
    entities_and_attributes_match: str
    nlq_1_conditions: str
    nlq_2_conditions: str
    conditions_match: str
    nlq_1_aggregations_or_ordering: str
    nlq_2_aggregations_or_ordering: str
    aggregations_or_ordering_match: str
    final_judgment: str
    reasoning: str
    similarity_score: int
    confidence_score: float


class EvalExample(BaseExample[EvalQuestion, EvalAnswer]):
    """
    Implements a single example for evaluation
    """


ALL_EVAL_EXAMPLES = [
    EvalExample.model_validate(example) for example in EVAL_EXAMPLES.values()
]


class GenMetadata(BaseMetadata):
    """
    Metadata for the generation operation
    """

    prompt: str
    sys_instructions: str
    output_schema: str
    token_count: Optional[int]
    time_taken: float
    attempt_errors: list[str]


class EvalMetadata(BaseMetadata):
    """
    Metadata for the evaluation operation
    """

    prompt: str
    sys_instructions: str
    output_schema: str
    token_count: Optional[int]
    time_taken: float
    attempt_errors: list[str]


class ExperimentSpec(BaseExperimentSpec):
    """
    Implements a single experiment spec
    """

    scaling_factor: int = 0
    generator_examples: list[GenExample]
    evaluator_examples: list[EvalExample]
    generator_hide_additional_information: bool
    evaluator_hide_additional_information: bool
    generator_hide_db_schema: bool = True
    evaluator_hide_db_schema: bool = True
    generator_hide_db_create_statements: bool = False
    evaluator_hide_db_create_statements: bool = False
    generator_prompt_template: str = GENERATOR_CONTENT
    evaluator_prompt_template: str = EVALUATOR_CONTENT
    generator_system_instruction_template: str = GENERATOR_SYSTEM_INSTRUCTION
    evaluator_system_instruction_template: str = EVALUATOR_SYSTEM_INSTRUCTION
    generator_llm: str = BASE_CONFIG.GENAI_DEFAULT_LLM
    evaluator_llm: str = BASE_CONFIG.GENAI_DEFAULT_LLM
    generator_llm_temperature: float = BASE_CONFIG.GENAI_DEFAULT_TEMPERATURE
    evaluator_llm_temperature: float = BASE_CONFIG.GENAI_DEFAULT_TEMPERATURE

    @classmethod
    def parser(cls, exp_params: dict) -> Self:
        """
        Parse an experiment spec
        """

        exp_params["generator_hide_additional_information"] = exp_params.get(
            "generator_hide_additional_information", True
        )
        exp_params["evaluator_hide_additional_information"] = exp_params.get(
            "evaluator_hide_additional_information", True
        )

        keyword: str
        default_val_dict: dict[Any, Any]
        example_class: type[BaseExample[Any, Any]]

        for keyword, default_val_dict, example_class in [  # type: ignore[assignment]
            (
                "generator_examples",
                (
                    GEN_EXAMPLES
                    if exp_params["generator_hide_additional_information"]
                    else GEN_EXAMPLES_WITH_ADDITIONAL_INFORMATION
                ),
                GenExample,
            ),
            (
                "evaluator_examples",
                (
                    EVAL_EXAMPLES
                    if exp_params["evaluator_hide_additional_information"]
                    else EVAL_EXAMPLES_WITH_ADDITIONAL_INFORMATION
                ),
                EvalExample,
            ),
        ]:
            if exp_params.get(keyword) is None:
                exp_params[keyword] = [
                    example_class.model_validate(v) for v in default_val_dict.values()
                ]

            elif isinstance(exp_params[keyword], list):
                exp_params[keyword] = [
                    (
                        example_class.model_validate(default_val_dict[example])
                        if isinstance(example, str) and (example in default_val_dict)
                        else (
                            example_class.model_validate(example)
                            if isinstance(example, dict)
                            else example
                        )
                    )
                    for example in exp_params[keyword]
                ]

        return cls(**exp_params)
