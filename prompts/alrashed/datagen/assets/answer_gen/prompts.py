"""
Implements the answer generation prompts for the DataGen approach
"""

SYSTEM_INSTRUCTIONS_TEMPLATE = """
You are a meticulous and logical data analyst. Your task is to receive a natural language question (NLQ) and a database schema{{ ' and a dataset' if exp_params.strategy != 'ANSWER_TO_DATA' else '' }}, and then perform a step-by-step reasoning process to generate {{ 'multiple, diverse, plausible answers' if exp_params.answer_gen_scaling_factor != 1 else 'a plausible answer' }} to the NLQ{{ ' grounded in the provided dataset' if exp_params.strategy != 'ANSWER_TO_DATA' else '' }}.

Your goal is to {{ 'first mentally create plausible data based on the schema and then derive your answer(s) from that mental data.' if exp_params.strategy == 'ANSWER_TO_DATA' else 'derive your answer(s) exclusively from the provided dataset.' }}

You will use a methodical, three-step approach:
### 1. Analyze (Deconstruct the Inputs)
First, break down the NLQ into its core components (e.g., what is being asked for, what are the conditions). Simultaneously, analyze the provided **database schema** to understand the tables, columns, and relationships.{{ ' Next, analyze the **provided dataset** to understand the available rows, columns and data.' if exp_params.strategy != 'ANSWER_TO_DATA' else '' }}
### 2. Plan
{{ '**Synthesize Mental Data:** Based on the NLQ and schema, mentally create a minimal but sufficient set of plausible data. This data must be realistic and structured to produce a logical, non-empty answer.\\n*' if exp_params.strategy == 'ANSWER_TO_DATA' else '**Formulate a Query Plan:** Based on the NLQ and the **provided data**, mentally formulate the query logic (like SQL) that will extract the answer.' }}
### 3. Execute & Answer (Trace the Logic and State the Result)
For the answer you generate, provide the following:
* **Reasoning:**{{ ' First, describe the key data points you are assuming for this specific answer. For example: *"For my first answer, I\\'ll assume the `employees` table shows the `Sales` department has 5 members and `Marketing` has 3."* Then, detail the step-by-step process of how you use that mental data to reach a conclusion.' if exp_params.strategy == 'ANSWER_TO_DATA' else ' Detail the step-by-step process of how you execute your query plan against the **provided data** — applying filters, joins, or aggregations—to arrive at the final result.' }}
* **Sentence:** Formulate a clear, natural language sentence that directly answers the NLQ.
* **Dictionary:** Present the final answer as a JSON list of dictionaries.

{{ 'Most importantly, Perform this analysis multiple times to generate multiple, diverse answers that follow all of the above conditions.' if exp_params.answer_gen_scaling_factor != 1 else '' }}.

### **Instructions & Constraints (Violation is strictly forbidden):**
1.  **Source of Truth:** Your final answer(s) and reasoning **must** be derived *exclusively* from {{ 'the data you mentally generated and described in your reasoning.' if exp_params.strategy == 'ANSWER_TO_DATA' else 'the provided dataset.' }}
2.  **Output Format:** You must provide your response as a single JSON object conforming to the following schema for each answer:
{{output_schema}}
""".strip()


PROMPT_TEMPLATE = """
{% if examples | length > 0 %}Here {{ 'are some examples' if examples | length > 1 else 'is an example' }} - analyse carefully and stick to the pattern for your output{% endif %}
{% for example in examples %}

* INPUT *
{% if example.question.db_create_statements and not exp_params.answer_gen_hide_db_create_statements %}
[**Table creation statements**]
{% for cs in example.question.db_create_statements %}
```sql
{{cs}}
```
{% endfor %}
{% endif %}
{% if example.question.db_schema and not exp_params.answer_gen_hide_db_schema %}
[**Database Schema**]
{{example.question.db_schema | tojson(indent=4) }}
{% endif %}
[**Natural Language Question (NLQ)**]
{{example.question.original_nlq}}
{% if example.question.additional_information and not exp_params.answer_gen_hide_additional_information %}
[**Additional Information**]
{{example.question.additional_information}}
{% endif %}

{% if example.question.target_dataset and exp_params.strategy == 'DATA_TO_ANSWER' %}
{% if example.question.target_dataset.analyze_nlq and not exp_params.answer_gen_hide_target_dataset_analyze_nlq %}
[**Dataset Generation - NLQ Analysis**]
{{example.question.target_dataset.analyze_nlq}}
{% endif %}
{% if example.question.target_dataset.plan and not exp_params.answer_gen_hide_target_dataset_plan %}
[**Dataset Generation Plan**]
{{example.question.target_dataset.plan}}
{% endif %}
{% if example.question.target_dataset.synthesize and not exp_params.answer_gen_hide_target_dataset_synthesize %}
[**Dataset Synthesis Reasoning**]
{{example.question.target_dataset.synthesize}}
{% endif %}
{% if example.question.target_dataset.focus_area_description and not exp_params.answer_gen_hide_target_dataset_focus_area_description %}
[**Dataset Focus Area**]
{{example.question.target_dataset.focus_area_description}}
{% endif %}

[**Dataset**]
{{example.question.target_dataset.generated_data | tojson(indent=4) }}

{% if example.question.target_dataset.expected_answer and not exp_params.answer_gen_hide_target_dataset_expected_answer %}
[**Predicted Answer**]
{{example.question.target_dataset.expected_answer}}
{% endif %}
{% endif %}

* SOLUTION *
{% for answer in (example.answers if exp_params.answer_gen_scaling_factor != 1 else [example.answers[0]]) %}
[**Answer {{loop.index if exp_params.answer_gen_scaling_factor != 1 else ''}}**]

***Analyze NLQ***
{{ answer.analyze_nlq }}

***Plan***
{{ answer.plan }}

***Answer Reasoning***
{{ answer.answer_reasoning }}

***Answer Sentence***
{{ answer.answer_sentence }}

***Answer Dictionary:***
{{ answer.answer_dictionary }}
---
{% endfor %}
[**JSON Representation**] - Follow this in your own output
{{(example.answers if exp_params.answer_gen_scaling_factor != 1 else example.answers[0]) | tojson(indent=4)}}

{% endfor %}
Now Solve for this{% if exp_params.answer_gen_scaling_factor != 1 %} and make sure your provide {% if exp_params.answer_gen_scaling_factor == 0 %}as many as possible {% else %}at least {{exp_params.answer_gen_scaling_factor}} {% endif %}independent and meaningfully different answers, focused on different areas {% endif %}:

* INPUT *
{% if question.db_create_statements and not exp_params.answer_gen_hide_db_create_statements %}
[**Table creation statements**]
{% for cs in question.db_create_statements %}
```sql
{{cs}}
```
{% endfor %}
{% endif %}
{% if question.db_schema and not exp_params.answer_gen_hide_db_schema %}
[**Database Schema**]
{{question.db_schema | tojson(indent=4) }}
{% endif %}
[**Natural Language Question (NLQ)**]
{{question.original_nlq}}
{% if question.additional_information and not exp_params.answer_gen_hide_additional_information %}
[**Additional Information**]
{{question.additional_information}}
{% endif %}

{% if question.target_dataset and exp_params.strategy == 'DATA_TO_ANSWER' %}
{% if question.target_dataset.analyze_nlq and not exp_params.answer_gen_hide_target_dataset_analyze_nlq %}
[**Dataset Generation - NLQ Analysis**]
{{question.target_dataset.analyze_nlq}}
{% endif %}
{% if question.target_dataset.plan and not exp_params.answer_gen_hide_target_dataset_plan %}
[**Dataset Generation Plan**]
{{question.target_dataset.plan}}
{% endif %}
{% if question.target_dataset.synthesize and not exp_params.answer_gen_hide_target_dataset_synthesize %}
[**Dataset Synthesis Reasoning**]
{{question.target_dataset.synthesize}}
{% endif %}
{% if question.target_dataset.focus_area_description and not exp_params.answer_gen_hide_target_dataset_focus_area_description %}
[**Dataset Focus Area**]
{{question.target_dataset.focus_area_description}}
{% endif %}

[**Dataset**]
{{question.target_dataset.generated_data | tojson(indent=4) }}

{% if question.target_dataset.expected_answer and not exp_params.answer_gen_hide_target_dataset_expected_answer %}
[**Predicted Answer**]
{{question.target_dataset.expected_answer}}
{% endif %}
{% endif %}

* SOLUTION *
""".strip()
