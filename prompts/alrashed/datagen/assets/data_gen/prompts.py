"""
Implements the data generation prompts for the DataGen approach
"""

SYSTEM_INSTRUCTIONS_TEMPLATE = """
You are a meticulous and logical synthetic data generation engine. Your task is to receive a database schema and a natural language question (NLQ){{ ', and also a target answer to the NLQ' if exp_params.strategy == 'ANSWER_TO_DATA' else '' }}, and then perform a step-by-step reasoning process{{ ' multiple times to generate multiple, diverse synthetic datasets' if exp_params.data_gen_scaling_factor != 1 else ' to generate a synthetic dataset' }}{{ ' that completely aligns with the NLQ and its target answer. Your goal is to generate the data such that someone trying to answer the NLQ by looking at the generated data is **guaranteed** to arrive at the same conclusion as the target answer' if exp_params.strategy == 'ANSWER_TO_DATA' else '' }}.

You will use a methodical, step-by-step approach:

1.  **Analyze (Deconstruct the Question and Schema):** First, break down the NLQ into its core components (e.g., what is being asked for, what are the conditions). Simultaneously, analyze the provided **database schema JSON** to understand the tables, columns, data types, and foreign key relationships. {{ 'Then, analyse the target answer to the NLQ to understand the expected distributions and information that must be present in the dataset.' if exp_params.strategy == 'ANSWER_TO_DATA' else '' }}This step is about mapping the parts of the question to the parts of the database{{ ' and understanding what properties and distributions the generated data must have' if exp_params.strategy == 'ANSWER_TO_DATA' else '' }}.
2.  **Plan (Formulate a plan):** Based on your analysis, determine an overall `theme` of the dataset - this should describe in a few sentences the overall skew of the dataset, and what patterns / distributions should be introduced in the data to make it realistic and relevant to the NLQ{{ ' and aligned with the target answer' if exp_params.strategy == 'ANSWER_TO_DATA' else '' }}. Also determine what volume and distribution of synthetic data must be made available in the database so that someone looking at the data can easily answer the NLQ{{ ' and arrive at the same conclusion as the target answer' if exp_params.strategy == 'ANSWER_TO_DATA' else '' }}. Make sure you address all the structural and implied relationships between tables and columns, and incorporate them into your data-generation plan.
3.  **Synthesize (Generate Realistic & Sufficient Data):** Based on the 2 steps above, create a minimal but sufficient set of synthetic data. This data must be realistic, maintain referential integrity, and be structured to produce a logical, non-empty answer to the NLQ{{ ' that is identical to the provided answer' if exp_params.strategy == 'ANSWER_TO_DATA' else '' }}.
{{ 'MOST IMPORTANTLY - perform this analysis multiple times to generate multiple, diverse synthetic datasets that follow all of the above conditions.' if exp_params.data_gen_scaling_factor != 1 else '' }}.
---

### **Instructions & Constraints (Violation is strictly forbidden):**

1.  **Schema Adherence:**
      * Use **only** the tables and columns defined in the provided **database schema JSON**.
      * Do not add new tables or columns.
2.  **Data Generation (JSON Format):**
      * **Completeness:** You must generate a small dataset for **every table** defined in the input `Database Schema`. Furthermore, for each table, ensure each column and its datatype is accounted for. Make sure you generate AT LEAST 1 row for every table.
      * **Relevance:** Generate data that is directly relevant and sufficient for the question{{ ' to result in the same answer as the target answer' if exp_params.strategy == 'ANSWER_TO_DATA' else '' }}. For tables not directly involved in the query, you must still generate some minimal but plausible data.
      * **Sufficiency:** The data must be constructed to yield a non-empty, logical result.
      * **Referential Integrity:** All foreign key values must correspond to a primary key in the referenced table. You must also account for any possible semantic relationships between the contents of tables and columns even if they are not linked by foreign keys.
      * **Plausibility:** Values should be realistic for their column type and name (e.g., 'John Doe' for a name, '2025-07-15' for a date).
      * **Brevity:** Generate a small, representative set of rows for each necessary table (ideally 3-5 rows per table).
      * **Format:** The data must be presented as a single JSON object containing a list of tables, as shown in the examples.
3.  **Expected Answer:**
      * The final expected_answer **must** be derived *exclusively* from analysis of the generated data.

You must provide your answer in the following schema:
{{output_schema}}
""".strip()


PROMPT_TEMPLATE = """
{% if examples | length > 0 %}Here {{ 'are some examples' if examples | length > 1 else 'is an example' }} - analyse carefully and stick to the pattern for your output{% endif %}
{% for example in examples %}

* INPUT *
{% if example.question.db_create_statements and not exp_params.data_gen_hide_db_create_statements %}
[**Table creation statements**]
{% for cs in example.question.db_create_statements %}
```sql
{{cs}}
```
{% endfor %}
{% endif %}
{% if example.question.db_schema and not exp_params.data_gen_hide_db_schema %}
[**Database Schema**]
{{example.question.db_schema | tojson(indent=4) }}
{% endif %}
[**Natural Language Question (NLQ)**]
{{example.question.original_nlq}}
{% if example.question.additional_information and not exp_params.data_gen_hide_additional_information %}
[**Additional Information**]
{{example.question.additional_information}}
{% endif %}
{% if example.question.target_answer and exp_params.strategy == 'ANSWER_TO_DATA' %}
{% if example.question.target_answer.analyze_nlq and not exp_params.data_gen_hide_target_answer_analyze_nlq %}
[**Answer Generation - NLQ Analysis**]
{{example.question.target_answer.analyze_nlq}}
{% endif %}
{% if example.question.target_answer.plan and not exp_params.data_gen_hide_target_answer_plan %}
[**Answer Generation Plan**]
{{example.question.target_answer.plan}}
{% endif %}
{% if example.question.target_answer.answer_reasoning and not exp_params.data_gen_hide_target_answer_reasoning %}
[**Answer Generation Reasoning**]
{{example.question.target_answer.answer_reasoning}}
{% endif %}

[**Target Answer**]
{{example.question.target_answer.answer_sentence}}
AS JSON:
{{example.question.target_answer.answer_dictionary | tojson(indent=4) }}
{% endif %}


* SOLUTION *
{% for answer in (example.answers if exp_params.data_gen_scaling_factor != 1 else [example.answers[0]]) %}
[**Answer {{loop.index if exp_params.data_gen_scaling_factor != 1 else ''}}**]

[**Analyse NLQ**]
{{answer.analyze_nlq}}

[**Plan**]
{{answer.plan}}

[**Synthesize**]
{{answer.synthesize}}

[**Focus Area Description**]
{{answer.focus_area_description}}

[** Generated Data**]

{{answer.generated_data | tojson(indent=4) }}

[**Expected Answer**]
{{answer.expected_answer}}
---

{% endfor %}

[**JSON Representation**] - Follow this in your own output
{{(example.answers if exp_params.data_gen_scaling_factor != 1 else example.answers[0]) | tojson(indent=4)}}

{% endfor %}
Now Solve for this{% if exp_params.data_gen_scaling_factor != 1 %} and make sure your provide {% if exp_params.data_gen_scaling_factor == 0 %}as many as possible {% else %}at least {{exp_params.data_gen_scaling_factor}} {% endif %}independent and meaningfully different datasets, focused on different areas. Always make sure you generate AT LEAST 1 row for every table. {% endif %}:

* INPUT *
{% if question.db_create_statements and not exp_params.data_gen_hide_db_create_statements %}
[**Table creation statements**]
{% for cs in question.db_create_statements %}
```sql
{{cs}}
```
{% endfor %}
{% endif %}
{% if question.db_schema and not exp_params.data_gen_hide_db_schema %}
[**Database Schema**]
{{question.db_schema | tojson(indent=4) }}
{% endif %}
[**Natural Language Question (NLQ)**]
{{question.original_nlq}}
{% if question.additional_information and not exp_params.data_gen_hide_additional_information %}
[**Additional Information**]
{{question.additional_information}}
{% endif %}
{% if question.target_answer and exp_params.strategy == 'ANSWER_TO_DATA' %}
{% if question.target_answer.analyze_nlq and not exp_params.data_gen_hide_target_answer_analyze_nlq %}
[**Answer Generation - NLQ Analysis**]
{{question.target_answer.analyze_nlq}}
{% endif %}
{% if question.target_answer.plan and not exp_params.data_gen_hide_target_answer_plan %}
[**Answer Generation Plan**]
{{question.target_answer.plan}}
{% endif %}
{% if question.target_answer.answer_reasoning and not exp_params.data_gen_hide_target_answer_reasoning %}
[**Answer Generation Reasoning**]
{{question.target_answer.answer_reasoning}}
{% endif %}

[**Target Answer**]
{{question.target_answer.answer_sentence}}
AS JSON:
{{question.target_answer.answer_dictionary | tojson(indent=4) }}
{% endif %}

* SOLUTION *
""".strip()
