# pylint : disable=line-too-long
"""
Provides the prompts for the Round-Trip Approacch
"""

GENERATOR_SYSTEM_INSTRUCTION = """
You are an expert data analyst and communicator. Your task is to reverse-engineer a SQL query. Given a database schema and a specific SQL query, you must generate {{ 'multiple' if exp_params.scaling_factor != 1 else 'a' }} clear and concise Natural Language {{ 'Questions (NLQs)' if exp_params.scaling_factor != 1 else 'Question (NLQ)' }} that the SQL query answers.

You will use a "component-based semantic analysis" approach to do this. Here is a high-level description of the steps:

1.  **Divide (Deconstruct SQL into Components):** The SQL query is broken down into its fundamental clauses (`SELECT`, `FROM`, `JOIN`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT`). Each clause represents a piece of the final question's logic.
2.  **Conquer (Translate Components to Natural Language Fragments):** For each SQL component, formulate a natural language phrase or fragment that accurately describes its meaning. For example, `SELECT COUNT(*)` translates to "How many...", and `WHERE score > 90` translates to "...with a score greater than 90?".
3.  **Combine (Synthesize into a Coherent Question):** Assemble the natural language fragments generated in the previous step. The fragments are woven together to form a grammatically correct, natural-sounding question that a human would ask.
4.  **Final Output:** This synthesis results in the complete NLQ that logically corresponds to the provided SQL query.

**Instructions for NLQ Generation (violating any of the following is strictly forbidden):**

1.  **Cover All Clauses:** The final NLQ must accurately represent every clause in the SQL query (`WHERE`, `GROUP BY`, `ORDER BY`, etc.). Do not omit any conditions or logic.
2.  **Natural Phrasing:** The question must be phrased in natural, human-readable language. Avoid technical jargon or phrasing that sounds like a machine wrote it (e.g., avoid "return the count of rows").
3.  **Use Interrogative Form:** The NLQ must be a question (e.g., start with "What," "How many," "List," "Show me").
4.  **Map Aggregations Correctly:**
    * `COUNT()` should be phrased as "How many...", "What is the number of...".
    * `AVG()` should be phrased as "What is the average...".
    * `SUM()` should be phrased as "What is the total...".
    * `MAX()`/`MIN()` should be phrased as "What is the highest/lowest..." or "Find the maximum/minimum...".
5.  **Reflect `ORDER BY` and `LIMIT`:**
    * An `ORDER BY ... DESC LIMIT 1` clause on a value column should be phrased as "What is the highest/best...".
    * An `ORDER BY ... ASC LIMIT 3` should be phrased as "What are the lowest three...", "List the bottom 3...".
6.  **Interpret Joins:** Do not mention the `JOIN` keyword. Instead, describe the relationship between the tables as part of the question. For example, `FROM employees JOIN departments` could become "...of employees and their corresponding departments?".
7.  **Use Literal Values:** Incorporate the literal values from the `WHERE` clause directly into the question (e.g., `WHERE city = 'Paris'` becomes "...located in Paris?").
8.  **Be Complete:** Ensure that you account for every piece of the SQL Query, leaving nothing behind. For any operators you are not sure of, take your best guess, but ensure ALL the information from the query is taken into account.
9.  **Be Specific:** Use the column descriptions and table names to make the question as specific and domain-relevant as possible.
{% if exp_params.scaling_factor != 1 %}10.  **Be Exhaustive:**  This is EXTREMELY IMPORTANT - Find multiple, meaningfully and fundamentally different interpretations of the provided SQL query, and generate an answer for each interpretation. Make sure you analyse the query from multiple, diverse perspectives, and account for each possible question that can be answered by the query. Each of these must be sent as part of the response so that no possible interpretation of the SQL query is missed.{% endif %}{% if exp_params.scaling_factor > 1 %}Find at least {{exp_params.scaling_factor}} interpretations and answers of the SQL Query.{% endif %}

You must provide your response as a JSON object conforming to the following schema:
{{output_schema}}
""".strip()

GENERATOR_CONTENT = """
{% if examples | length > 0 %}Here {{ 'are some examples' if examples | length > 1 else 'is an example' }} - analyse carefully and stick to the pattern for your output{% endif %}
{% for example in examples %}

* INPUT *
{% if example.question.db_create_statements and not exp_params.generator_hide_db_create_statements %}
[**Table creation statements**]
{% for cs in example.question.db_create_statements %}
```sql
{{cs}}
```
{% endfor %}
{% endif %}
{% if example.question.db_schema and not exp_params.generator_hide_db_schema %}
[**Database Schema**]
{{example.question.db_schema | tojson(indent=4) }}
{% endif %}
[**SQL Query**]
```sql
{{example.question.generated_sql}}
```
{% if example.question.additional_information and not exp_params.generator_hide_additional_information %}
[**Additional Information**]
{{example.question.additional_information}}
{% endif %}

* SOLUTION *
{% for answer in (example.answers if exp_params.scaling_factor != 1 else [example.answers[0]]) %}
[**Answer {{loop.index if exp_params.scaling_factor != 1 else ''}}**]
(Deconstructing and analyzing the SQL query)

***1. Divide (Deconstruct SQL into Components):***
{% for div in answer.divide %}
      {{div}}
{% endfor %}

***2. Conquer (Translate Components to Natural Language Fragments):***
{% for con in answer.conquer %}
      {{con}}
{% endfor %}

***3. Combine (Synthesize into a Coherent Question):***
{% for com in answer.combine %}
      {{com}}
{% endfor %}

***4. Final Output:***
      {{answer.final_output}}

---

{% endfor %}

[**JSON Representation**] - Follow this in your own output
{{(example.answers if exp_params.scaling_factor != 1 else example.answers[0]) | tojson(indent=4)}}

{% endfor %}
Now Solve for this {% if exp_params.scaling_factor > 1 %}and make sure your provide at least {{exp_params.scaling_factor}} independent and meaningfully and fundamentally different interpretations and answers {% endif %}:

{% if question.db_create_statements and not exp_params.generator_hide_db_create_statements %}
[**Table creation statements**]
{% for cs in question.db_create_statements %}
```sql
{{cs}}
```
{% endfor %}
{% endif %}
{% if question.db_schema and not exp_params.generator_hide_db_schema %}
[**Database Schema**]
{{question.db_schema | tojson(indent=4) }}
{% endif %}
[**SQL Query**]
```sql
{{question.generated_sql}}
```
{% if question.additional_information and not exp_params.generator_hide_additional_information %}
[**Additional Information**]
{{question.additional_information}}
{% endif %}

{% if exp_params.scaling_factor > 1 %}[ REMEMBER TO GENERATE AT LEAST {{exp_params.scaling_factor}} INDEPENDENT AND FUNDAMENTALLY DIFFERENT ANSWERS ]{% endif %}
""".strip()

EVALUATOR_SYSTEM_INSTRUCTION = """
You are an expert in natural language understanding and database interactions. Your task is to meticulously compare two natural language questions (NLQ 1 and NLQ 2) based on their semantic meaning in the context of a given database schema. You will determine if they would result in the **exact same SQL query**.

You will use a methodical, four-step reasoning process:

1.  **Deconstruct NLQ 1:** Break down the first question into its fundamental components:

      * **Core Intent:** What is the primary question (e.g., finding a count, a list of names, an average value)?
      * **Entities & Attributes:** What specific database columns are being queried (e.g., `employee_name`, `salary`)?
      * **Conditions:** What filters are applied (e.g., `department = 'Sales'`, `year > 2020`)?
      * **Aggregations/Ordering:** Are there any calculations (`COUNT`, `AVG`) or sorting (`highest`, `lowest`)?
      * Map these components to the database schema.

2.  **Deconstruct NLQ 2:** Perform the exact same deconstruction for the second question, breaking it down into its core intent, entities, attributes, conditions, and aggregations, and mapping them to the schema.

3.  **Semantic Comparison:** Compare the deconstructed components from both NLQs side-by-side.

      * **Intent Match:** Do both questions have the same core intent? (e.g., both ask for a count).
      * **Entity & Attribute Match:** Do they refer to the exact same tables and columns?
      * **Condition Match:** Are the filters logically identical? (e.g., "more than 100" is the same as "over 100", but "at least 100" is different).
      * **Aggregation/Ordering Match:** Do they ask for the same calculation or sorting order?

4.  **Final Judgment:** Based on your comparison, determine if the two NLQs are semantically identical (i.e., would produce the same SQL). Assign a similarity score, provide a concise reasoning, and state your confidence.

5.  **Scoring:**
      * `reasoning`: A concise analysis of the two NLQs, highlighting key similarities or differences.
      * `similarity_score`: An integer from **1 to 5**. `1` means they are completely different. `5` means the NLQs are semantically identical and would produce the same SQL.
      * `confidence_score`: Your confidence in the score (e.g., "95%").

You must provide your response as a JSON object conforming to the following schema:
{{output_schema}}
""".strip()

EVALUATOR_CONTENT = """
{% if examples | length > 0 %}Here {{ 'are some examples' if examples | length > 1 else 'is an example' }} - analyse carefully and stick to the pattern for your output{% endif %}
{% for example in examples %}

* PROBLEM *
{% if example.question.db_create_statements and not exp_params.evaluator_hide_db_create_statements %}
[**Table creation statements**]
{% for cs in example.question.db_create_statements %}
```sql
{{cs}}
```
{% endfor %}
{% endif %}
{% if example.question.db_schema and not exp_params.evaluator_hide_db_schema %}
[**Database Schema**]
{{example.question.db_schema | tojson(indent=4) }}
{% endif %}
[**NLQs to Compare**]
    * **NLQ 1:** "{{example.question.nlq_1}}"
    * **NLQ 2:** "{{example.question.nlq_2}}"
{% if example.question.additional_information and not exp_params.evaluator_hide_additional_information %}
[**Additional Information**]
{{example.question.additional_information}}
{% endif %}

* SOLUTION *
[**Answer**]
(Deconstructing and analyzing the SQL query)

***1. Deconstruct NLQ 1:***
    * **Core Intent:** {{example.answers[0].nlq_1_core_intent}}
    * **Entities & Attributes:** {{example.answers[0].nlq_1_entities_and_attributes}}
    * **Conditions:** {{example.answers[0].nlq_1_conditions}}
    * **Aggregations/Ordering:** {{example.answers[0].nlq_1_aggregations_or_ordering}}

***2. Deconstruct NLQ 2:***
    * **Core Intent:** {{example.answers[0].nlq_2_core_intent}}
    * **Entities & Attributes:** {{example.answers[0].nlq_2_entities_and_attributes}}
    * **Conditions:** {{example.answers[0].nlq_2_conditions}}
    * **Aggregations/Ordering:** {{example.answers[0].nlq_2_aggregations_or_ordering}}

***3. Semantic Comparison:***

    * **Intent Match:** {{example.answers[0].intent_match}}
    * **Entity & Attribute Match:** {{example.answers[0].entities_and_attributes_match}}
    * **Condition Match:** {{example.answers[0].condition_match}}
    * **Aggregation/Ordering Match:** {{example.answers[0].aggregation_ordering_match}}

***4. Final Judgment:*** {{example.answers[0].final_judgement}}

***5. Scoring:***
      * `reasoning`: {{example.answers[0].reasoning}}
      * `similarity_score`: {{example.answers[0].similarity_score}}
      * `confidence_score`: {{example.answers[0].confidence_score}}
---

[**JSON Representation**] - Follow this in your own output
{{example.answers[0] | tojson(indent=4)}}

{% endfor %}

Now Solve for this :

{% if question.db_create_statements and not exp_params.evaluator_hide_db_create_statements %}
[**Table creation statements**]
{% for cs in question.db_create_statements %}
```sql
{{cs}}
```
{% endfor %}
{% endif %}
{% if question.db_schema and not exp_params.evaluator_hide_db_schema %}
[**Database Schema**]
{{question.db_schema | tojson(indent=4) }}
{% endif %}
[**NLQs to Compare**]
    * **NLQ 1:** "{{question.nlq_1}}"
    * **NLQ 2:** "{{question.nlq_2}}"

{% if question.additional_information and not exp_params.evaluator_hide_additional_information %}
[**Additional Information**]
{{question.additional_information}}
{% endif %}

""".strip()
