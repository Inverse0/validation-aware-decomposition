# Prompts

The pipeline loads every LLM prompt from this folder. The `.txt` files are Python `str.format` templates (doubled braces are literal JSON braces); the RT and SE modules hold the Jinja templates of Alrashed et al.

| Prompt | File | Model |
| --- | --- | --- |
| Requirement of a non-root block, from the question and the blocks that read it | `tasks/requirement_prompt.txt`, with `tasks/reader_template.txt` for each reader | GPT-6 Luna, max effort |
| Description of what an input computes, from its SQL | `tasks/describe_prompt.txt` | GPT-6 Luna, max effort |
| Judging instructions added to a block's requirement | `tasks/stance_root.txt`, `tasks/stance_supplier.txt` | part of the validator's prompt |
| Coherence of an intermediate relation | `semantic_llm/coherence_prompt.txt` | GPT-6 Luna, high effort |
| Relevance of an intermediate relation to the question | `semantic_llm/relevance_prompt.txt` | GPT-6 Luna, high effort |
| Validation of a block or a whole query | `verify/prompt.txt` | the validator, medium effort |
| RT: back-translation and question similarity | `alrashed/roundtrip/assets/prompts.py` | the validator, medium effort |
| SE: expected answer | `alrashed/datagen/assets/answer_gen/prompts.py` | the validator, medium effort |
| SE: database instance | `alrashed/datagen/assets/data_gen/prompts.py` | the validator, medium effort |

`tasks/schema_legend.txt`, `tasks/not_null_sentence.txt` and `tasks/unique_sentence.txt` explain the schema notation above each schema in a prompt, and `semantic_llm/closed_form.txt` explains a relation inside a correlated subquery.
