# `task_illustrations__library__books_in_section_count`

## Summary
- Domain: `illustrations`
- Scene id: `library`
- Implementation source: `src/trace_tasks/tasks/illustrations/library/books_in_section_count.py`

## Task Contract
Counts visible books in one labeled library shelf section.

## Program Contract

Program: `count(filter(books, section(book)=target_section)); scene=library; scope=books_in_section_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `books_in_section_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `books`, `section`, `book`, `target_section`, `library`, `books_in_section_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; the number of counted book witnesses projected from the same execution trace as the annotation.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(filter(books, section(book)=target_section)); scene=library; scope=books_in_section_count` |

## Program Metadata
- Program signatures: `count.scoped_attribute`
- Base program contract: `count(filter(books, section(book)=target_section)); scene=library; scope=books_in_section_count`
- Parameter axes: `target_section`
- Arguments:
  - `books`: semantic_role; allowed `visible_library_books`; source `program_schema_concrete`
  - `target_section`: semantic_role; allowed `sampled_library_section`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer_count`
- Generator `answer_gt.type`: `integer`
- The answer value is the number of counted book witnesses projected from the same execution trace as the annotation.

## Annotation Contract
- Annotation schema: `point_set`
- Generator `annotation_gt.type`: `point_set`
- Annotation is an unordered set of final-image pixel center points, one per counted book. Do not include shelf labels, section boxes, decor, or prompt/answer labels.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/library/illustrations_library_v0.json`.
- The prompt asks for one resolved section name; section sampling is trace metadata, not a public query branch.
- Render randomness, sampled fonts/styles, query operands, and verifier payloads must be explicit in the instance trace.
