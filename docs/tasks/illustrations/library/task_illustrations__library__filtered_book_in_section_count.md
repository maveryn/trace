# `task_illustrations__library__filtered_book_in_section_count`

## Summary
- Domain: `illustrations`
- Scene id: `library`
- Implementation source: `src/trace_tasks/tasks/illustrations/library/filtered_book_in_section_count.py`

## Task Contract
Counts visible books in one labeled library shelf section filtered by a prompt-visible book attribute predicate.

## Program Contract

Program: `count(filter(books, section(book)=target_section and book_attribute(book)=target_attribute_value)); scene=library; scope=filtered_book_in_section_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `filtered_book_in_section_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `books`, `section`, `book`, `target_section`, `book_attribute`, `target_attribute_value`, `library`, `filtered_book_in_section_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; the number of counted book witnesses projected from the same execution trace as the annotation.
Annotation witnesses: `annotation` uses the `point_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `book_color_in_section_count`, `horizontal_book_in_section_count`, `upright_book_in_section_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `book_color_in_section_count` | `count(filter(books, section(book)=target_section and color(book)=target_color)); scene=library; scope=filtered_book_in_section_count` |
| `horizontal_book_in_section_count` | `count(filter(books, section(book)=target_section and orientation(book)=horizontal)); scene=library; scope=filtered_book_in_section_count` |
| `upright_book_in_section_count` | `count(filter(books, section(book)=target_section and orientation(book)=upright)); scene=library; scope=filtered_book_in_section_count` |

## Program Metadata
- Program signatures: `count.scoped_attribute`
- Base program contract: `count(filter(books, section(book)=target_section and book_attribute(book)=target_attribute_value)); scene=library; scope=filtered_book_in_section_count`
- Parameter axes: `target_section`, `target_attribute`, `target_attribute_value`
- Arguments:
  - `books`: semantic_role; allowed `visible_library_books`; source `program_schema_concrete`
  - `target_section`: semantic_role; allowed `sampled_library_section`; source `program_schema_concrete`
  - `target_attribute`: object_attribute; allowed `color`, `orientation`; source `query_id`
  - `target_attribute_value`: object_attribute; allowed `sampled_color`, `upright`, `horizontal`; source `program_schema_concrete|query_id`
- Argument metadata status: `curated`
- Supported query ids: `book_color_in_section_count`, `horizontal_book_in_section_count`, `upright_book_in_section_count`

## Answer Contract
- Answer schema: `integer_count`
- Generator `answer_gt.type`: `integer`
- The answer value is the number of counted book witnesses projected from the same execution trace as the annotation.

## Annotation Contract
- Annotation schema: `point_set`
- Generator `annotation_gt.type`: `point_set`
- Annotation is an unordered set of final-image pixel center points, one per counted book matching the section and attribute predicate. Do not include non-matching books, shelf labels, section boxes, or decor.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/library/illustrations_library_v0.json`.
- Query ids are semantic because the prompt-visible predicate changes between color, upright orientation, and horizontal orientation.
- Sampled section names and sampled colors are trace metadata unless they are rendered into the prompt as resolved operands.
