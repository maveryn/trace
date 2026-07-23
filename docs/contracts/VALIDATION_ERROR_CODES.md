# Trace Validation Error Codes

Catalog of `validation_report.json` `error_code` values.

## Naming
Use lowercase snake case with category prefixes:
`schema_*`, `trace_ref_*`, `image_*`, `count_*`, `version_*`, `identity_*`, `io_*`, `config_*`, `prompt_*`.

## Codes by category
### Schema
- `schema_missing_field`
- `schema_type_mismatch`
- `schema_invalid_value`
- `schema_non_string_key`
- `schema_non_finite_number`
- `schema_unsupported_type`
- `schema_canonicalization_failed`

### Trace reference
- `trace_ref_missing`
- `trace_ref_not_found`
- `trace_ref_hash_mismatch`
- `trace_ref_index_out_of_range`
- `trace_shard_read_failed`
- `trace_shard_manifest_mismatch`

### Image integrity
- `image_path_not_relative`
- `image_file_not_found`
- `image_hash_missing`
- `image_hash_mismatch`

### Counts
- `count_per_task_shortfall`
- `count_unexpected_task_present`

### Version
- `version_mixed_instance_version`
- `version_unsupported_instance_version`

### Identity
- `identity_instance_id_mismatch`
- `identity_non_deterministic_order`

### I/O
- `io_trace_write_failed`
- `io_atomic_finalize_failed`
- `io_validation_report_write_failed`
- `io_failure_bundle_write_failed`

### Config/registry
- `config_task_not_registered`
- `config_registry_file_missing`
- `config_registry_hash_mismatch`
- `config_build_report_schema_mismatch`

### Prompt validation
- `prompt_metadata_missing`
- `prompt_bundle_not_found`
- `prompt_bundle_invalid`
- `prompt_key_missing`
- `prompt_variant_count_mismatch`
- `prompt_query_id_index_out_of_range`
- `prompt_required_slot_missing`
- `prompt_unresolved_placeholder`

## Maintenance
If a code is added/renamed, update:
1. validator implementation,
2. this catalog,
3. related tests.
