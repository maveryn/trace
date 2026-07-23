"""Machine-readable validation/build error codes."""

SCHEMA_MISSING_FIELD = "schema_missing_field"
SCHEMA_TYPE_MISMATCH = "schema_type_mismatch"
SCHEMA_INVALID_VALUE = "schema_invalid_value"
SCHEMA_NON_STRING_KEY = "schema_non_string_key"
SCHEMA_NON_FINITE_NUMBER = "schema_non_finite_number"
SCHEMA_UNSUPPORTED_TYPE = "schema_unsupported_type"
SCHEMA_CANONICALIZATION_FAILED = "schema_canonicalization_failed"

TRACE_REF_MISSING = "trace_ref_missing"
TRACE_REF_NOT_FOUND = "trace_ref_not_found"
TRACE_REF_HASH_MISMATCH = "trace_ref_hash_mismatch"
TRACE_REF_INDEX_OUT_OF_RANGE = "trace_ref_index_out_of_range"
TRACE_SHARD_READ_FAILED = "trace_shard_read_failed"
TRACE_SHARD_MANIFEST_MISMATCH = "trace_shard_manifest_mismatch"

IMAGE_PATH_NOT_RELATIVE = "image_path_not_relative"
IMAGE_FILE_NOT_FOUND = "image_file_not_found"
IMAGE_HASH_MISSING = "image_hash_missing"
IMAGE_HASH_MISMATCH = "image_hash_mismatch"

COUNT_PER_TASK_SHORTFALL = "count_per_task_shortfall"
COUNT_UNEXPECTED_TASK_PRESENT = "count_unexpected_task_present"

VERSION_MIXED_INSTANCE_VERSION = "version_mixed_instance_version"
VERSION_UNSUPPORTED_INSTANCE_VERSION = "version_unsupported_instance_version"

IDENTITY_INSTANCE_ID_MISMATCH = "identity_instance_id_mismatch"
IDENTITY_NON_DETERMINISTIC_ORDER = "identity_non_deterministic_order"

IO_TRACE_WRITE_FAILED = "io_trace_write_failed"
IO_ATOMIC_FINALIZE_FAILED = "io_atomic_finalize_failed"
IO_VALIDATION_REPORT_WRITE_FAILED = "io_validation_report_write_failed"
IO_FAILURE_BUNDLE_WRITE_FAILED = "io_failure_bundle_write_failed"

CONFIG_TASK_NOT_REGISTERED = "config_task_not_registered"
CONFIG_REGISTRY_FILE_MISSING = "config_registry_file_missing"
CONFIG_REGISTRY_HASH_MISMATCH = "config_registry_hash_mismatch"
CONFIG_BUILD_REPORT_SCHEMA_MISMATCH = "config_build_report_schema_mismatch"

PROMPT_METADATA_MISSING = "prompt_metadata_missing"
PROMPT_BUNDLE_NOT_FOUND = "prompt_bundle_not_found"
PROMPT_BUNDLE_INVALID = "prompt_bundle_invalid"
PROMPT_KEY_MISSING = "prompt_key_missing"
PROMPT_VARIANT_COUNT_MISMATCH = "prompt_variant_count_mismatch"
PROMPT_VARIANT_INDEX_OUT_OF_RANGE = "prompt_query_id_index_out_of_range"
PROMPT_REQUIRED_SLOT_MISSING = "prompt_required_slot_missing"
PROMPT_UNRESOLVED_PLACEHOLDER = "prompt_unresolved_placeholder"

TEXT_LEGIBILITY_INVALID = "text_legibility_invalid"
TEXT_LEGIBILITY_CONTRAST_FAILED = "text_legibility_contrast_failed"

MARKER_LEGIBILITY_INVALID = "marker_legibility_invalid"
MARKER_LEGIBILITY_CONTRAST_FAILED = "marker_legibility_contrast_failed"
