# Data Format

Every finalized Trace training instance records:

- `instance_id`: content-derived stable identifier
- `task`, `domain`, `scene_id`, and `query_id`: public taxonomy identity
- `prompt` and `prompt_variants`: active and alternate output contracts
- `images`: image ids, formats, hashes, and dataset-relative paths
- `answer_gt`: typed final answer
- `annotation_gt`: typed image-space evidence
- `reward_contract`: deterministic scoring contract
- `trace_ref`: sidecar shard, line index, and trace-record hash
- `versions`: generator, prompt, renderer, and contract versions

The sidecar trace contains the scene specification, query specification,
render specification, execution trace, symbolic witness, and projected
annotation used to construct the instance. Verifiers consume these metadata
contracts rather than treating rendered pixels as ground truth.

JSONL is the default portable export format. Optional Parquet export can retain
relative image paths or embed encoded image bytes.
