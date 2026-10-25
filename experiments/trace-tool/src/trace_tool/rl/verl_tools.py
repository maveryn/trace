"""verl ``BaseTool`` adapters around trace_tool.tools (UNTESTED; needs verl).

tool_config.yaml entry:

    tools:
      - class_name: trace_tool.rl.verl_tools.ImageZoomInTool
        config: {type: native, family: qwen2.5-vl}

Leave ``tool_schema`` out of the YAML: verl validates YAML schemas into
pydantic models that drop ``items``/``minItems``/``enum``, which would make the
RL prompt differ from the evaluation prompt. The adapters hand verl a
duck-typed schema whose ``model_dump()`` returns our exact dict.

Image indexing follows the eval harness: ``agent_data.image_data`` holds the
prompt image(s) followed by every image returned by a tool, in order. Unlike
the eval harness, index 0 is the image as verl loaded it (already resized to
the processor's pixel bounds), not the original file; for TRACE at the default
bounds this differs only by rounding to the patch grid (28 px for Qwen2.5-VL). verl also runs the
calls of one turn in parallel against the images from before that turn, while
the eval harness runs them in order (as Qwen-Agent does), so a second call in
the same turn can zoom into the first call's output only in evaluation.
"""

from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace
from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

from trace_tool.models import get_family
from trace_tool.tools import TOOL_FACTORIES
from trace_tool.tools.base import ToolContext


class _ExactSchema:
    """Quacks like OpenAIFunctionToolSchema for BaseTool / ToolAgentLoop."""

    def __init__(self, schema: dict[str, Any]):
        self._schema = schema
        self.type = schema["type"]
        self.function = SimpleNamespace(name=schema["function"]["name"])

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return copy.deepcopy(self._schema)


class _TraceTool(BaseTool):
    tool_name: str = ""

    def __init__(self, config: dict, tool_schema: Any = None):
        # coordinate frame / patch grid of the policy (tool_config: config.family)
        self._family = get_family(config.get("family", "qwen2.5-vl"))
        self._impl = TOOL_FACTORIES[self.tool_name](self._family)
        super().__init__(config, _ExactSchema(self._impl.schema()))

    def get_openai_tool_schema(self) -> Any:
        return _ExactSchema(self._impl.schema())

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs: Any) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        images = list(getattr(agent_data, "image_data", None) or [])
        timeout = float(self.config.get("timeout_s", 60))
        try:
            # verl's image_data holds the images as the policy sees them, so shown size == stored size
            ctx = ToolContext(images)
            result = await asyncio.wait_for(asyncio.to_thread(self._impl, parameters, ctx), timeout=timeout)
        except asyncio.TimeoutError:
            return ToolResponse(text=f"Error: {self.name} timed out after {timeout:.0f}s"), 0.0, {"ok": 0.0}
        response: dict[str, Any] = {}
        if result.images:
            response["image"] = list(result.images)
        response["text"] = result.text or ("" if result.images else "Tool returned no output.")
        return ToolResponse(**response), 0.0, {"ok": float(result.ok)}


class ImageZoomInTool(_TraceTool):
    tool_name = "image_zoom_in_tool"


class ImageRotateTool(_TraceTool):
    tool_name = "image_rotate_tool"


class ImageFlipTool(_TraceTool):
    tool_name = "image_flip_tool"


class ImageDrawTool(_TraceTool):
    tool_name = "image_draw_tool"


class CalculatorTool(_TraceTool):
    tool_name = "calculator"


class OCRTool(_TraceTool):
    tool_name = "ocr"
