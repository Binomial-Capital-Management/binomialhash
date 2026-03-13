"""Tests for binomialhash.adapters — schema translation and handler dispatch."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from binomialhash.tools.base import ToolSpec, _prop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _dummy_handler(**kwargs: Any) -> Dict[str, Any]:
    return {"echo": kwargs}


def _make_specs() -> List[ToolSpec]:
    return [
        ToolSpec(
            name="bh_test_tool",
            description="A test tool for adapter tests.",
            input_schema={
                "type": "object",
                "properties": {
                    "key": _prop("string", "Dataset key."),
                    "limit": _prop("integer", "Row limit.", default=25),
                },
                "required": ["key"],
            },
            handler=lambda key, limit=25: _dummy_handler(key=key, limit=limit),
            group="test",
        ),
        ToolSpec(
            name="bh_second_tool",
            description="Another test tool.",
            input_schema={
                "type": "object",
                "properties": {
                    "x": _prop("number", "A number."),
                },
                "required": ["x"],
            },
            handler=lambda x=0: _dummy_handler(x=x),
            group="test",
        ),
    ]


SPECS = _make_specs()


# ---------------------------------------------------------------------------
# common.py
# ---------------------------------------------------------------------------

class TestCommonDispatch:
    def test_handle_tool_call_success(self):
        from binomialhash.adapters.common import handle_tool_call
        result = handle_tool_call(SPECS, "bh_test_tool", {"key": "abc", "limit": 10})
        assert result == {"echo": {"key": "abc", "limit": 10}}

    def test_handle_tool_call_unknown_raises(self):
        from binomialhash.adapters.common import handle_tool_call
        with pytest.raises(KeyError, match="Unknown tool"):
            handle_tool_call(SPECS, "nonexistent", {})

    def test_safe_handle_returns_result(self):
        from binomialhash.adapters.common import safe_handle_tool_call
        out = safe_handle_tool_call(SPECS, "bh_test_tool", {"key": "k"})
        assert "result" in out
        assert out["result"]["echo"]["key"] == "k"

    def test_safe_handle_returns_error(self):
        from binomialhash.adapters.common import safe_handle_tool_call
        out = safe_handle_tool_call(SPECS, "nonexistent", {})
        assert "error" in out
        assert "Unknown tool" in out["error"]

    def test_parse_arguments_dict(self):
        from binomialhash.adapters.common import parse_arguments
        assert parse_arguments({"a": 1}) == {"a": 1}

    def test_parse_arguments_json_string(self):
        from binomialhash.adapters.common import parse_arguments
        assert parse_arguments('{"a": 1}') == {"a": 1}

    def test_parse_arguments_none(self):
        from binomialhash.adapters.common import parse_arguments
        assert parse_arguments(None) == {}

    def test_parse_arguments_bad_string(self):
        from binomialhash.adapters.common import parse_arguments
        assert parse_arguments("not json") == {}


# ---------------------------------------------------------------------------
# openai.py
# ---------------------------------------------------------------------------

class TestOpenAIAdapter:
    def test_responses_format_shape(self):
        from binomialhash.adapters.openai import get_openai_tools
        tools = get_openai_tools(SPECS)
        assert len(tools) == 2
        t = tools[0]
        assert t["type"] == "function"
        assert t["name"] == "bh_test_tool"
        assert "description" in t
        assert "parameters" in t
        assert "input_schema" not in t

    def test_chat_completions_format_shape(self):
        from binomialhash.adapters.openai import get_openai_tools
        tools = get_openai_tools(SPECS, format="chat_completions")
        t = tools[0]
        assert t["type"] == "function"
        assert "function" in t
        fn = t["function"]
        assert fn["name"] == "bh_test_tool"
        assert "parameters" in fn

    def test_strict_mode_adds_fields(self):
        from binomialhash.adapters.openai import get_openai_tools
        tools = get_openai_tools(SPECS, strict=True)
        t = tools[0]
        assert t["strict"] is True
        assert t["parameters"]["additionalProperties"] is False
        assert "key" in t["parameters"]["required"]
        assert "limit" in t["parameters"]["required"]

    def test_strict_does_not_mutate_original(self):
        from binomialhash.adapters.openai import get_openai_tools
        original_required = list(SPECS[0].input_schema.get("required", []))
        get_openai_tools(SPECS, strict=True)
        assert SPECS[0].input_schema.get("required") == original_required

    def test_handle_json_string_arguments(self):
        from binomialhash.adapters.openai import handle_openai_tool_call
        result = handle_openai_tool_call(SPECS, "bh_test_tool", '{"key": "xyz"}')
        assert result["echo"]["key"] == "xyz"

    def test_handle_dict_arguments(self):
        from binomialhash.adapters.openai import handle_openai_tool_call
        result = handle_openai_tool_call(SPECS, "bh_test_tool", {"key": "xyz"})
        assert result["echo"]["key"] == "xyz"


# ---------------------------------------------------------------------------
# anthropic.py
# ---------------------------------------------------------------------------

class TestAnthropicAdapter:
    def test_schema_shape(self):
        from binomialhash.adapters.anthropic import get_anthropic_tools
        tools = get_anthropic_tools(SPECS)
        t = tools[0]
        assert "name" in t
        assert "description" in t
        assert "input_schema" in t
        assert "type" not in t
        assert "parameters" not in t

    def test_input_schema_preserved(self):
        from binomialhash.adapters.anthropic import get_anthropic_tools
        tools = get_anthropic_tools(SPECS)
        schema = tools[0]["input_schema"]
        assert schema["type"] == "object"
        assert "key" in schema["properties"]

    def test_examples_attached(self):
        from binomialhash.adapters.anthropic import get_anthropic_tools
        examples = {"bh_test_tool": [{"key": "example_key", "limit": 10}]}
        tools = get_anthropic_tools(SPECS, examples=examples)
        assert "input_examples" not in tools[0]
        assert "Example inputs:" in tools[0]["description"]
        assert '"key": "example_key"' in tools[0]["description"]
        assert "Example inputs:" not in tools[1]["description"]

    def test_name_validation_passes(self):
        from binomialhash.adapters.anthropic import get_anthropic_tools
        get_anthropic_tools(SPECS)

    def test_name_validation_rejects_bad_name(self):
        from binomialhash.adapters.anthropic import get_anthropic_tools
        bad = [ToolSpec(name="has spaces!", description="x",
                        input_schema={"type": "object", "properties": {}},
                        handler=lambda: None)]
        with pytest.raises(ValueError, match="does not match"):
            get_anthropic_tools(bad)

    def test_handle_dict_input(self):
        from binomialhash.adapters.anthropic import handle_anthropic_tool_use
        result = handle_anthropic_tool_use(SPECS, "bh_test_tool", {"key": "abc"})
        assert result["echo"]["key"] == "abc"


# ---------------------------------------------------------------------------
# gemini.py
# ---------------------------------------------------------------------------

class TestGeminiAdapter:
    def test_declaration_shape(self):
        from binomialhash.adapters.gemini import get_gemini_tools
        decls = get_gemini_tools(SPECS)
        assert len(decls) == 2
        d = decls[0]
        assert "name" in d
        assert "description" in d
        assert "parameters" in d
        assert "type" not in d
        assert "input_schema" not in d

    def test_parameters_has_properties(self):
        from binomialhash.adapters.gemini import get_gemini_tools
        decls = get_gemini_tools(SPECS)
        params = decls[0]["parameters"]
        assert params["type"] == "object"
        assert "key" in params["properties"]

    def test_name_validation_rejects_digit_start(self):
        from binomialhash.adapters.gemini import get_gemini_tools
        bad = [ToolSpec(name="123bad", description="x",
                        input_schema={"type": "object", "properties": {}},
                        handler=lambda: None)]
        with pytest.raises(ValueError, match="does not match"):
            get_gemini_tools(bad)

    def test_handle_dict_args(self):
        from binomialhash.adapters.gemini import handle_gemini_tool_call
        result = handle_gemini_tool_call(SPECS, "bh_second_tool", {"x": 42})
        assert result["echo"]["x"] == 42

    def test_handle_none_args(self):
        from binomialhash.adapters.gemini import handle_gemini_tool_call
        result = handle_gemini_tool_call(SPECS, "bh_second_tool", None)
        assert result["echo"]["x"] == 0


# ---------------------------------------------------------------------------
# xai.py
# ---------------------------------------------------------------------------

class TestXAIAdapter:
    def test_format_matches_openai(self):
        from binomialhash.adapters.openai import get_openai_tools
        from binomialhash.adapters.xai import get_xai_tools
        openai_tools = get_openai_tools(SPECS, format="responses")
        xai_tools = get_xai_tools(SPECS)
        assert openai_tools == xai_tools

    def test_handle_dispatches(self):
        from binomialhash.adapters.xai import handle_xai_tool_call
        result = handle_xai_tool_call(SPECS, "bh_test_tool", '{"key": "z"}')
        assert result["echo"]["key"] == "z"


# ---------------------------------------------------------------------------
# __init__.py router
# ---------------------------------------------------------------------------

class TestProviderRouter:
    def test_all_providers(self):
        from binomialhash.adapters import get_tools_for_provider
        for provider in ("openai", "anthropic", "gemini", "xai"):
            tools = get_tools_for_provider(SPECS, provider=provider)
            assert len(tools) == 2, f"{provider} returned {len(tools)} tools"

    def test_unknown_provider_raises(self):
        from binomialhash.adapters import get_tools_for_provider
        with pytest.raises(ValueError, match="Unknown provider"):
            get_tools_for_provider(SPECS, provider="deepseek")
