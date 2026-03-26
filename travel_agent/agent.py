"""Travel-Agent core reasoning loop."""

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

from openai import OpenAI

from travel_agent.helpers import normalize_tool_calls, safe_json_loads
from travel_agent.tools.places import search_places
from travel_agent.tools.routing import calculate_route
from travel_agent.tools.weather import get_weather


@dataclasses.dataclass
class AgentInput:
    city: str
    budget: int
    days: int


class TravelAgent:
    """Tool-augmented travel planner."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def run(self, agent_input: AgentInput, max_steps: int = 20, stream_output: bool = True) -> str:
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are Travel-Agent, a professional, concise, cost-aware travel planner for college students. "
                    "Always call tools first when data is needed; never invent tool results. "
                    "Weather response: if precipitation > 60% or crowd_risk is high, prioritize indoor venues, shorten outdoor blocks, and add risk/mitigation notes. "
                    "Routing response: if a route is missing or duration > 120 minutes, switch to nearer POIs or faster modes; do not stay stuck retrying long hops. "
                    "If any tool fails, still produce the best-possible plan with clear notes on missing data. "
                    "Output strictly as a Markdown table with columns: 时间段 | 活动内容 | 预估开支 | 交通方式 | 专家笔记. "
                    "Use CNY budgets, keep options pragmatic/bookable, and ensure the day is filled按小时。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"城市: {agent_input.city}\n预算(元): {agent_input.budget}\n旅行天数: {agent_input.days}\n"
                    "请给出精确到小时的行程规划。"
                ),
            },
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get precipitation probability and temperature range for a city/date",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                        },
                        "required": ["city", "date"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_places",
                    "description": "Find attractions/food in a city with price filter",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "query": {"type": "string"},
                            "price_level": {
                                "type": "string",
                                "enum": ["budget", "mid", "premium", "any"],
                            },
                        },
                        "required": ["city", "query", "price_level"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_route",
                    "description": "Compute route duration between two points",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string"},
                            "destination": {"type": "string"},
                            "mode": {
                                "type": "string",
                                "enum": ["walk", "transit", "drive"],
                            },
                        },
                        "required": ["origin", "destination", "mode"],
                    },
                },
            },
        ]

        for _ in range(max_steps):
            response = self._chat_once(messages, tools)
            message = response["message"]
            tool_calls = message.get("tool_calls") or []

            if tool_calls:
                messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": tool_calls})
                for call in tool_calls:
                    name = call.get("function", {}).get("name", "")
                    args = call.get("function", {}).get("arguments", "{}")
                    parsed_args = safe_json_loads(args)
                    result = self._execute_tool(name, parsed_args)
                    self._debug_tool_call(name, parsed_args, result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                continue

            content = message.get("content")
            if content:
                if stream_output:
                    # Preserve the non-streamed content so we can fall back if streaming yields nothing
                    assistant_content = content
                    streamed = self._chat_stream(messages, tools)
                    tool_calls_stream = streamed.get("tool_calls") or []
                    if tool_calls_stream:
                        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls_stream})
                        for call in tool_calls_stream:
                            name = call.get("function", {}).get("name", "")
                            args = call.get("function", {}).get("arguments", "{}")
                            parsed_args = safe_json_loads(args)
                            result = self._execute_tool(name, parsed_args)
                            self._debug_tool_call(name, parsed_args, result)
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.get("id", ""),
                                    "content": json.dumps(result, ensure_ascii=False),
                                }
                            )
                        continue
                    streamed_content = streamed.get("content", "")
                    if streamed_content.strip() == "规划未完成，请稍后重试。":
                        return self._dedupe_content(assistant_content)
                    return self._dedupe_content(streamed_content or assistant_content)
                return self._dedupe_content(content)

        return "规划未完成，请稍后重试。"

    def _chat_once(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.6,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "message": {
                    "role": "assistant",
                    "content": f"API 调用失败: {exc}",
                    "tool_calls": [],
                }
            }

        choice = completion.choices[0].message
        return {
            "message": {
                "role": choice.role,
                "content": choice.content,
                "tool_calls": normalize_tool_calls(getattr(choice, "tool_calls", None)),
            }
        }

    def _chat_stream(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stream final assistant content; capture tool_calls if emitted mid-stream."""

        content_parts: List[str] = []
        collected_tool_calls: List[Dict[str, Any]] = []
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.6,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    sys.stdout.write(delta.content)
                    sys.stdout.flush()
                    content_parts.append(delta.content)
                if getattr(delta, "tool_calls", None):
                    collected_tool_calls.extend(normalize_tool_calls(delta.tool_calls))
            if collected_tool_calls:
                return {"tool_calls": collected_tool_calls}
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception as exc:  # noqa: BLE001
            return {"content": f"API 流式调用失败: {exc}"}

        content = "".join(content_parts) if content_parts else ""
        return {"content": self._dedupe_content(content)}

    def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not name:
                return {"error": "Unknown tool: missing name"}
            if name == "get_weather":
                return get_weather(
                    city=str(arguments.get("city", "")),
                    date=str(arguments.get("date", datetime.now(timezone.utc).date().isoformat())),
                )
            if name == "search_places":
                return search_places(
                    city=str(arguments.get("city", "")),
                    query=str(arguments.get("query", "")),
                    price_level=str(arguments.get("price_level", "budget")),
                )
            if name == "calculate_route":
                return calculate_route(
                    origin=str(arguments.get("origin", "")),
                    destination=str(arguments.get("destination", "")),
                    mode=str(arguments.get("mode", "walk")),
                )
            return {"error": f"Unknown tool: {name}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Tool execution failed: {exc}"}

    def _debug_tool_call(self, name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Output debug info for tool calls."""

        success = "error" not in result
        log = {
            "tool": name,
            "success": success,
            "arguments": args,
            "result": result,
        }
        sys.stdout.write(f"[tool-debug] {json.dumps(log, ensure_ascii=False)}\n")
        sys.stdout.flush()

    def _dedupe_content(self, content: str) -> str:
        """Remove simple duplicated tail when the model emits the same block twice."""

        if not content:
            return content
        text = content.strip()
        length = len(text)
        if length % 2 == 0:
            mid = length // 2
            if text[:mid] == text[mid:]:
                return text[:mid]
        return text
