"""Travel-Agent entrypoint (orchestrates CLI and agent)."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

from dotenv import load_dotenv
from openai import OpenAI

from travel_agent.agent import AgentInput, TravelAgent


def _parse_cli_args() -> Tuple[AgentInput, str]:
	parser = argparse.ArgumentParser(description="Travel-Agent: hourly travel planner")
	parser.add_argument("--city", required=True, help="Destination city, e.g., 北京")
	parser.add_argument("--budget", type=int, required=True, help="Budget in CNY for the full trip")
	parser.add_argument("--days", type=int, required=True, help="Number of travel days")
	parser.add_argument(
		"--model",
		default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
		help="Model name (DeepSeek-compatible)",
	)
	args = parser.parse_args()
	return AgentInput(city=args.city, budget=args.budget, days=args.days), args.model


def main() -> int:
	load_dotenv()
	agent_input, model = _parse_cli_args()

	base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
	api_key = os.getenv("DEEPSEEK_API_KEY")
	if not api_key:
		print("DEEPSEEK_API_KEY 未设置，无法调用模型。", file=sys.stderr)
		return 1

	client = OpenAI(base_url=base_url, api_key=api_key)
	agent = TravelAgent(client, model)
	result = agent.run(agent_input)
	print(result)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
