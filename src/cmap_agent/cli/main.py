from __future__ import annotations

import argparse
import json
from cmap_agent.storage.sqlserver import SQLServerStore
from cmap_agent.tools.default_registry import build_default_registry
from cmap_agent.agent.context import build_system_prompt
from cmap_agent.agent.runner import execute_plan
from cmap_agent.llm.openai_client import OpenAIClient
from cmap_agent.llm.anthropic_client import AnthropicClient

def main():
    p = argparse.ArgumentParser(description="CMAP Agent CLI (dev)")
    p.add_argument("--user-id", type=int, required=True)
    p.add_argument("--thread-id", type=str, default=None)
    p.add_argument("--provider", type=str, choices=["openai","anthropic"], default="openai")
    p.add_argument("--model", type=str, default="gpt-4.1-mini")
    p.add_argument("message", type=str)
    args = p.parse_args()

    store = SQLServerStore.from_env()
    reg = build_default_registry()
    sys_prompt = build_system_prompt(reg)

    thread_id = args.thread_id or store.create_thread(user_id=args.user_id, client_tag="cli")
    store.add_message(thread_id=thread_id, user_id=args.user_id, role="user", content=args.message)

    history = store.get_recent_messages(thread_id, limit=20)
    conversation = []
    for m in history:
        role = (m.get("Role") or "user").lower()
        if role not in ("system","user","assistant"):
            role = "assistant"
        conversation.append({"role": role, "content": m.get("Content") or ""})

    cmap_key = store.load_cmap_api_key(args.user_id)
    if not cmap_key:
        raise SystemExit("No CMAP API key available. Set CMAP_API_KEY_FALLBACK or implement SQL lookup.")

    llm = OpenAIClient(args.model) if args.provider=="openai" else AnthropicClient(args.model)

    final, tool_trace = execute_plan(
        llm=llm,
        registry=reg,
        system_prompt=sys_prompt,
        conversation=conversation,
        user_message=args.message,
        ctx={"thread_id": thread_id, "user_id": args.user_id, "cmap_api_key": cmap_key},
    )

    store.add_message(thread_id=thread_id, user_id=args.user_id, role="assistant", content=final.assistant_message)
    print("\nASSISTANT:\n", final.assistant_message)
    if final.code:
        print("\nCODE:\n", final.code)
    if final.artifacts:
        print("\nARTIFACTS:\n", json.dumps(final.artifacts, indent=2))
    if tool_trace:
        print("\nTOOL TRACE:\n", json.dumps(tool_trace, indent=2))

if __name__ == "__main__":
    main()
