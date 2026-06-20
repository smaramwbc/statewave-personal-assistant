"""Demo developer profiles seeded into Statewave on first run.

Two personas — both are developers building real things with LLM APIs.
Their episode histories read like actual API exploration sessions, not support tickets.

dev_alice  — primary demo user; building a RAG-based code assistant
dev_bob    — secondary user; 6 sessions, used for token-budget demo
"""

from typing import Any

PROFILES: list[dict[str, Any]] = [
    {
        "user_id": "dev_alice",
        "display_name": "Alice Chen",
        "description": "Indie dev building a RAG-based code assistant, 4 sessions",
        "episodes": [
            {
                "user_message": (
                    "What chunk size should I use when embedding my codebase? "
                    "Files range from 50 to 2000 lines."
                ),
                "assistant_response": (
                    "For code, chunk by logical unit (function or class) rather than token count. "
                    "If you must split by tokens, 512 with 64-token overlap works well — "
                    "it keeps enough context for the embedding to be meaningful without diluting the signal.\n\n"
                    "```python\ndef chunk_code(text: str, size: int = 512, overlap: int = 64) -> list[str]:\n"
                    "    tokens = tokenizer.encode(text)\n"
                    "    chunks = []\n"
                    "    for i in range(0, len(tokens), size - overlap):\n"
                    "        chunks.append(tokenizer.decode(tokens[i : i + size]))\n"
                    "    return chunks\n```\n\n"
                    "Start at 512, benchmark retrieval precision, then tune."
                ),
            },
            {
                "user_message": (
                    "My batch embedding job keeps hitting 429s. I'm sending ~800 requests "
                    "per minute but the docs say there's no rate cap on my plan."
                ),
                "assistant_response": (
                    "There's an undocumented per-IP burst limit (~500 req/10s) introduced "
                    "in the last infra update — it's separate from the per-account monthly cap. "
                    "I've flagged it for the docs team.\n\n"
                    "Fix with exponential backoff:\n\n"
                    "```python\nimport asyncio\nfrom tenacity import retry, wait_exponential, stop_after_attempt\n\n"
                    "@retry(\n    wait=wait_exponential(min=1, max=60),\n    stop=stop_after_attempt(6),\n)\n"
                    "async def embed_with_retry(client, text: str):\n"
                    "    return await client.embeddings.create(input=text, model='text-embedding-3-small')\n```\n\n"
                    "Also consider batching: the API accepts up to 2048 strings per call."
                ),
            },
            {
                "user_message": "How do I stream completions so users see tokens as they arrive?",
                "assistant_response": (
                    "Use the async stream context manager:\n\n"
                    "```python\nasync with client.beta.chat.completions.stream(\n"
                    "    model='gpt-4o-mini',\n"
                    "    messages=[{'role': 'user', 'content': prompt}],\n"
                    ") as stream:\n"
                    "    async for event in stream:\n"
                    "        if event.type == 'content.delta':\n"
                    "            yield event.delta\n```\n\n"
                    "In FastAPI, wrap it in a `StreamingResponse` with `media_type='text/event-stream'`. "
                    "This cuts time-to-first-token dramatically for your users."
                ),
            },
            {
                "user_message": (
                    "My retrieval quality dropped after I switched to cosine similarity. "
                    "Was using dot product before."
                ),
                "assistant_response": (
                    "If your embeddings are L2-normalised (which text-embedding-3-* are), "
                    "cosine similarity and dot product are mathematically equivalent — "
                    "so the change itself shouldn't affect quality.\n\n"
                    "More likely culprits:\n"
                    "- Chunk size changed between indexing runs\n"
                    "- Index wasn't rebuilt after switching models\n"
                    "- Query isn't being embedded with the same model as the corpus\n\n"
                    "Check that `model` is identical in both your indexing and query calls."
                ),
            },
        ],
    },
    {
        "user_id": "dev_bob",
        "display_name": "Bob Martinez",
        "description": "Dev integrating LLM API into a team productivity tool, 6 sessions",
        "episodes": [
            {
                "user_message": "How do I pick the right model for a summarisation task? Cost matters.",
                "assistant_response": (
                    "For summarisation: gpt-4o-mini is the clear cost/quality winner right now. "
                    "It handles documents up to ~128k tokens and costs ~$0.15/1M input tokens. "
                    "Only step up to gpt-4o if you need nuanced reasoning over complex multi-doc inputs.\n\n"
                    "Rule of thumb: start with mini, only upgrade if evals fail."
                ),
            },
            {
                "user_message": "We're building a Slack bot. What's the latency like on streaming?",
                "assistant_response": (
                    "Time-to-first-token with streaming is typically 300–800ms for gpt-4o-mini. "
                    "For Slack, streaming is usually the wrong pattern — Slack's 3-second acknowledgement "
                    "window doesn't map well to SSE. Better approach:\n\n"
                    "1. Acknowledge the slash command immediately (return 200 in <3s)\n"
                    "2. Kick off the LLM call in the background\n"
                    "3. Use `response_url` to POST the result back when ready\n\n"
                    "This decouples Slack's timeout from your LLM latency entirely."
                ),
            },
            {
                "user_message": "Can I cache responses to cut costs?",
                "assistant_response": (
                    "Yes — prompt caching cuts input token cost by ~75% for repeated prefixes. "
                    "It activates automatically when a prompt prefix ≥1024 tokens is reused within 5 minutes.\n\n"
                    "For your Slack bot, put your system prompt and any static context first, "
                    "then append the user-specific part at the end. The static prefix will be cached.\n\n"
                    "```python\nmessages = [\n"
                    "    {'role': 'system', 'content': STATIC_SYSTEM_PROMPT},  # cached\n"
                    "    {'role': 'user', 'content': user_message},            # not cached\n"
                    "]\n```"
                ),
            },
            {
                "user_message": "We need to support function calling. How do I define a tool?",
                "assistant_response": (
                    "Define tools as a list of JSON Schema objects:\n\n"
                    "```python\ntools = [{\n"
                    "    'type': 'function',\n"
                    "    'function': {\n"
                    "        'name': 'search_docs',\n"
                    "        'description': 'Search the company knowledge base.',\n"
                    "        'parameters': {\n"
                    "            'type': 'object',\n"
                    "            'properties': {\n"
                    "                'query': {'type': 'string', 'description': 'Search query'},\n"
                    "            },\n"
                    "            'required': ['query'],\n"
                    "        },\n"
                    "    },\n"
                    "}]\n\n"
                    "response = await client.chat.completions.create(\n"
                    "    model='gpt-4o-mini', messages=messages, tools=tools\n"
                    ")\n```\n\n"
                    "Check `response.choices[0].finish_reason == 'tool_calls'` to know when to execute."
                ),
            },
            {
                "user_message": "How should I handle errors and retries in production?",
                "assistant_response": (
                    "Classify errors before retrying:\n\n"
                    "- `RateLimitError` (429) → retry with exponential backoff, jitter\n"
                    "- `APIConnectionError` / `Timeout` → retry up to 3x\n"
                    "- `APIStatusError` 5xx → retry once, then alert\n"
                    "- `AuthenticationError` (401/403) → do NOT retry, fail fast\n\n"
                    "The SDK has built-in retry logic you can configure:\n\n"
                    "```python\nclient = AsyncOpenAI(max_retries=3)  # default is 2\n```\n\n"
                    "Add your own layer with `tenacity` for custom backoff policies."
                ),
            },
            {
                "user_message": "What's the token limit for gpt-4o-mini and how do I count tokens?",
                "assistant_response": (
                    "gpt-4o-mini: 128k token context window, 16k max output tokens.\n\n"
                    "Count tokens before sending:\n\n"
                    "```python\nimport tiktoken\n\nenc = tiktoken.encoding_for_model('gpt-4o-mini')\n\n"
                    "def count_tokens(messages: list[dict]) -> int:\n"
                    "    total = 0\n"
                    "    for msg in messages:\n"
                    "        total += 4  # per-message overhead\n"
                    "        total += len(enc.encode(msg['content']))\n"
                    "    return total + 2  # reply priming\n```\n\n"
                    "Keep your prompt + expected reply under 120k to leave room for overhead."
                ),
            },
        ],
    },
]
