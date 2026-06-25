# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run the app
uvicorn app.main:app --reload

# Seed demo users into Statewave
python -m scripts.seed

# Run all tests
pytest -v

# Run a single test file
pytest tests/test_routes.py -v

# Run a single test
pytest tests/test_routes.py::test_chat_endpoint -v

# Lint
ruff check .

# Type check
mypy app scripts

# Demo CLI
python -m scripts.compare compare --user dev_alice --message "your message"
python -m scripts.compare inspect --user dev_alice
python -m scripts.compare budget --user dev_bob
python -m scripts.compare chat --user dev_alice
```

## Architecture

This is a FastAPI app that demonstrates **Statewave** — a memory indexing service that replaces naive chat history injection with ranked, token-bounded memory retrieval. The core insight: raw conversation history injected into prompts fills the context window within 8–10 turns and gives no ranking signal. Statewave extracts structured memories from episodes and returns only the highest-ranked facts within a token budget.

### Request flow

```
POST /api/v1/chat
  → StatewaveClient.get_context(user_id, message)   # ranked memory, token-bounded
  → LLMService.chat(message, assembled_context)      # system prompt + injected memory
  → StatewaveClient.record_episode(...)              # fire-and-forget, non-fatal
  → ChatResponse
```

### Key service contracts

- **`app/services/statewave.py`** — async httpx client wrapping four Statewave endpoints: `record_episode`, `compile_memories`, `get_context`, `list_memories`. The `get_context` call returns a `ContextBundle` with pre-assembled text and a token estimate, ready for direct injection into the LLM system prompt.
- **`app/services/llm.py`** — thin AsyncOpenAI wrapper. Builds the system prompt by appending the Statewave `assembled_context` string. No memory logic lives here.
- **`app/api/routes.py`** — three endpoints: `POST /api/v1/chat` (the main flow above), `GET /api/v1/memory/{user_id}` (inspect compiled memory state), and `POST /api/v1/compare` (side-by-side stateless vs memory-backed). All services are injected via FastAPI `Depends()`.

### Demo data

`app/data/profiles.py` contains two pre-seeded users (`dev_alice` Alice Chen and `dev_bob` Bob Martinez) with fabricated episode histories. `scripts/seed.py` ingests these into Statewave. `dev_alice` is the primary demo user (4 sessions), `dev_bob` is used for token budget demos (6 sessions).

### Configuration

All config lives in `app/core/config.py` (Pydantic Settings). Required: `LLM_API_KEY`. Optional but important: `STATEWAVE_BASE_URL` (default `http://localhost:8100`), `STATEWAVE_MAX_TOKENS` (default `800`). Copy `.env.example` to `.env`.

### Testing approach

Tests mock both Statewave and OpenAI — no real HTTP calls. `tests/conftest.py` provides fixtures. `pytest-httpx` intercepts httpx calls in `test_statewave_client.py`; FastAPI's `TestClient` + `unittest.mock.patch` handle route tests.

### Code style

- Full mypy strict mode; all code must be typed.
- Ruff with 100-char line length; rules: E, F, I, UP, B, SIM.
- Async throughout — all service methods and route handlers are `async def`.
