# LM Studio Setup

This project uses [LM Studio](https://lmstudio.ai/) to run the generation
LLM entirely locally — no OpenAI, Gemini, or Anthropic API key is used or
required anywhere in this project.

## How the connection works

LM Studio can expose a local server that speaks the same protocol as
OpenAI's `/v1/chat/completions` REST API. This project's `src/llm.py`
uses the `openai` Python package purely as an HTTP client, pointed at
that local server instead of `api.openai.com`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",   # LM Studio's local server
    api_key="lm-studio",                    # placeholder only — never checked, never sent externally
)
```

The `api_key` value is required by the SDK's constructor but is not a
real credential — LM Studio's local server does not validate it, and
nothing is transmitted outside your machine.

## One-time setup

1. Install LM Studio and open it.
2. Download and load a chat-capable local model (any reasonably capable
   instruction-tuned model works to start; heavier models will be slower
   but may follow the grounding instructions in `src/prompt.py` more
   reliably — this is worth revisiting once the pipeline is working).
3. Go to the **Local Server** tab in LM Studio.
4. Note the exact model name shown there.
5. Click **Start Server**. By default it listens on
   `http://localhost:1234/v1`.

## Configure this project

In `.env`:

```
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=<paste the exact model name from LM Studio here>
```

The model name is intentionally not hardcoded anywhere in the codebase,
because it changes whenever you switch models in LM Studio.

## Verify the connection

```bash
python scripts/test_lm_studio.py
```

This script connects to LM Studio, lists available models, sends "Say
hello in one sentence.", and prints the reply. If LM Studio's server
isn't running, it prints:

```
LM Studio server is not available. Start the local server and try again.
```

instead of crashing or silently falling back to any external API. The
rest of the application (`src/llm.py`) follows the same fail-clearly
principle.
