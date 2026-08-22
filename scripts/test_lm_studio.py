"""
scripts/test_lm_studio.py

Smallest possible check that LM Studio's local server is running and
reachable, before anything else in the project depends on it.

What this script does:
    1. Connects to LM Studio's OpenAI-compatible local endpoint.
    2. Lists the models currently available on the server.
    3. Selects the model configured via LM_STUDIO_MODEL (.env).
    4. Sends one message: "Say hello in one sentence."
    5. Prints the response.
    6. Fails with a clear, human-readable message (not a stack trace) if
       LM Studio is not running or the configured model isn't loaded.

Run with:
    python scripts/test_lm_studio.py

Before running:
    1. Open LM Studio.
    2. Load a chat model.
    3. Go to the "Local Server" tab and click "Start Server".
    4. Copy the exact model name shown there into LM_STUDIO_MODEL in .env.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import LM_STUDIO_BASE_URL, LM_STUDIO_MODEL, LM_STUDIO_PLACEHOLDER_KEY

try:
    from openai import OpenAI, APIConnectionError
except ImportError:
    print("The 'openai' package is not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def list_available_models(client: "OpenAI") -> list[str]:
    """
    Query LM Studio for the models it currently has available.
    """
    models = client.models.list()
    return [m.id for m in models.data]


def main() -> None:
    if not LM_STUDIO_MODEL:
        print(
            "LM_STUDIO_MODEL is not set.\n"
            "Open LM Studio, check the model name in the 'Local Server' tab, "
            "and set LM_STUDIO_MODEL in your .env file to match it exactly."
        )
        return

    client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_PLACEHOLDER_KEY)

    print(f"Connecting to LM Studio at {LM_STUDIO_BASE_URL} ...")

    try:
        available_models = list_available_models(client)
        print(f"Available models: {available_models}")
        if LM_STUDIO_MODEL not in available_models:
            print(
                f"WARNING: '{LM_STUDIO_MODEL}' is not in the list of models LM Studio "
                "currently reports as loaded. The request below may still fail."
            )
    except APIConnectionError:
        print("LM Studio server is not available. Start the local server and try again.")
        return
    except Exception as e:
        print(f"Could not list models (continuing anyway): {e}")

    print(f"Requesting model: {LM_STUDIO_MODEL}")

    try:
        response = client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
        )
    except APIConnectionError:
        print("LM Studio server is not available. Start the local server and try again.")
        return
    except Exception as e:
        # Covers cases like: server running but the named model isn't loaded.
        print(f"LM Studio responded with an error: {e}")
        return

    reply = response.choices[0].message.content
    print("\nLM Studio replied:")
    print(reply)


if __name__ == "__main__":
    main()
