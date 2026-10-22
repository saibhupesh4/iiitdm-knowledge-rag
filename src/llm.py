"""
llm.py

Handles communication with the local LLM running through LM Studio.

Architecture:

    prompt.py
        ↓
    llm.generate()
        ↓
    LM Studio Local Server
        ↓
    Local LLM

The OpenAI Python SDK is used only as a client for LM Studio's
OpenAI-compatible local API.

No external LLM API is used.
"""

from __future__ import annotations

from typing import Any

try:
    from openai import (
        OpenAI,
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
    )
except ImportError:
    OpenAI = None
    APIConnectionError = Exception
    APIStatusError = Exception
    APITimeoutError = Exception


# =============================================================
# Project configuration
# =============================================================

from src.config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_MODEL,
    LM_STUDIO_PLACEHOLDER_KEY,
)


# =============================================================
# Cached client
# =============================================================

_client: Any | None = None


# =============================================================
# Custom exceptions
# =============================================================

class LMStudioUnavailableError(RuntimeError):
    """Raised when LM Studio cannot be reached."""


class LMStudioConfigurationError(RuntimeError):
    """Raised when LM Studio configuration is invalid."""


class LMStudioGenerationError(RuntimeError):
    """Raised when LM Studio generation fails."""


# =============================================================
# Configuration validation
# =============================================================

def _validate_configuration() -> None:
    """Validate LM Studio configuration."""

    if not LM_STUDIO_BASE_URL:
        raise LMStudioConfigurationError(
            "LM_STUDIO_BASE_URL is not configured."
        )

    if not LM_STUDIO_MODEL:
        raise LMStudioConfigurationError(
            "LM_STUDIO_MODEL is not configured."
        )

    if LM_STUDIO_MODEL.strip().upper() == "YOUR_MODEL_NAME":
        raise LMStudioConfigurationError(
            "LM_STUDIO_MODEL is still set to 'YOUR_MODEL_NAME'. "
            "Replace it with your actual LM Studio model identifier."
        )


# =============================================================
# Client creation
# =============================================================

def get_client() -> "OpenAI":
    """
    Return a cached OpenAI-compatible client connected to LM Studio.
    """

    global _client

    if _client is not None:
        return _client

    if OpenAI is None:
        raise ImportError(
            "The 'openai' package is not installed.\n"
            "Install it using:\n"
            "    pip install openai"
        )

    _validate_configuration()

    try:
        _client = OpenAI(
            base_url=LM_STUDIO_BASE_URL.rstrip("/"),
            api_key=LM_STUDIO_PLACEHOLDER_KEY,
            timeout=300.0,
        )

    except Exception as exc:
        raise LMStudioConfigurationError(
            "Failed to initialize the LM Studio client.\n"
            f"Base URL: {LM_STUDIO_BASE_URL}\n"
            f"Reason: {exc}"
        ) from exc

    return _client


# =============================================================
# Response extraction
# =============================================================

def _extract_text(message: Any) -> str:
    """
    Extract the best available text from an LM Studio response.

    Normal models usually return:

        message.content

    Only user-facing message content is returned. Internal reasoning is
    never treated as an answer.
    """

    if message is None:
        return ""

    # ---------------------------------------------------------
    # First priority: normal visible content
    # ---------------------------------------------------------

    content = getattr(message, "content", None)

    if content is not None:

        if isinstance(content, str):
            text = content.strip()

            if text:
                return text

        # Some API implementations may return content as a list
        if isinstance(content, list):

            parts = []

            for item in content:

                if isinstance(item, str):
                    parts.append(item)

                elif isinstance(item, dict):
                    text = item.get("text")

                    if text:
                        parts.append(str(text))

                else:
                    text = getattr(item, "text", None)

                    if text:
                        parts.append(str(text))

            text = "".join(parts).strip()

            if text:
                return text

    return ""


# =============================================================
# Model generation
# =============================================================

def generate(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2000,
) -> str:
    """
    Generate an answer using the local LM Studio model.

    Parameters
    ----------
    system_prompt : str
        System instructions.

    user_prompt : str
        Question and retrieved evidence.

    temperature : float
        Sampling temperature.

    max_tokens : int
        Maximum number of generated tokens.

    Returns
    -------
    str
        Generated answer.
    """

    # ---------------------------------------------------------
    # Validate inputs
    # ---------------------------------------------------------

    if not isinstance(system_prompt, str):
        raise ValueError(
            "system_prompt must be a string."
        )

    if not isinstance(user_prompt, str):
        raise ValueError(
            "user_prompt must be a string."
        )

    if not system_prompt.strip():
        raise ValueError(
            "system_prompt cannot be empty."
        )

    if not user_prompt.strip():
        raise ValueError(
            "user_prompt cannot be empty."
        )

    if not isinstance(temperature, (int, float)):
        raise ValueError(
            "temperature must be numeric."
        )

    if temperature < 0:
        raise ValueError(
            "temperature cannot be negative."
        )

    if not isinstance(max_tokens, int):
        raise ValueError(
            "max_tokens must be an integer."
        )

    if max_tokens <= 0:
        raise ValueError(
            "max_tokens must be greater than 0."
        )

    # ---------------------------------------------------------
    # Get client
    # ---------------------------------------------------------

    client = get_client()

    messages = [
        {
            "role": "system",
            "content": system_prompt.strip(),
        },
        {
            "role": "user",
            "content": user_prompt.strip(),
        },
    ]

    # ---------------------------------------------------------
    # Generate response
    # ---------------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=LM_STUDIO_MODEL,
            messages=messages,
            temperature=float(temperature),
            max_tokens=max_tokens,
        )

    except APIConnectionError as exc:

        raise LMStudioUnavailableError(
            "LM Studio server is not available.\n"
            "Make sure LM Studio is running and the local server "
            "is started."
        ) from exc

    except APITimeoutError as exc:

        raise LMStudioUnavailableError(
            "LM Studio did not respond within the timeout period."
        ) from exc

    except APIStatusError as exc:

        raise LMStudioGenerationError(
            "LM Studio returned an API error.\n"
            f"Status code: {getattr(exc, 'status_code', 'unknown')}\n"
            f"Model: {LM_STUDIO_MODEL}"
        ) from exc

    except Exception as exc:

        raise LMStudioGenerationError(
            "Unexpected error while generating a response.\n"
            f"Reason: {exc}"
        ) from exc

    # ---------------------------------------------------------
    # Validate response
    # ---------------------------------------------------------

    if response is None:

        raise LMStudioGenerationError(
            "LM Studio returned no response."
        )

    choices = getattr(response, "choices", None)

    if not choices:

        raise LMStudioGenerationError(
            "LM Studio returned no completion choices."
        )

    choice = choices[0]

    if choice is None:

        raise LMStudioGenerationError(
            "LM Studio returned an empty completion choice."
        )

    message = getattr(choice, "message", None)

    if message is None:

        raise LMStudioGenerationError(
            "LM Studio returned a completion without a message."
        )

    # ---------------------------------------------------------
    # Extract answer
    # ---------------------------------------------------------

    answer = _extract_text(message)

    if not answer:

        # Useful diagnostic information
        content = getattr(message, "content", None)
        raise LMStudioGenerationError(
            "LM Studio generated no usable text.\n"
            f"message.content: {content!r}"
        )

    return answer


# =============================================================
# Testing
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("IIITDM Knowledge RAG - LM Studio Test")
    print("=" * 60)

    print()
    print(f"Base URL: {LM_STUDIO_BASE_URL}")
    print(f"Model:    {LM_STUDIO_MODEL}")

    print()
    print("[test] Sending test request...")

    try:

        answer = generate(
            system_prompt=(
                "You are a helpful assistant. "
                "Answer briefly and directly."
            ),
            user_prompt=(
                "Say hello in one short sentence."
            ),
            temperature=0.1,
            max_tokens=2000,
        )

        print()
        print("[SUCCESS]")
        print("-" * 60)
        print(answer)
        print("-" * 60)

    except LMStudioUnavailableError as exc:

        print()
        print("[LM STUDIO UNAVAILABLE]")
        print(exc)

    except LMStudioConfigurationError as exc:

        print()
        print("[CONFIGURATION ERROR]")
        print(exc)

    except LMStudioGenerationError as exc:

        print()
        print("[GENERATION ERROR]")
        print(exc)

    except Exception as exc:

        print()
        print("[UNEXPECTED ERROR]")
        print(exc)

    print()
    print("=" * 60)
