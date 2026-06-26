"""LiteLLM chat-model factory (in-process model gateway)."""

from __future__ import annotations

from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLMRouter
from litellm import Router

from financial_qa.app.infrastructure.settings import REPO_ROOT, get_settings
from financial_qa.app.integrations.llm.model_list_loader import load_model_list

_NUM_RETRIES = 3
_RETRY_AFTER = 5
_ROUTING_STRATEGY = "latency-based-routing"


def create_chat_model(*, temperature: float = 0.0, streaming: bool = True) -> ChatLiteLLMRouter:
    """Build a ChatLiteLLMRouter for the configured model.

    Provider/model are driven by ``assets/model_list.yaml`` + env (see ``.env``); switching provider
    means editing those, not this code.
    """
    settings = get_settings()
    # Populate os.environ for ${VAR} expansion in the model list (no-op when already set, e.g. docker).
    load_dotenv(REPO_ROOT / ".env")

    router = Router(
        model_list=load_model_list(),
        num_retries=_NUM_RETRIES,
        retry_after=_RETRY_AFTER,
        routing_strategy=_ROUTING_STRATEGY,
    )
    return ChatLiteLLMRouter(
        router=router,
        model=settings.model_name,
        temperature=temperature,
        streaming=streaming,
    )
