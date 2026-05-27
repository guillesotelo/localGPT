import os
import httpx
from langchain_openai import AzureChatOpenAI
from constants import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    AZURE_MAX_TOKENS,
    AZURE_TEMPERATURE,
)

def load_azure_model() -> AzureChatOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set")

    # Build clients at call time so load_dotenv() has already run.
    # SSL verification is disabled to match the rest of the app.
    # httpx does not share Python's ssl._create_unverified_context override,
    # so we must disable verify here explicitly.
    http_client = httpx.Client(
        verify=False,
        timeout=httpx.Timeout(120.0, connect=15.0),
    )
    http_async_client = httpx.AsyncClient(
        verify=False,
        timeout=httpx.Timeout(120.0, connect=15.0),
    )

    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
        api_key=api_key,
        max_tokens=AZURE_MAX_TOKENS,
        temperature=AZURE_TEMPERATURE,
        streaming=True,
        http_client=http_client,
        http_async_client=http_async_client,
    )
