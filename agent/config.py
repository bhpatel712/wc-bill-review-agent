"""
config.py  (WK4.3 support)

Loads Azure OpenAI settings from a local .env file (never committed, never
pasted into chat -- see .env.example for the template) and builds the
LangChain chat model used by explain.py.

Embeddings (WK4.1/WK4.2) deliberately do NOT go through Azure -- they use a
free, local model (FastEmbed, see build_index.py) so indexing the ~60 rule
documents costs nothing and needs no credentials at all. Only the final
generation step (WK4.3, turning a retrieved rule into a plain-language
explanation) needs Azure OpenAI, since that's the one step that actually
benefits from a strong hosted model instead of something running locally.
"""
import os

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

REQUIRED_VARS = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
]

# Microsoft Foundry resources created from mid-2026 onward hand out a
# "v1" endpoint (.../openai/v1/) instead of the older per-api-version
# endpoint. The v1 surface is plain OpenAI-API-compatible -- no
# api-version query param, no monthly version churn -- so this project
# uses langchain_openai's plain ChatOpenAI (base_url + api_key) against
# it rather than AzureChatOpenAI, which is built for the older
# azure_endpoint + api_version pattern. See agent/.env.example.
V1_PATH_HINT = "/openai/v1"


def get_missing_vars():
    return [v for v in REQUIRED_VARS if not os.environ.get(v)]


def get_chat_model(temperature=0.2):
    missing = get_missing_vars()
    if missing:
        raise RuntimeError(
            "Missing required Azure OpenAI settings: " + ", ".join(missing) + "\n"
            "Copy agent/.env.example to agent/.env and fill in your values from "
            "your Microsoft Foundry project's Deployments page (select your chat "
            "deployment -- endpoint, key, and deployment name are shown there)."
        )
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    if V1_PATH_HINT not in endpoint:
        raise RuntimeError(
            f"AZURE_OPENAI_ENDPOINT ({endpoint}) doesn't look like a v1 endpoint "
            f"-- expected it to contain '{V1_PATH_HINT}'. If you copied the "
            "'Project endpoint' from the Foundry portal instead of the resource's "
            "Azure OpenAI endpoint, or copied the classic (no /openai/v1) endpoint, "
            "this will 404. See agent/.env.example."
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        base_url=endpoint,
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        temperature=temperature,
    )


if __name__ == "__main__":
    missing = get_missing_vars()
    if missing:
        print(f"NOT configured yet -- missing: {', '.join(missing)}")
        print("Copy agent/.env.example to agent/.env and fill in your Azure OpenAI values.")
    else:
        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        print("Azure OpenAI settings found:")
        print(f"  endpoint:        {endpoint}")
        if V1_PATH_HINT not in endpoint:
            print(f"  WARNING: endpoint doesn't contain '{V1_PATH_HINT}' -- see agent/.env.example")
        print(f"  chat deployment: {os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT']}")
        print("  api key:         (set, not printed)")
