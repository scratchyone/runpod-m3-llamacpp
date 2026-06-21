import os
from functools import lru_cache

from utils import JobInput

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv():
        return False


@lru_cache(maxsize=1)
def get_client():
    from openai import OpenAI

    return OpenAI(
        base_url=os.getenv("LLAMA_OPENAI_BASE_URL", "http://localhost:3098/v1/"),
        api_key=os.getenv("LLAMA_OPENAI_API_KEY", "unused"),
    )


@lru_cache(maxsize=1)
def default_model_id():
    configured_model = os.getenv("LLAMA_DEFAULT_MODEL", "").strip()
    if configured_model:
        return configured_model
    return get_client().models.list().data[0].id


def ensure_model(openai_input):
    if "model" not in openai_input:
        openai_input["model"] = default_model_id()


def normalize_route(route):
    if not route:
        return route

    route = route.strip()
    if not route.startswith("/"):
        route = f"/{route}"

    for prefix in ("/openai/v1", "/v1"):
        if route.startswith(prefix):
            route = route[len(prefix):]
            break

    return route or "/models"


def response_to_dict(response):
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response


class LlamaCPPEngine:
    def __init__(self):
        load_dotenv()

    async def generate(self, job_input):
        openai_engine = LlamaCPPOpenAIEngine()
        openai_input = dict(job_input.openai_input)
        openai_input.setdefault("stream", job_input.stream)

        if "messages" in openai_input:
            route = "/v1/chat/completions"
        elif "prompt" in openai_input:
            route = "/v1/completions"
        elif isinstance(job_input.llm_input, str):
            openai_input["prompt"] = job_input.llm_input
            route = "/v1/completions"
        else:
            openai_input["messages"] = job_input.llm_input
            route = "/v1/chat/completions"

        ensure_model(openai_input)
        openai_job = JobInput(
            {
                "openai_route": route,
                "openai_input": openai_input,
            }
        )

        generate = openai_engine.generate(openai_job)
        async for batch in generate:
            yield batch


class LlamaCPPOpenAIEngine:
    def __init__(self):
        load_dotenv()

    async def generate(self, job_input):
        openai_input = dict(job_input.openai_input or {})
        route = normalize_route(job_input.openai_route)

        if route != "/models":
            ensure_model(openai_input)

        if route == "/models":
            async for response in self._handle_model_request():
                yield response
        elif route in ["/chat/completions", "/completions"]:
            async for response in self._handle_chat_or_completion_request(
                openai_input,
                chat=route == "/chat/completions",
            ):
                yield response
        else:
            yield {
                "error": (
                    "invalid route: expected /v1/models, /v1/chat/completions, "
                    "or /v1/completions"
                )
            }

    async def _handle_model_request(self):
        try:
            response = get_client().models.list()

            yield {
                "object": "list",
                "data": [response_to_dict(model) for model in response.data],
            }
        except Exception as e:
            yield {"error": str(e)}

    async def _handle_chat_or_completion_request(
        self, openai_input, chat=False
    ):
        try:
            client = get_client()
            if chat:
                response = client.chat.completions.create(**openai_input)
            else:
                response = client.completions.create(**openai_input)

            if not openai_input.get("stream", False):
                yield response_to_dict(response)
                return

            for chunk in response:
                yield response_to_dict(chunk)

            yield {"done": True}

        except Exception as e:
            yield {"error": str(e)}
