import os
import sys
import traceback
from typing import Any

import runpod
from runpod import RunPodLogger

from engine import LlamaCPPEngine, LlamaCPPOpenAIEngine
from utils import JobInput

DEFAULT_MAX_CONCURRENCY = 1
HANDLER_MODE_ONE_SHOT = "one-shot"
HANDLER_MODE_STREAM = "stream"

max_concurrency = int(os.getenv("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))
handler_mode = (
    os.getenv("RUNPOD_HANDLER_MODE", HANDLER_MODE_ONE_SHOT)
    .strip()
    .lower()
)
log = RunPodLogger()
llama_engine = LlamaCPPEngine()
openai_engine = LlamaCPPOpenAIEngine()


def select_engine(job_input: JobInput):
    return openai_engine if job_input.openai_route else llama_engine


async def generate(job: Any):
    job_input = JobInput(job.get("input") or {})
    engine = select_engine(job_input)

    async for batch in engine.generate(job_input):
        yield batch


async def one_shot_handler(job: Any):
    try:
        output = [batch async for batch in generate(job)]
        if len(output) == 1:
            return output[0]
        return output
    except Exception as exc:
        return handle_error(exc)


async def stream_handler(job: Any):
    try:
        async for batch in generate(job):
            yield batch
    except Exception as exc:
        yield handle_error(exc)


def handle_error(exc: Exception):
    message = str(exc)
    log.error(f"Error during inference: {message}")
    log.error(traceback.format_exc())
    if "cuda" in message.lower():
        sys.exit(1)
    return {"error": message}


def serverless_config():
    if handler_mode == HANDLER_MODE_STREAM:
        return {
            "handler": stream_handler,
            "concurrency_modifier": lambda _x: max_concurrency,
            "return_aggregate_stream": True,
        }

    if handler_mode != HANDLER_MODE_ONE_SHOT:
        raise RuntimeError(
            "RUNPOD_HANDLER_MODE must be "
            f"'{HANDLER_MODE_ONE_SHOT}' or '{HANDLER_MODE_STREAM}'."
        )

    return {
        "handler": one_shot_handler,
        "concurrency_modifier": lambda _x: max_concurrency,
    }


runpod.serverless.start(serverless_config())
