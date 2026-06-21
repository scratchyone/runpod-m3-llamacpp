# runpod-m3-llamacpp

RunPod **serverless** worker that serves **MiniMax-M3 (abliterated)** GGUF models via
[llama.cpp](https://github.com/ggml-org/llama.cpp) with an **OpenAI-compatible** API.

MiniMax-M3 is not yet supported in any released llama.cpp build, so the image
compiles `llama-server` from Daniel Han's preliminary M3 PR branch
([ggml-org/llama.cpp#24523](https://github.com/ggml-org/llama.cpp/pull/24523),
`danielhanchen/llama.cpp@minimax-m3`). The OpenAI/RunPod handler is vendored from
[ViniciosLugli/runpod-serverless](https://github.com/ViniciosLugli/runpod-serverless).

## Image

GitHub Actions builds and pushes on every push to `main`:

```
ghcr.io/scratchyone/runpod-m3-llamacpp:latest
```

## Usage (RunPod serverless template env)

- `LLAMA_SERVER_CMD_ARGS` — e.g.
  `-m /runpod-volume/models/Minimax-M3-abliterated-clean-Q3_K_S.gguf -ngl 999 --split-mode layer --ctx-size 16384 -fa`
- `RUNPOD_HANDLER_MODE=stream`
- `LLAMA_STARTUP_TIMEOUT_SECONDS=1800` (large model load from network volume)
- `MAX_CONCURRENCY=1`

Endpoint: attach a network volume holding the GGUF, set `gpuCount` (4× H100 for
the Q3_K_S quant, ~184 GB), call `https://api.runpod.ai/v2/<ID>/openai/v1`.
