# syntax=docker/dockerfile:1
#
# RunPod serverless worker: llama.cpp (MiniMax-M3) + OpenAI-compatible handler.
# llama.cpp is compiled from Daniel Han's preliminary MiniMax-M3 PR branch,
# since M3 is not yet in any released llama.cpp build.
#   PR: https://github.com/ggml-org/llama.cpp/pull/24523
# Handler: vendored from ViniciosLugli/runpod-serverless (audited).

######################## Builder: compile llama-server (CUDA) ########################
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

ARG LLAMA_REPO=https://github.com/antirez/llama.cpp-deepseek-v4-flash.git
ARG LLAMA_BRANCH=main
# H100 = compute capability 9.0; single arch keeps the CUDA compile fast.
ARG CUDA_ARCH=90

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential ca-certificates \
      libcurl4-openssl-dev libssl-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch ${LLAMA_BRANCH} ${LLAMA_REPO} .

# Static build -> llama-server is self-contained except for system CUDA libs
# (cuBLAS/cudart), which are provided by the runtime image below.
# GGML_SCHED_MAX_SPLIT_INPUTS: fixed-size array bound for distinct graph-input
# tensors copied across GPU split boundaries when pipeline parallelism is active
# (n_copies > 1, i.e. multi-GPU --split-mode layer). Default 30 (guarded by
# #ifndef in ggml/src/ggml-backend.cpp) overflows for DeepSeek-V4-Flash's MoE
# graph across 5 GPUs -> GGML_ASSERT(n_graph_inputs < ...) SIGABRT at load.
# Bump it via a compile define; arrays are just pointers so the cost is trivial.
ARG GGML_MAX_SPLIT_INPUTS=128
RUN cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DGGML_NATIVE=OFF \
      -DLLAMA_CURL=ON \
      -DLLAMA_OPENSSL=ON \
      -DCMAKE_C_FLAGS="-DGGML_SCHED_MAX_SPLIT_INPUTS=${GGML_MAX_SPLIT_INPUTS}" \
      -DCMAKE_CXX_FLAGS="-DGGML_SCHED_MAX_SPLIT_INPUTS=${GGML_MAX_SPLIT_INPUTS}" \
  && cmake --build build --config Release -j 2 --target llama-server \
  && ls -la build/bin/

######################## Runtime: CUDA runtime + python handler ########################
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip libgomp1 libcurl4 libssl3 openssl curl bash ca-certificates \
  && ln -sf /usr/bin/python3 /usr/bin/python \
  && rm -rf /var/lib/apt/lists/*

# Self-contained llama-server binary from the builder stage.
COPY --from=builder /src/build/bin/llama-server /app/llama-server

WORKDIR /work
COPY src/ /work/
RUN pip3 install --no-cache-dir -r /work/requirements.txt \
  && chmod +x /work/start.sh /app/llama-server

ENTRYPOINT ["/work/start.sh"]
