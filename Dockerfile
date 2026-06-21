# syntax=docker/dockerfile:1
#
# RunPod serverless worker: llama.cpp (MiniMax-M3) + OpenAI-compatible handler.
# llama.cpp is compiled from Daniel Han's preliminary MiniMax-M3 PR branch,
# since M3 is not yet in any released llama.cpp build.
#   PR: https://github.com/ggml-org/llama.cpp/pull/24523
# Handler: vendored from ViniciosLugli/runpod-serverless (audited).

######################## Builder: compile llama-server (CUDA) ########################
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

ARG LLAMA_REPO=https://github.com/danielhanchen/llama.cpp.git
ARG LLAMA_BRANCH=minimax-m3
# H100 = compute capability 9.0; single arch keeps the CUDA compile fast.
ARG CUDA_ARCH=90

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential ca-certificates libcurl4-openssl-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch ${LLAMA_BRANCH} ${LLAMA_REPO} .

# Static build -> llama-server is self-contained except for system CUDA libs
# (cuBLAS/cudart), which are provided by the runtime image below.
RUN cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DGGML_NATIVE=OFF \
      -DLLAMA_CURL=ON \
  && cmake --build build --config Release -j 2 --target llama-server \
  && ls -la build/bin/

######################## Runtime: CUDA runtime + python handler ########################
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip libgomp1 libcurl4 curl bash ca-certificates \
  && ln -sf /usr/bin/python3 /usr/bin/python \
  && rm -rf /var/lib/apt/lists/*

# Self-contained llama-server binary from the builder stage.
COPY --from=builder /src/build/bin/llama-server /app/llama-server

WORKDIR /work
COPY src/ /work/
RUN pip3 install --no-cache-dir -r /work/requirements.txt \
  && chmod +x /work/start.sh /app/llama-server

ENTRYPOINT ["/work/start.sh"]
