import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from find_cached import CACHE_DIR, find_model_path

LLAMA_SERVER = "/app/llama-server"
LLAMA_PORT = "3098"
LLAMA_HOST = "0.0.0.0"
LOG_PATH = "llama.server.log"
DEFAULT_ARGS = "-hf unsloth/gemma-3-270m-it-GGUF:IQ2_XXS --ctx-size 512 -ngl 999"
MODEL_FLAGS = {"-m", "--model", "-hf", "--hf-repo", "-hfr"}
MMPROJ_FLAGS = {"--mmproj", "-mm", "--mmproj-url", "-mmu"}
HOST_FLAGS = {"--host"}


def env(name, default=""):
    return os.getenv(name, default).strip()


def contains_flag(args, flags):
    return any(
        arg == flag or arg.startswith(f"{flag}=")
        for arg in args
        for flag in flags
    )


def resolve_cached_file(model, path, cache_dir, label):
    resolved = find_model_path(model, path, cache_dir)
    if resolved is None:
        raise RuntimeError(
            f"{label} not found in RunPod cache. "
            f"model='{model}', path='{path}', cache_dir='{cache_dir}'"
        )
    return resolved


def build_llama_args():
    raw_args = env("LLAMA_SERVER_CMD_ARGS")
    if not raw_args:
        print(
            "launcher.py: Warning: LLAMA_SERVER_CMD_ARGS is not set. "
            f"Defaulting to {DEFAULT_ARGS}"
        )
        raw_args = DEFAULT_ARGS

    args = shlex.split(raw_args)
    if contains_flag(args, {"--port"}):
        raise RuntimeError("LLAMA_SERVER_CMD_ARGS must not define --port; port 3098 is managed by the worker.")
    if not contains_flag(args, HOST_FLAGS):
        args.extend(["--host", env("LLAMA_SERVER_HOST", LLAMA_HOST)])

    cache_dir = env("LLAMA_CACHE_DIR", CACHE_DIR)
    cached_model = env("LLAMA_CACHED_MODEL")
    cached_gguf_path = env("LLAMA_CACHED_GGUF_PATH")
    argv = []

    if cached_model:
        if not cached_gguf_path:
            raise RuntimeError("LLAMA_CACHED_GGUF_PATH is required when LLAMA_CACHED_MODEL is set.")
        if contains_flag(args, MODEL_FLAGS):
            raise RuntimeError("Do not define -m/-hf model flags in LLAMA_SERVER_CMD_ARGS when cached model mode is enabled.")
        model_path = resolve_cached_file(cached_model, cached_gguf_path, cache_dir, "cached model")
        argv.extend(["-m", model_path])
    else:
        print("launcher.py: Launcher cached-model mode off; using llama.cpp's own download cache (LLAMA_CACHE) on the volume. First start downloads; later starts reuse it.")

    mmproj_sources = [
        bool(env("LLAMA_CACHED_MMPROJ_PATH")),
        bool(env("LLAMA_MMPROJ_PATH")),
        bool(env("LLAMA_MMPROJ_URL")),
    ]
    if sum(mmproj_sources) > 1:
        raise RuntimeError("Set only one of LLAMA_CACHED_MMPROJ_PATH, LLAMA_MMPROJ_PATH, or LLAMA_MMPROJ_URL.")
    if any(mmproj_sources) and contains_flag(args, MMPROJ_FLAGS):
        raise RuntimeError("Do not define --mmproj/--mmproj-url in LLAMA_SERVER_CMD_ARGS when mmproj env vars are set.")

    cached_mmproj_path = env("LLAMA_CACHED_MMPROJ_PATH")
    if cached_mmproj_path:
        if not cached_model:
            raise RuntimeError("LLAMA_CACHED_MODEL is required when LLAMA_CACHED_MMPROJ_PATH is set.")
        mmproj_path = resolve_cached_file(cached_model, cached_mmproj_path, cache_dir, "cached mmproj")
        argv.extend(["--mmproj", mmproj_path])

    local_mmproj_path = env("LLAMA_MMPROJ_PATH")
    if local_mmproj_path:
        if not os.path.isfile(local_mmproj_path):
            raise RuntimeError(f"LLAMA_MMPROJ_PATH does not exist: {local_mmproj_path}")
        argv.extend(["--mmproj", local_mmproj_path])

    mmproj_url = env("LLAMA_MMPROJ_URL")
    if mmproj_url:
        argv.extend(["--mmproj-url", mmproj_url])

    argv.extend(args)
    argv.extend(["--port", LLAMA_PORT])
    return argv


def stop_existing_llama_server():
    subprocess.run(["pkill", "llama-server"], stderr=subprocess.DEVNULL, check=False)


def start_handler(handler_args):
    return subprocess.call([sys.executable, "-u", "handler.py", *handler_args])


def wait_for_ready_http(process, port, timeout_seconds):
    """Poll llama.cpp's /health until ready. llama-server writes straight to the
    container stdout/stderr, so its logs (including the download progress bar)
    show up natively in the RunPod console instead of being scraped here."""
    url = f"http://127.0.0.1:{port}/health"
    start = time.monotonic()
    deadline = start + timeout_seconds
    last_note = 0.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {process.returncode}.")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            # connection refused (still downloading / port not bound) or
            # HTTP 503 "loading model" -> keep waiting.
            pass
        now = time.monotonic()
        if now - last_note >= 20:
            print(
                f"launcher.py: waiting for llama-server "
                f"(downloading/loading model, ~{int((now - start) // 60)} min elapsed)...",
                flush=True,
            )
            last_note = now
        time.sleep(3)
    raise RuntimeError(f"llama-server did not become ready within {timeout_seconds} seconds.")


def main():
    try:
        llama_args = build_llama_args()
        timeout_seconds = int(env("LLAMA_STARTUP_TIMEOUT_SECONDS", "120"))
    except Exception as exc:
        print(f"launcher.py: Error: {exc}", file=sys.stderr)
        return 1

    print("launcher.py: Stopping existing llama-server instances (if any)...")
    stop_existing_llama_server()
    command = [LLAMA_SERVER, *llama_args]
    print("launcher.py: Running " + shlex.join(command))

    env_vars = os.environ.copy()
    env_vars["LD_LIBRARY_PATH"] = "/app"
    # Inherit stdout/stderr so llama-server output (including the \r download
    # progress bar) streams natively to the container log / RunPod console.
    # Readiness is detected via the /health endpoint, not by scraping logs.
    process = subprocess.Popen(command, env=env_vars)

    def cleanup(signum, _frame):
        print("launcher.py: Cleaning up...")
        process.terminate()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        wait_for_ready_http(process, LLAMA_PORT, timeout_seconds)
        print("launcher.py: llama-server is ready, delegating to the handler script.")
        return start_handler(sys.argv[1:])
    except Exception as exc:
        print(f"launcher.py: Error: {exc}", file=sys.stderr)
        return 1
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
