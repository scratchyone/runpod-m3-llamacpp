import argparse
import os
import sys

CACHE_DIR = "/runpod-volume/huggingface-cache/hub"


def find_model_path(model_name, file_in_repo="model.gguf", cache_dir=CACHE_DIR):
    cache_name = model_name.replace("/", "--").lower()
    snapshots_dir = os.path.join(cache_dir, f"models--{cache_name}", "snapshots")

    if not os.path.isdir(snapshots_dir):
        return None

    snapshots = sorted(os.listdir(snapshots_dir), reverse=True)
    for snapshot in snapshots:
        candidate = os.path.join(snapshots_dir, snapshot, file_in_repo)
        if os.path.isfile(candidate):
            return candidate

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Find the full GGUF path from the Hugging Face cache."
    )
    parser.add_argument(
        "model", type=str, help="The model name from Hugging Face"
    )
    parser.add_argument(
        "path",
        type=str,
        help="The path to the GGUF file within the model repository",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("LLAMA_CACHE_DIR", CACHE_DIR),
        help="Hugging Face cache directory",
    )
    args = parser.parse_args()

    model_path = find_model_path(args.model, args.path, args.cache_dir)
    if model_path is None:
        print(
            (
                "Error: cached file not found. "
                f"Model='{args.model}', Path='{args.path}', Cache dir='{args.cache_dir}'"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    print(model_path, end="")


if __name__ == "__main__":
    main()
