from evalscope.perf.main import run_perf_benchmark
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run performance benchmark.")
    parser.add_argument("host", help="API host and port, e.g., http://127.0.0.1:8000", default="http://localhost:8000")
    args = parser.parse_args()
    api_url = f"{args.apihost}/v1/chat/completions"

    task_cfg = {
        "url": api_url,
        "parallel": 16,
        "model": "DeepSeek-V3-Fast",
        "number": 100,
        "api": "openai",
        "api_key": "sk-2jc7k79eca#",
        "dataset": "openqa",
        "stream": True,
    }

    run_perf_benchmark(task_cfg)


if __name__ == "__main__":
    main()
