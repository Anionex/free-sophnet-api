import uvicorn
from config import cfg_obj


def main():
    # 使用命令行参数直接启动uvicorn
    import sys

    sys.argv = [
        "uvicorn",
        "src.forward:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        f"--workers={cfg_obj.workers}",
        f"--log-level={cfg_obj.log_level.lower()}",
        "--loop=uvloop",
        "--http=httptools",
        "--limit-concurrency=1000",
        "--backlog=2048",
        f"--timeout-keep-alive={30}",
    ]
    uvicorn.main()

def main_single_worker():
        # 使用命令行参数直接启动uvicorn
    import sys

    sys.argv = [
        "uvicorn",
        "src.forward:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        f"--workers=1",
        f"--log-level={cfg_obj.log_level.lower()}",
        "--loop=uvloop",
        "--http=httptools",
        "--limit-concurrency=1000",
        "--backlog=2048",
        f"--timeout-keep-alive={30}",
    ]
    uvicorn.main()

if __name__ == "__main__":
    main()
