"""Run the web server and hourly scheduler as separate child processes."""

from __future__ import annotations

import signal
import subprocess
import sys
import time


def main() -> int:
    commands = [
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        [sys.executable, "-m", "app.scheduler"],
    ]
    processes = [subprocess.Popen(command) for command in commands]

    def stop(_signum: int, _frame: object) -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
    finally:
        stop(signal.SIGTERM, None)
        for process in processes:
            process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
