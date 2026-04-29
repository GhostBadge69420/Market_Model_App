import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
PYTHON_BIN = ROOT_DIR / "venv" / "bin" / "python"
STREAMLIT_BIN = ROOT_DIR / "venv" / "bin" / "streamlit"


def stream_output(prefix, pipe):
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            print(f"[{prefix}] {line}", end="")
    finally:
        pipe.close()


def start_process(name, command, workdir):
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(
        command,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    thread = threading.Thread(
        target=stream_output,
        args=(name, process.stdout),
        daemon=True,
    )
    thread.start()
    return process


def stop_process(process):
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def main():
    if not PYTHON_BIN.exists() or not STREAMLIT_BIN.exists():
        print("Virtual environment executables were not found in venv/bin.")
        print("Activate or create the venv first, then run this script again.")
        return 1

    django_cmd = [str(PYTHON_BIN), "manage.py", "runserver", "127.0.0.1:8000"]
    streamlit_cmd = [str(STREAMLIT_BIN), "run", "app.py", "--server.headless", "true"]

    processes = [start_process("Django", django_cmd, BACKEND_DIR)]
    time.sleep(2)
    processes.append(start_process("Streamlit", streamlit_cmd, ROOT_DIR))

    def shutdown(signum=None, frame=None):
        print("\nStopping servers...")
        for process in processes:
            stop_process(process)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            for process in processes:
                if process.poll() is not None:
                    shutdown()
                    return process.returncode or 0
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()
        return 0


if __name__ == "__main__":
    sys.exit(main())
