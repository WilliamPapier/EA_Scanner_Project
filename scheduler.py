# scheduler.py
import time, subprocess, sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTERVAL = 60  # seconds

def run(script):
    print("Running", script)
    subprocess.run([sys.executable, script], cwd=SCRIPT_DIR)

if __name__ == "__main__":
    while True:
        run("scanner.py")
        run("infer_and_write_params.py")
        run("news_updater.py")
        time.sleep(INTERVAL)