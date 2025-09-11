#!/usr/bin/env python3
"""
EA Scanner Project Setup and Runner
"""

import os
import sys
import subprocess

def install_dependencies():
    """Install required Python packages"""
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_scanner():
    """Run the EA Scanner"""
    print("Starting EA Scanner...")
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    scheduler_path = os.path.join(src_dir, "scheduler.py")
    if not os.path.exists(scheduler_path):
        print(f"Error: {scheduler_path} not found")
        return
    subprocess.run([sys.executable, scheduler_path])

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "install":
            install_dependencies()
        elif sys.argv[1] == "run":
            run_scanner()
        else:
            print("Usage: python setup.py [install|run]")
    else:
        print("EA Scanner Project")
        print("Available commands:")
        print("  python setup.py install  - Install dependencies")
        print("  python setup.py run      - Run the scanner")

if __name__ == "__main__":
    main()