import sys
print(f"Python executable: {sys.executable}")
print(f"Is venv: {'pytorch_env' in sys.executable or 'venv' in sys.executable}")
