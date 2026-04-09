"""
update_and_run_notebook.py
==========================
Clears old outputs, then executes the notebook in-place using nbconvert.
All cell outputs (prints, plots, etc.) are saved INTO the .ipynb file.
"""
import json
import subprocess
import sys
from pathlib import Path

NOTEBOOK = Path(r"c:\Users\User\WildlifeAlberta\notebooks\phase2_image_classification (1).ipynb")

# -- Step 1: Clear old outputs so notebook runs clean --
print("Reading notebook...")
with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("Notebook cleared and kernel updated.")

# -- Step 2: Execute notebook (timeout = 1 hour for CPU training) --
print("")
print("Executing notebook... this may take 15-20 minutes on CPU.")
print("Training 10 epochs with batch_size 4 on ~182 images...")
print("")

result = subprocess.run(
    [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=3600",
        "--ExecutePreprocessor.kernel_name=python3",
        str(NOTEBOOK),
    ],
)

if result.returncode == 0:
    print("")
    print("DONE - Notebook executed successfully!")
    print("Open the notebook to see all outputs (training logs, plots, confusion matrix, labeled images).")
else:
    print("")
    print(f"FAILED - exit code {result.returncode}")
    sys.exit(1)
