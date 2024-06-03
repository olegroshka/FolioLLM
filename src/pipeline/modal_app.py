import modal

from src.pipeline.etf_pipeline import run_pipeline

image = modal.Image.debian_slim().pip_install_from_requirements("/mnt/data/requirements.txt")

app = modal.App(
    "FolioLLM-pipeline"
)  # Note: prior to April 2024, "app" was called "stub"

@app.function()
def run():
    run_pipeline()

@app.local_entrypoint()
def main():
    print("Run finetuning", run.remote())
