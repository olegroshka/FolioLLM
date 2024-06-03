import modal
from src.pipeline.etf_pipeline import run_pipeline

# Create an image with the necessary dependencies from requirements.txt
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

# Define the application
app = modal.App("FolioLLM-pipeline", image=image)

@app.function()
def run():
    run_pipeline()

@app.local_entrypoint()
def main():
    print("Run finetuning", run.remote())
