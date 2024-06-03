import modal
from modal import Secret
import os

from src.pipeline.etf_pipeline import run_pipeline

# Create an image with the necessary dependencies from requirements.txt
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

# Define the application
app = modal.App(
    "FolioLLM-pipeline",
    image=image,
    secrets=[Secret.from_name("my-huggingface-secret"), Secret.from_name("my-wandb-secret")]
)

@app.function()
def run():
    # Set the W&B API key from the secret
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = wandb_api_key
    run_pipeline()

@app.local_entrypoint()
def main():
    print("Run finetuning", run.remote())
