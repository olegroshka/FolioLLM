import modal
from modal import Secret
import os

from src.pipeline.etf_pipeline import run_pipeline

# Create an image with the necessary dependencies from requirements.txt
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

# Define the application with mounts and GPU requirements
app = modal.App(
    "FolioLLM-pipeline",
    image=image,
    secrets=[Secret.from_name("my-huggingface-secret"), Secret.from_name("my-wandb-secret")],
    mounts=[
        modal.Mount.from_local_dir("data", remote_path="/root/data")
    ]
)


@app.function(gpu="A100")  # Request a specific GPU type, e.g., A100, V100, etc.
def run():
    # Define the absolute path for the JSON file
    json_structured_file = "/root/data/etf_data_v3_plain.json"

    # Ensure the W&B API key is set from the secret
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = wandb_api_key

    run_pipeline(json_structured_file=json_structured_file)


@app.local_entrypoint()
def main():
    print("Run finetuning", run.remote())
