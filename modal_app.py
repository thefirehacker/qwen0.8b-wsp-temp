"""Run the fact write on a Modal L4. Usage: modal run modal_app.py"""
import modal

app = modal.App("wsp")
vol = modal.Volume.from_name("wsp")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "safetensors",
        "huggingface_hub",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_dir(".", remote_path="/root/wsp", ignore=[".venv", ".git", "blog"])
)


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    volumes={"/vol": vol},
)
def write():
    import os
    import subprocess
    os.environ["HF_HOME"] = "/vol/hf"
    subprocess.check_call(
        [
            "python", "wsp.py", "write",
            "--model", "Qwen/Qwen3.5-0.8B",
            "--device", "cuda",
            "--facts", "examples/profile.json",
            "--out", "/vol/my-model",
        ],
        cwd="/root/wsp",
    )
    vol.commit()


@app.local_entrypoint()
def main():
    write.remote()
