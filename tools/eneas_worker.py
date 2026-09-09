"""Runs only in the private Eneas environment; never imported by ComfyUI."""

import json
import logging
import os
from pathlib import Path
import sys
import socket
import subprocess
import time
from contextlib import contextmanager
from urllib.request import urlopen
from urllib.error import URLError

import numpy as np
from huggingface_hub import snapshot_download
import torch


def model_path(repo, root):
    revisions = json.loads(Path(__file__).with_name("eneas_models.json").read_text())
    return snapshot_download(repo, revision=revisions[repo], cache_dir=str(root / "huggingface/hub"),
                             allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.model", "*.txt", "*.pt"],
                             ignore_patterns=["onnx/*", "coreml/*"], max_workers=4)


@contextmanager
def category_server(work, root):
    binary = Path(__file__).resolve().parents[1] / ".eneas/ollama/bin/ollama"
    if not binary.is_file():
        raise RuntimeError("Install the private Ollama runtime: bash custom_nodes/TrentNodes/tools/install_eneas.sh")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    host = f"http://127.0.0.1:{port}"
    os.environ["OLLAMA_HOST"] = host
    env = dict(os.environ, OLLAMA_MODELS=str(root / "ollama"), OLLAMA_NO_CLOUD="1", OLLAMA_KEEP_ALIVE="5m",
               HOME=str(work / "ollama-home"))
    with (work / "ollama.log").open("w") as log:
        process = subprocess.Popen([str(binary), "serve"], env=env, stdout=log, stderr=log)
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError("Private Ollama failed to start: " + (work / "ollama.log").read_text()[-3000:])
                try:
                    with urlopen(host + "/api/tags", timeout=1):
                        break
                except (URLError, TimeoutError):
                    time.sleep(0.25)
            else:
                raise RuntimeError("Private Ollama startup timed out")
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run(work):
    request = json.loads((work / "request.json").read_text())
    root = Path(request["model_root"])
    torch.cuda.set_device(request["device"])
    # Upstream tests equality with 'cuda'; set the current device explicitly first.
    from eneas.segmentation import UniqueInstanceSegmenter, GenericCategorySegmenter

    mode = request["mode"]
    florence = request["florence_path"]
    if mode != "track_points" and not florence:
        florence = model_path("microsoft/Florence-2-large", root)
    if mode == "category":
        segmenter = GenericCategorySegmenter(
            grounding_model_path=florence,
            image_text_model_path=model_path("google/siglip2-base-patch16-naflex", root),
            sam2_model_path=str(Path(model_path("facebook/sam2.1-hiera-large", root)) / "sam2.1_hiera_large.pt"),
            device="cuda")
        with category_server(work, root):
            import ollama
            # Upstream suppresses pull failures; require the validator to be available.
            ollama.pull("qwen3-vl:2b-instruct-q8_0")
            result = segmenter.segment(str(work / "frames"), category=request["text"],
                                       accept_threshold=request["accept_threshold"], reject_threshold=request["reject_threshold"])
    else:
        segmenter = UniqueInstanceSegmenter(segmentation_model_path=model_path("OpenIXCLab/SeC-4B", root),
                                            grounding_model_path=florence, device="cuda", sam_encoder="long-large")
        args = {"text": request["text"]} if mode == "track_text" else {
            "points": [p[:2] for p in request["points"]], "labels": [int(p[2]) for p in request["points"]]}
        result = segmenter.segment(str(work / "frames"), annotation_frame=f'{request["annotation_frame"]:08d}.jpg',
                                   offload_frames_to_gpu=not request["offload_frames_to_cpu"], **args)
    masks = np.zeros((request["count"], request["height"], request["width"]), dtype=np.uint8)
    if set(result.masks) != set(range(request["count"])):
        raise RuntimeError("Eneas did not return every input frame; refusing to misalign the masks")
    for index, value in result.masks.items():
        if mode == "category":
            for instance in value:
                masks[index] |= instance
        else:
            masks[index] = value
    np.save(work / "masks.npy", masks, allow_pickle=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(Path(sys.argv[1]))
