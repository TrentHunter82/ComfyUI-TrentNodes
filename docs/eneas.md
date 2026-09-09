# Eneas Segment & Track (Trent)

Add **Trent → Segmentation → Eneas Segment & Track (Trent)**. Connect an `IMAGE` batch from Load Image or a video loader, then connect `mask` to Mask to Image → Preview Image, a compositor, or an inpainting node.

Modes:

- `track_text`: describe a particular object, such as “the person in the red jacket.” Eneas uses Florence-2 to locate it and SeC-4B to track it through the batch.
- `track_points`: enter pixel coordinates and labels, e.g. `[[180,120,1],[20,20,0]]`. Label 1 includes the object; 0 excludes background. Text is ignored.
- `category`: enter a category such as “person.” Florence-2, SigLIP2, Qwen3-VL and SAM2.1 detect instances independently in each frame. The output combines all detected instances into one mask per frame. This mode does not track instance identities.

`annotation_frame` is zero-based and applies to tracking modes. Tracking runs in both directions. Category thresholds apply only to category mode. Frames stay on CPU by default to reduce VRAM use.

Output: float32 `MASK` of shape `[frames,height,width]`, in original frame order and resolution. White (1) is the selected foreground, black (0) is background. These are binary segmentation masks, not soft hair/alpha mattes. Batch frames are staged as high-quality JPEGs because upstream SeC's frame reader requires JPEGs.

## Installation and model storage

On this machine the installed configuration uses `/mnt/d/models/eneas` (`D:\models\eneas`) and reuses `/mnt/d/models/florence2/large`. Qwen weights go in `eneas/ollama`; the other weights go in `eneas/huggingface/hub`. Local settings are in `TrentNodes/.eneas/config.json`.

To reproduce the installation in WSL:

```bash
cd /home/trent/ComfyUI
bash custom_nodes/TrentNodes/tools/install_eneas.sh /mnt/d/models/eneas /mnt/d/models/florence2/large
```

Requires `uv`, Git, curl, Linux x86-64, and an NVIDIA CUDA GPU. Omit the second argument to download Florence-2 automatically. The installer sets up Eneas in `.eneas/source/.venv` using its upstream lockfile and Ollama 0.33.3 in `.eneas/ollama`. No packages are installed into ComfyUI's Python environment. Models download on first use; interrupted Hugging Face downloads can resume. Setup never runs automatically during node import.

The installed Eneas source is pinned to `c59a028f19573720fc1c52fea21874d33e02ed3f`; Hugging Face revisions are pinned in `tools/eneas_models.json`. The installer applies one upstream correction: reverse propagation must begin at the annotation frame, otherwise frame 0 is omitted when annotating frame 1. Keep that fix until upgrading to a verified upstream release.

The node follows TrentNodes' existing automatic node registration. Eneas runs in a fresh subprocess because its Transformers 4.53 dependency conflicts with this ComfyUI environment's Transformers 5.8. ComfyUI models are unloaded before inference; Eneas memory is released on worker exit. Models reload each execution, so large weights on the mounted Windows drive add startup time. Cancelling interrupts the worker process group and cleans up temporary frames. Category mode starts a private loopback-only Ollama server for that run and stops it afterward; it does not require a system service or send frames to a cloud provider.

Models: SeC-4B, Florence-2-large, SigLIP2 base NaFlex, SAM2.1 Hiera large, and `qwen3-vl:2b-instruct-q8_0`. Qwen uses the upstream Ollama tag; the Hugging Face models use the recorded commit revisions.

Sources: [Eneas](https://github.com/speridlabs/eneas), [ComfyUI node documentation](https://docs.comfy.org/custom-nodes/overview), [Ollama Linux installation](https://docs.ollama.com/linux).
