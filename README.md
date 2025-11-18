# Text-to-Video Generator (AI Video Synthesis)

Create short AI-generated videos from natural language prompts using diffusion models and transformer-based conditioning. This project demonstrates end-to-end multimodal generation with a modern web UI and a Python backend designed for GPU-accelerated inference.


## What it does

- Converts a text prompt (e.g., "a golden retriever surfing at sunset in cinematic style") into a short video.
- Uses text encoders (Transformers) to condition a video diffusion model.
- Streams generation progress and returns an MP4/WebM preview and a downloadable file.
- Exposes clean HTTP APIs for programmatic use.
- Ships with a modern frontend including a 3D animated hero section powered by a Spline scene.


## Tech stack

- Backend: Python, FastAPI, PyTorch
- Models: Latent diffusion for video, transformer text encoders (CLIP/T5)
- Frontend: React (Vite), Tailwind CSS, Framer Motion, Lucide icons
- 3D Hero Animation: Spline (react-spline) with the provided scene asset
- Data/Persistence: MongoDB (pre-configured, optional for job/status storage)


## Architecture overview

1) Frontend (port 3000)
- Prompt input, settings (seed, steps, guidance, resolution, duration, fps)
- Submit to backend and display progress + results
- Hero section uses the provided Spline animation (AI aura radial gradient) as a centerpiece visual

2) Backend (port 8000)
- FastAPI service to orchestrate text encoding and diffusion sampling
- Optional job queue (MongoDB) to track generation requests and statuses
- GPU-accelerated inference with PyTorch; CPU fallback possible but slow

3) Models
- Text encoder: Transformer (e.g., CLIP or T5) encodes prompt to latent embeddings
- Video diffusion: Latent video diffusion UNet denoises a temporal latent tensor conditioned on text
- VAE: Encodes/decodes frames between pixel and latent spaces


## Project status

- The scaffold is ready (frontend + backend running). The generation endpoints are specified below and intended for implementation. Heavy model downloads and GPU acceleration are expected in a real deployment.


## API design (planned)

- POST /api/generate
  - Body: { prompt: string, negative_prompt?: string, num_inference_steps?: int, guidance_scale?: float, width?: int, height?: int, fps?: int, duration_seconds?: float, seed?: int }
  - Returns: { job_id: string }

- GET /api/status/{job_id}
  - Returns: { status: "queued"|"running"|"succeeded"|"failed", progress?: float, eta_seconds?: number, error?: string }

- GET /api/result/{job_id}
  - Returns: binary video (MP4/WebM) or JSON with signed URL if using object storage

- GET /test
  - Health and database connectivity probe (already available)

Note: The repository currently includes a hello/test endpoint for validation. The generation endpoints above are the blueprint to implement when integrating models.


## Model choices and recommendations

- Open-source video diffusion starters:
  - Stable Video Diffusion (SV3D / SVD) from Stability
  - ModelScope text-to-video (2.1) by DAMO
  - Zeroscope / AnimateDiff variants for text-to-video or image-to-video pipelines
- Text encoders: CLIP (ViT-L/14) or T5 (large) depending on pipeline support
- Sampling: DDIM / Euler / Heun with classifier-free guidance
- Typical settings: 20–40 steps, guidance 6–12, 384×224 or 512×288 at 8–16 fps, 2–4 seconds


## Performance and hardware

- Recommended: NVIDIA GPU with >= 12–24 GB VRAM for smooth generation
- Mixed precision (fp16/bf16) to reduce memory footprint
- xFormers or Torch 2.0+ scaled dot-product attention for speed
- CPU-only is possible but extremely slow and generally not recommended


## Safety, ethics, and responsible use

- Add content filters (prompt and output) to block illegal or harmful content
- Include metadata and watermarks where applicable
- Respect copyright and likeness rights; obtain consent for people-identifiable content
- Provide user reporting and takedown mechanisms


## Frontend experience

- Clean, minimal prompt UI with progress and result playback
- Adjustable parameters (steps, guidance, resolution, duration, fps)
- Spline hero animation: The landing page features a centered AI aura animation in purple/blue/orange gradients for a futuristic vibe

Using the provided Spline scene in React:

- Install and import: `import Spline from '@splinetool/react-spline'`
- Render the component with 100% width/height and avoid negative z-index; add `pointer-events-none` to overlays so they don't block interaction
- Scene URL: https://prod.spline.design/4cHQr84zOGAHOehh/scene.splinecode

Example snippet:

```
import Spline from '@splinetool/react-spline';

export default function Hero() {
  return (
    <div className="relative h-[60vh] w-full">
      <Spline scene="https://prod.spline.design/4cHQr84zOGAHOehh/scene.splinecode" style={{ width: '100%', height: '100%' }} />
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-black/30" />
      <div className="absolute inset-0 grid place-items-center">
        <h1 className="text-3xl md:text-5xl font-semibold tracking-tight">Text to Video</h1>
      </div>
    </div>
  );
}
```


## Backend integration outline

1) Environment
- Python 3.10+
- CUDA toolkit + compatible PyTorch build for GPU
- Environment variables (examples):
  - DATABASE_URL, DATABASE_NAME (optional for job storage)
  - MODEL_VARIANT (e.g., "svd", "zeroscope")
  - TORCH_DTYPE (e.g., fp16)

2) Dependencies (typical)
- torch, torchvision, torchaudio (CUDA build)
- diffusers, transformers, accelerate, xformers (optional), safetensors
- fastapi, uvicorn, pydantic
- pymongo or motor (if using MongoDB), boto3/minio (if using object storage)

3) Inference flow (typical pseudo)

```
# 1) Load pipeline
pipe = DiffusionVideoPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.enable_model_cpu_offload() or pipe.to("cuda").enable_xformers_memory_efficient_attention()

# 2) Encode text
embeds = pipe.encode_prompt(prompt, negative_prompt)

# 3) Sample latents through UNet over T timesteps
video_latents = pipe.unet_sample(embeds, steps=num_inference_steps, guidance=guidance_scale, seed=seed)

# 4) Decode using VAE and postprocess to mp4
frames = pipe.vae_decode(video_latents)
video = encode_video(frames, fps=fps)
```

4) Job orchestration
- Create a job document with status queued
- Run generation, update progress periodically
- Store artifact (grid of frames or mp4) to disk or object storage
- Expose result endpoint and signed URL


## Local development

Prerequisites:
- Node.js 18+
- Python 3.10+
- (Optional) NVIDIA GPU drivers + CUDA for acceleration

Steps:
- Copy or export environment variables for backend (database optional)
- Start services (the dev environment here auto-starts both frontend and backend)
- For local-only: run the frontend on port 3000 and backend on 8000, then set VITE_BACKEND_URL in the frontend env

Environment variables (frontend):
- VITE_BACKEND_URL=http://localhost:8000

Environment variables (backend):
- DATABASE_URL, DATABASE_NAME (optional)
- MODEL paths or IDs as required by your chosen pipeline


## Deployment notes

- Use a GPU-enabled host (A10, T4, L4, A100, H100) for reliable performance
- Containerize both services; pin model versions
- Warm up model weights at startup; cache tokenizer/encoder
- Store outputs in an object store (S3, GCS, MinIO) and serve via CDN
- Add observability: request metrics, GPU memory, queue depth


## Evaluation and quality tips

- Human eval on text-video alignment, temporal consistency, artifact score
- Automated: FVD, CLIP-score on sampled frames, motion smoothness proxies
- Prompt engineering: style tokens (cinematic, hyperrealistic), negative prompts
- Use seeds for reproducibility; keep guidance moderate to avoid flicker


## Roadmap

- Implement /api/generate with streaming progress
- Add negative prompt and style presets
- Add safety classifier and NSFW filtering
- Add user gallery and shareable links
- Fine-tuning adapters (LoRA) for custom domains
- Multi-clip composition and soundtrack support


## Acknowledgements and citations

- Diffusion and latent diffusion models
  - Ho et al., Denoising Diffusion Probabilistic Models (2020)
  - Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models (2022)
- Video diffusion references
  - Voleti et al., MCVD (2022)
  - Blattmann et al., Stable Video Diffusion (2023)
  - Yang et al., ModelScope T2V (2023)
- Libraries: PyTorch, HuggingFace Diffusers, Transformers, Accelerate, xFormers
- Spline for realtime 3D web animations


## License

This project is provided for research and educational purposes. Check the individual model licenses and asset terms (including Spline scene) before commercial use.
