# Wan2.2-I2V-A14B on Google Cloud TPUs (TorchTPU)

- **Model:** [`Wan-AI/Wan2.2-I2V-A14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) (27B total / 14B active dual-transformer MoE image-to-video diffusion model + 3D VAE, `bf16`)
- **License:** Apache-2.0
- **Engines Tested:** `vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1` (`U=4` on TPU v7x) and `sglang-torchtpu` (`main`, `U=8` on TPU v7x / `U=4` on TPU v5p)

## Verified End-to-End Image-to-Video Results (`vllm-omni 0.33.0rc1`, 4-Chip TPU v7x)

Every generated MP4 below was decoded with PyAV and verified to be a valid 81-frame `16 fps` H.264 video (`pixel_std = 46.3–75.6`, `valid_video = True`):

| TPU Hardware & Engine | Parallelism & Caching | Resolution × Frames (Steps) | Phase | Boot Time (Warm) | End-to-End Latency | Output MP4 Verified |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **TPU v7x (4 chips)** — **`vllm-omni 0.33.0rc1`** | **`U=4 + Cache-DiT`** | **`832×480 × 81f`** (20 steps) | Cold | 57.0 s | 26.08 s (`24.44 s` infer) | **Yes** (`3.62 MB`, `h264`, 81f, `pixel_std = 69.6`) |
| **TPU v7x (4 chips)** — **`vllm-omni 0.33.0rc1`** | **`U=4 + Cache-DiT`** | **`832×480 × 81f`** (20 steps) | **Warm** | 57.0 s | **24.03 s** (`22.63 s` infer) | **Yes** (`4.64 MB`, `h264`, 81f, `pixel_std = 56.9`) |
| **TPU v7x (4 chips)** — `vllm-omni 0.33.0rc1` | `U=4` (exact) | `832×480 × 81f` (20 steps) | **Warm** | 57.0 s | **42.07 s** (`71.29 s` cold) | **Yes** (`3.04 MB`, `h264`, 81f, `pixel_std = 53.7`) |
| **TPU v7x (4 chips)** — **`vllm-omni 0.33.0rc1`** | **`U=4 + Cache-DiT`** | **`1280×720 × 81f`** (20 steps) | Cold | 57.0 s | 88.27 s (`86.30 s` infer) | **Yes** (`10.32 MB`, `h264`, 81f, `pixel_std = 72.8`) |
| **TPU v7x (4 chips)** — **`vllm-omni 0.33.0rc1`** | **`U=4 + Cache-DiT`** | **`1280×720 × 81f`** (20 steps) | **Warm** | 57.0 s | **84.09 s** (`83.37 s` infer) | **Yes** (`10.11 MB`, `h264`, 81f, `pixel_std = 75.3`) |
| **TPU v7x (4 chips)** — `vllm-omni 0.33.0rc1` | `U=4` (exact) | `1280×720 × 81f` (20 steps) | **Warm** | 57.0 s | **140.21 s** (`168.27 s` cold) | **Yes** (`10.91 MB`, `h264`, 81f, `pixel_std = 57.7`) |

## Key Engineering & Engine Notes

1. **`vllm-omni 0.33.0rc1` + `Cache-DiT` cuts `Wan2.2-I2V-A14B` warm latency by `1.67×–1.75×`:** Passing `--cache-backend cache_dit` with `--ulysses-degree 4` drops 81-frame image-to-video latency from `42.07 s` to **`24.03 s`** at 480p and from `140.21 s` to **`84.09 s`** at 720p (PyAV-verified H.264, `pixel_std = 56.9–75.3`). Server boot takes `57.0 s` from warm cache (pass `--init-timeout 1800 --stage-init-timeout 1800` when reading from cold object storage).
2. **Upload reference image via `multipart/form-data` (`input_reference`):** Send the conditioning PNG as the `input_reference` file upload field alongside `prompt`, `size`, `num_frames`, `num_inference_steps`, and `fps` on `/v1/videos`.
3. **`sglang-torchtpu` findings on `Wan2.2-I2V-A14B`:**
   - Default `dit_cpu_offload=True` across 8 workers can exhaust host container RAM (`~65 GB` RSS per worker) when loading both 26.6 GB transformers simultaneously; pass `--dit-cpu-offload false` when running on 4-chip TPU v7x (`768 GB` HBM).
   - With tiled parallel VAE decode (`--vae-config.use-parallel-decode --vae-config.parallel-decode-mode tiled`), both 4-chip TPU v7x (`U=8`) and 4-chip TPU v5p (`U=4`) raise `PjRtBuffer::CopyToRemoteDevice: Remote size 23040 is not identical to local size 737280` inside `parallel_conv.py:344` (`halo_exchange`). Without parallel VAE decode (`--warmup-mode off`), the server boots in `54.0 s` from RAM disk, but video jobs remained queued without completing within `900 s`.

## Reproduce

```bash
# Recommended: vLLM-Omni 0.33.0rc1 on 4x TPU v7x (U=4 + Cache-DiT)
vllm serve /path/to/Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --omni --served-model-name Wan-AI/Wan2.2-I2V-A14B \
  --host 0.0.0.0 --port 8000 --ulysses-degree 4 --cache-backend cache_dit \
  --init-timeout 1800 --stage-init-timeout 1800

# Submit an async multipart/form-data image-to-video request
curl -X POST http://127.0.0.1:8000/v1/videos \
  -F "model=Wan-AI/Wan2.2-I2V-A14B" \
  -F "prompt=Gentle ripples across the alpine lake as clouds drift slowly across sunlit snow peaks." \
  -F "input_reference=@/path/to/reference_frame.png;type=image/png" \
  -F "size=832x480" -F "num_frames=81" -F "num_inference_steps=20" -F "fps=16"
```
