# Wan2.2-T2V-A14B on Google Cloud TPUs (TorchTPU)

- **Model:** [`Wan-AI/Wan2.2-T2V-A14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) (27B total / 14B active dual-transformer MoE video diffusion model + T5-XXL + 3D causal VAE, `bf16`)
- **License:** Apache-2.0
- **Engines Tested:** `sglang-torchtpu` (`main`, `U=8`, `tpu_splash_attn`, `--vae-config.use-parallel-decode --vae-config.parallel-decode-mode tiled`) and `vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1`

## Verified Dual-Engine End-to-End Video Generation Results (`81` Frames, `16 fps` H.264 MP4)

Both **`vllm-omni 0.33.0rc1` (`U=4`)** and **`sglang-torchtpu` (`U=8`, tiled parallel VAE decode)** produce verified 81-frame H.264 MP4 videos end-to-end on 4-chip TPU v7x:

| TPU Hardware & Engine | Parallelism & Caching Flags | Resolution × Frames (Steps) | Phase | End-to-End Latency | Peak HBM / Core | Output MP4 Verified |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **TPU v7x (4 chips)** — **`vllm-omni 0.33.0rc1`** | **`U=4 + Cache-DiT`** | **`832×480 × 81f`** (20 steps) | Cold | 56.08 s | — | **Yes** (`2.98 MB`, `h264`, 81f, `pixel_std = 51.5`) |
| **TPU v7x (4 chips)** — **`vllm-omni 0.33.0rc1`** | **`U=4 + Cache-DiT`** | **`832×480 × 81f`** (20 steps) | **Warm** | **22.03 s** (`21.63 s` infer) | — | **Yes** (`1.99 MB`, `h264`, 81f, `pixel_std = 65.1`) |
| **TPU v7x (4 chips)** — `vllm-omni 0.33.0rc1` | `CFG=2×U=2 + Cache-DiT` | `832×480 × 81f` (20 steps) | **Warm** | 24.03 s (`22.57 s` infer) | — | **Yes** (`2.87 MB`, `h264`, 81f, `pixel_std = 66.1`) |
| **TPU v7x (4 chips)** — `vllm-omni 0.33.0rc1` | `U=4` (exact) | `832×480 × 81f` (20 steps) | **Warm** | **40.06 s** (`212.42 s` cold) | — | **Yes** (`1.94 MB`, `h264`, 81f, `pixel_std = 61.2`) |
| **TPU v7x (4 chips)** — `vllm-omni 0.33.0rc1` | `CFG=2×U=2` (exact) | `832×480 × 81f` (20 steps) | **Warm** | 42.04 s (`64.08 s` cold) | — | **Yes** (`2.44 MB`, `h264`, 81f, `pixel_std = 55.9`) |
| **TPU v7x (4 chips)** — **`vllm-omni 0.33.0rc1`** | **`U=4 + Cache-DiT`** | **`1280×720 × 81f`** (20 steps) | Cold | 104.11 s | — | **Yes** (`4.38 MB`, `h264`, 81f, `pixel_std = 62.2`) |
| **TPU v7x (4 chips)** — **`vllm-omni 0.33.0rc1`** | **`U=4 + Cache-DiT`** | **`1280×720 × 81f`** (20 steps) | **Warm** | **74.07 s** (`73.78 s` infer) | — | **Yes** (`4.92 MB`, `h264`, 81f, `pixel_std = 53.3`) |
| **TPU v7x (4 chips)** — `vllm-omni 0.33.0rc1` | `CFG=2×U=2 + Cache-DiT` | `1280×720 × 81f` (20 steps) | **Warm** | 78.06 s (`94.09 s` cold) | — | **Yes** (`5.77 MB`, `h264`, 81f, `pixel_std = 64.5`) |
| **TPU v7x (4 chips)** — `vllm-omni 0.33.0rc1` | `U=4` (exact) | `1280×720 × 81f` (20 steps) | **Warm** | **137.20 s** (`229.41 s` cold) | — | **Yes** (`4.30 MB`, `h264`, 81f, `pixel_std = 64.1`) |
| **TPU v7x (4 chips)** — `vllm-omni 0.33.0rc1` | `CFG=2×U=2` (exact) | `1280×720 × 81f` (20 steps) | **Warm** | 142.11 s (`158.13 s` cold) | — | **Yes** (`4.05 MB`, `h264`, 81f, `pixel_std = 58.6`) |
| **TPU v7x (4 chips / 8 cores)** — `sglang-torchtpu` | `U=8`, `tpu_splash_attn`, tiled parallel VAE | `832×480 × 81f` (20 steps) | **Warm** | 68.99 s (`77.61 s` cold) | 60.3 GB | **Yes** (`438 KB`, `h264`, 81f) |
| **TPU v7x (4 chips / 8 cores)** — `sglang-torchtpu` | `U=8`, `tpu_splash_attn`, tiled parallel VAE | `1280×720 × 81f` (20 steps) | **Warm** | 148.71 s (`172.08 s` cold) | 71.4 GB | **Yes** (`772 KB`, `h264`, 81f) |

### Key Takeaways & Memory Budget Findings
- **`Cache-DiT` on `vllm-omni 0.33.0rc1` cuts warm latency by `1.82×–1.85×`:** Adding `--cache-backend cache_dit` lowers `U=4` warm end-to-end latency from `40.06 s` to **`22.03 s`** at 480p (**`3.13×` faster** than `sglang-torchtpu`) and from `137.20 s` to **`74.07 s`** at 720p (**`2.01×` faster** than `sglang-torchtpu`), while keeping full 81-frame H.264 output variance (`pixel_std = 53.3–65.1`).
- **`U=4` beats `CFG=2 × U=2` on 4 TPU v7x devices:** Pure Ulysses sequence parallelism (`U=4`) delivers `5%–9%` lower warm latency than `CFG=2 × U=2` both without caching (`40.06 s` vs. `42.04 s` at 480p; `137.20 s` vs. `142.11 s` at 720p) and with `Cache-DiT` (`22.03 s` vs. `24.03 s` at 480p; `74.07 s` vs. `78.06 s` at 720p).
- **1.85× faster DiT denoising on TPU v7x vs. TPU v5p:** In stage-level profiling of 20-step `832×480×81f` denoising (`sglang-torchtpu`), `DenoisingStage` completed in **`16.52 s` (`0.809 s/step`)** on 4-chip TPU v7x (`U=8`, `tpu_splash_attn`) vs. **`30.47 s` (`1.518 s/step`)** on 4-chip TPU v5p (`U=4`). Without `--text-encoder-cpu-offload`, 4-chip TPU v5p (`380 GB` total HBM) fell short by `17 MB` during `DecodingStage` (`76.17M` requested vs. `59.10M` free), and `--dit-layerwise-offload true` fails at `layerwise_offload.py:1609` due to CPU/TPU tensor type mismatch on `variable.set_data(tensor)`.

## Reproduce

```bash
# Option A: vLLM-Omni 0.33.0rc1 on 4x TPU v7x (U=4 + Cache-DiT, lowest warm latency)
vllm serve /path/to/Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --omni --served-model-name Wan-AI/Wan2.2-T2V-A14B \
  --host 0.0.0.0 --port 8000 --ulysses-degree 4 --cache-backend cache_dit

# Option B: SGLang on 4x TPU v7x (U=8, tiled parallel VAE decode)
sglang serve \
  --model-path /path/to/Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --host 0.0.0.0 --port 8000 \
  --num-gpus 8 --ulysses-degree 8 \
  --attention-backend tpu_splash_attn \
  --vae-config.use-parallel-decode \
  --vae-config.parallel-decode-mode tiled

python3 ../scripts/benchmark_video_diffusion.py \
  --model Wan-AI/Wan2.2-T2V-A14B --width 832 --height 480 --frames 81 --steps 20
```
