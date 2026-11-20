# Wan2.2-Animate-2-14B-Distilled on Google Cloud TPUs (TorchTPU)

- **Model:** [`Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers) (14B distilled character-animation video diffusion model, `bf16`)
- **License:** Apache-2.0
- **Engines Audited:** `sglang-torchtpu` (`main`, `U=8` on 4-chip TPU v7x with tiled parallel VAE decode) and `vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1` (`U=4`)

## Verified Dual-Engine Serving & Kernel Audit Findings

1. **`sglang-torchtpu` (`main`): clean startup when omitting `--attention-backend`:**
   - Passing `--attention-backend tpu_splash_attn` raises an explicit startup validation error in `sglang-torchtpu`:
     `ValueError: Wan-Animate-2 does not support selecting the DiT attention backend (...): its in-context self-attention runs on torch flex_attention only. Drop the flag;`
   - Dropping `--attention-backend` lets `Wan2.2-Animate-2-14B-Distilled` boot cleanly to `/health 200 OK` in **`393 s` (`59 s` warm cache)** on 4-chip TPU v7x (`U=8`) with tiled parallel VAE decode.
2. **Required conditioning inputs (`input_reference` + `video_path`):**
   - On `/v1/videos`, `sglang` maps JSON `"input_reference"` to `image_path` and `"video_path"` to the driving reference video (omitting `"video_path"` returns `Wan-Animate-2 requires 'video_path' to be a string, got None`).
3. **Upstream `torch_tpu` kernel boundary (`flex_attention`):**
   - Once both `"input_reference"` and `"video_path"` are supplied, execution reaches `WanAnimate2BeforeDenoisingStage` and then fails inside TorchDynamo fake-tensor tracing on `torch_tpu 0.2.0` (`PrivateUse1`) with:
     `NotImplementedError: could not find kernel for HigherOrderOperator flex_attention at dispatch key DispatchKey.AutocastPrivateUse1`
4. **`vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1` compatibility boundary:**
   - Launching `vllm serve .../Wan2.2-Animate-2-14B-Distilled-Diffusers --omni` fails worker initialization (`vllm_omni/diffusion/registry.py:592`) with:
     `ValueError: Model class WanAnimate2Pipeline not found in diffusion model registry.`
   - **Verdict:** Serving `Wan2.2-Animate-2-14B-Distilled` end-to-end on TorchTPU requires upstream `torch_tpu` dispatch support for `flex_attention` under `AutocastPrivateUse1` (in `sglang`) or adding `WanAnimate2Pipeline` to `vllm-omni`'s diffusion registry. Use **`Wan2.2-T2V-A14B`** (`40.06 s` 480p / `137.20 s` 720p on `vllm-omni`) or **`Wan2.2-I2V-A14B`** (`42.07 s` 480p / `140.21 s` 720p on `vllm-omni`) for production TorchTPU video generation today.

## Reproduce

```bash
# Omit --attention-backend so Wan-Animate-2 boots cleanly on TorchTPU
sglang serve \
  --model-path /path/to/Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers \
  --host 0.0.0.0 --port 8000 \
  --num-gpus 8 --ulysses-degree 8 \
  --vae-config.use-parallel-decode \
  --vae-config.parallel-decode-mode tiled
```
