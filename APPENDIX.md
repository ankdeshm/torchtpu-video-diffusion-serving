# Appendix — Video Diffusion VAE Memory Analysis, H.264 Post-Processing A/B Study & Diagnostics

This appendix captures the root-cause analysis of 3D causal VAE memory limits, our live local-vs-remote H.264 post-processing systems A/B benchmark, stage-level profiling (`DenoisingStage` vs. `DecodingStage`), and exact failure diagnostics across `Wan2.2` and `SANA-Video-2B`.

---

## 1. Solving the 3D Causal VAE `DecodingStage` VMEM OOM

Without parallel VAE decoding, `Wan2.2-T2V-A14B` completed its 20-step DiT denoising (`16.52 s` on TPU v7x, `30.47 s` on TPU v5p) and then failed during 81-frame 3D VAE decoding due to monolithic VMEM/HBM activation spikes.

Adding **`--vae-config.use-parallel-decode --vae-config.parallel-decode-mode tiled`** splits the 3D VAE decode across spatial/temporal tiles and across TPU cores:

- **TPU v7x (4 chips / 8 TensorCores, `768 GB` total HBM, `U=8`):**
  - `832×480 × 81f`: Peak HBM settled at **`60.3 GB`** per core (cold `63.4 GB`), producing a verified `438 KB` H.264 MP4 in **`68.99 s` warm** (`77.61 s` cold).
  - `1280×720 × 81f`: Peak HBM settled at **`71.4 GB`** per core (~25 GB headroom per core), producing a verified `772 KB` H.264 MP4 in **`148.71 s` warm** (`172.08 s` cold).
- **TPU v5p (4 chips, `380 GB` total HBM, `U=4`):**
  - With all 27B dual-DiT parameters (`transformer` + `transformer_2`) plus T5-XXL resident in HBM simultaneously, 4-chip TPU v5p hit an HBM allocation shortfall of just `17 MB` (`Attempting to allocate 76.17M; 59.10M free`) during `DecodingStage`, highlighting the value of TPU v7x's `2×` higher aggregate HBM capacity (`768 GB` vs. `380 GB`) on a 4-chip host.

---

## 2. Systems Engineering A/B Study: Local TPU Host CPU vs. Remote AVX-512 H.264 Encode Offload

A common architectural question in video-serving clusters is whether to offload final RGB24 -> H.264 MP4 encoding from the TPU host VM to a separate remote CPU offload node (`c3-standard-44` with AVX-512). We measured 81-frame RGB24 -> MP4 (`libx264 ultrafast`, `CRF 20`) end-to-end latency directly from a 224-vCPU TPU v7x host:

| Workload (81 Frames, Raw RGB24 Size) | Local TPU v7x Host (`4` Threads) | Local TPU v7x Host (`16` Threads) | Remote `c3-standard-44` Offload (`16` Threads, incl. TCP Transfer) | Engineering Verdict (`c=1`) |
| :--- | :---: | :---: | :---: | :--- |
| **`480p_81f`** (`832×480`, `97.0 MB`) | 253.5 ms (`319.5 FPS`) | **239.4 ms (`338.4 FPS`)** | 277.7 ms (`291.7 FPS`) | Local 16-thread encode is **14% faster** than remote offload (`-38.3 ms`) |
| **`720p_81f`** (`1280×720`, `224.0 MB`) | 531.6 ms (`152.4 FPS`) | **469.2 ms (`172.6 FPS`)** | 520.2 ms (`155.7 FPS`) | Local 16-thread encode is **10% faster** than remote offload (`-51.0 ms`) |

**Key Insight:** At `c=1`, serializing and transmitting `97–224 MB` of uncompressed RGB24 frames over TCP adds more overhead than the remote AVX-512 CPU saves, and H.264 encoding (`0.24–0.47 s`) is `<0.4%` of total 81-frame video generation latency (`69–149 s`). Keep H.264 encoding **local on the TPU host** unless running high multi-stream concurrency where host CPU saturation becomes a bottleneck.

---

## 3. Stage-Level Profiling & Engine Compatibility Diagnostics

1. **Stage-Level `DenoisingStage` Comparison (`Wan2.2-T2V-A14B`, `832×480×81f`, 20 steps):**
   - TPU v7x (`4` chips / `8` cores, `U=8`, `tpu_splash_attn`): **`16.52 s` (`0.809 s/step`)**
   - TPU v5p (`4` chips, `U=4`, `torch_sdpa`): **`30.47 s` (`1.518 s/step`)** -> **1.85× faster denoising on TPU v7x**
2. **`Wan2.2-T2V-A14B` on `vllm-omni 0.33.0rc1` (`U=4` on 4-chip TPU v7x):**
   - Initial JSON requests returned `HTTP 400: body.prompt Field required` because `vllm-omni`'s `/v1/videos` endpoint parses `multipart/form-data` fields (`prompt`, `size`, `num_frames`, `num_inference_steps`, `fps`).
   - Switching to `multipart/form-data` booted in `90.3 s` (warm cache; `586.1 s` cold) and generated PyAV-verified 81-frame `16 fps` H.264 MP4s in **`40.06 s` warm** (`212.42 s` cold; `1.94 MB`, `pixel_std = 61.2`) at `832×480` and **`137.20 s` warm** (`229.41 s` cold; `4.30 MB`, `pixel_std = 64.1`) at `1280×720`.
3. **`Wan2.2-Animate-2-14B-Distilled` attention backend constraint:**
   - Rejects `--attention-backend tpu_splash_attn` (`ValueError: its in-context self-attention runs on torch flex_attention only`). Dropping `--attention-backend` yields clean startup (`523.7 s` on TPU v7x, `U=8`) and requires `"image_path"` conditioning.
4. **`SANA-Video-2B` upstream compatibility boundaries:**
   - `sglang-torchtpu` (`main`, `U=1` & `U=4`) booted cleanly (`59.6–194.1 s`) but failed inside scheduler execution with a tensor-dimension mismatch (`size of tensor a (128)`).
   - `vllm-omni 0.33.0rc1` (`U=4`) failed orchestrator startup with `PjRtBuffer::CopyToRemoteDevice: Remote size 9175040 is not identical to local size 4165632`.
