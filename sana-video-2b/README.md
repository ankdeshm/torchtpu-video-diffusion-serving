# SANA-Video-2B on Google Cloud TPUs (TorchTPU)

- **Model:** [`Efficient-Large-Model/SANA-Video_2B_720p_diffusers`](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_720p_diffusers) (2B linear-attention video diffusion model targeting 720p synthesis, `bf16`)
- **License:** Apache-2.0
- **Engines Audited:** `sglang-torchtpu` (`main`, `U=1` & `U=4`) and `vllm-torchtpu v0.33.0` + `vllm-omni 0.33.0rc1` (`U=1` & `U=4`)

## Dual-Engine Compatibility Audit (Current Upstream Limitations)

Both engines were tested honestly across TPU v7x and TPU v5p; neither currently completes end-to-end 81-frame video synthesis on TorchTPU without upstream pipeline fixes:

| Engine | Hardware & Parallelism | Boot Status | Observed Failure |
| :--- | :--- | :---: | :--- |
| `sglang-torchtpu` (`main`) | TPU v5p (`U=1` and `U=4`) | Healthy (`59.6–194.1 s`) | Request execution failed inside scheduler (`The size of tensor a (128) must match ...`) at both `832×480×81f` and `1280×720×81f`. |
| `vllm-torchtpu v0.33.0` + `vllm-omni` | TPU v7x (`U=4`) | Startup error | Orchestrator dummy run failed with `PjRtBuffer::CopyToRemoteDevice: INVALID_ARGUMENT: Remote size 9175040 is not identical to local size 4165632` due to uneven Ulysses tensor sharding on linear-attention tokens. |

We document these exact upstream failure signatures transparently so teams evaluating `SANA-Video-2B` on TorchTPU can track the required upstream sharding/shape fixes, while using **`Wan2.2-T2V-A14B`** (`68.99 s` 480p / `148.71 s` 720p verified MP4s on TPU v7x) for production TorchTPU video generation today.
