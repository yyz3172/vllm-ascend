# TurboQuant 4bit Attention Optimization Plan

## Baseline

- Date: 2026-06-20
- Command: `source xrx_infoenvs && XRX_TQ4BIT_PROFILE=1 python tests/e2e/singlecard/xrx_turboquant4bit_smoke.py`
- Profile: `/root/x00827378/perflog2/rank0_3556277_20260620175317500_ascend_pt`
- Smoke output:
  - `Shin, and I’m a student of the University of York...`
  - `TurboQuant技术是一种专注于量子计算...`
- `TurboquantAttentionPaged4bit`: 84 calls, avg 29.68 us, min 27.26 us, max 33.74 us.
- `TurboquantPackKvForCache4bit`: not used in this baseline. The smoke script uses `VLLM_ASCEND_TURBOQUANT_ENCODE_OP=0` because the current fused 4bit pack path causes bad generation in real vLLM decode.
- `reshape_and_cache`: avg 33680.90 us with fallback pack. This is not the attention optimization target.

## Correctness Gates

- Run smoke with two prompts after every kept attention change.
- Confirm `TurboquantAttentionPaged4bit` exists in profile output.
- Keep an optimization only if average `TurboquantAttentionPaged4bit` improves versus the latest accepted baseline. Revert changes with no improvement or worse output.
- Do not re-enable fused 4bit pack for smoke until its generation correctness issue is fixed separately.

## Planned Attention Work

1. Remove obvious rollback residue and redundant work in `turboquant_attention_paged4bit`.
   - Status: completed for current low-risk items.
   - Candidate: duplicate local variable assignment in `ProcessSplitBn`.
   - Candidate: hoist `lowMask` initialization out of each `LoadPackedTileRows` call, if measurable.
   - Accepted: hoisted `lowMask` init once per worker.
   - Accepted: removed the unnecessary vector pipe barrier after Q cast in `LoadQGroup`.

2. Reduce scalar norm handling in decode.
   - Status: pending.
   - Rejected: copy full 4-row physical group with `DataCopyPad` and read norm from UB. It regressed smoke profile, likely because the added 264B padded copy and MTE2->S sync cost more than two scalar norm byte reads for the current decode workload.
   - Rejected: removing the Vector-only barriers inside `LoadPackedTileRows` row unpack. Smoke output stayed normal, but profile regressed slightly.
   - Candidate: copy only the 8 norm bytes for a group into a small UB buffer or batch-convert norms after index unpack without increasing index copy size.
   - Risk: bf16/fp16 bit reinterpret must remain exact.

3. Reduce QK/PV scalar loops.
   - Status: in progress.
   - Candidate: replace per-row QK `ReduceSum` loop with larger vector batches or Cube QK/PV path.
   - Accepted: Candidate C reordered `VectorQk` to cast each K row once and reuse it across all GQA heads with a separate reduce workspace.
   - Accepted: Candidate D split `OnlineSoftmaxUpdateTile` into a per-G softmax/state pass and an m-major PV pass so each V row is cast once and reused across all GQA heads.
   - Rejected: Candidate E attempted to remove the per-GQA-head vector barrier after `outAcc *= alpha`, but it could not be validated because the NPU entered a no-process HBM/AICore busy state before a valid smoke profile was generated. The change was reverted to keep the latest measured baseline.
   - Risk: larger change; must validate against existing attention unit tests and smoke profile.

4. Reduce output-stage vector barriers.
   - Status: pending.
   - Rejected: removing per-GQA-head vector pipe barriers between `Muls` and `Cast` in `WriteFinalOutput`. It regressed smoke profile, so the barriers are still required for the current implementation.

5. Revisit tiling/core use.
   - Status: pending.
   - Rejected: removing the fixed 16-core cap in the 4bit attention host tiling and deriving data-parallel MIX groups from physical AIC/AIV counts. It increased launch/idle worker overhead for the current small-BN smoke workload.
   - Candidate: evaluate SplitBNS only for longer sequences. Current smoke uses SplitBN vector path.
   - Risk: workspace and SyncAll behavior.

## Progress Log

- 2026-06-20: Reproduced bad output with fused 4bit pack enabled.
- 2026-06-20: Confirmed `encode=0 decode=1` smoke/debug output is sane, while `encode=1 decode=0` is bad. Fused 4bit pack correctness remains a separate issue.
- 2026-06-20: Changed `xrx_turboquant4bit_smoke.py` to set env before vLLM import and to use fallback pack for correctness baseline.
- 2026-06-20: Established attention baseline: `TurboquantAttentionPaged4bit` avg 29.68 us.
- 2026-06-20: Accepted `lowMask` hoist. Validation profile `/root/x00827378/perflog2/rank0_3573370_20260620181130047_ascend_pt` has `TurboquantAttentionPaged4bit` count 84, avg 29.30 us, min 26.60 us, max 34.54 us. Smoke output for both prompts is normal.
- 2026-06-20: Rejected full 4-row group `DataCopyPad` for norm reads. Profile `/root/x00827378/perflog2/rank0_3598949_20260620183033441_ascend_pt` has `TurboquantAttentionPaged4bit` count 85, avg 29.53 us, min 26.86 us, max 50.20 us. Smoke output was normal, but performance regressed versus 29.30 us.
- 2026-06-20: Accepted removal of the vector pipe barrier after Q cast in `LoadQGroup`. Validation profile `/root/x00827378/perflog2/rank0_3612733_20260620184256189_ascend_pt` has `TurboquantAttentionPaged4bit` count 84, avg 28.78 us, min 27.30 us, max 30.76 us. Smoke output for both prompts is normal.
- 2026-06-20: Rejected removal of Vector-only barriers inside `LoadPackedTileRows` row unpack. Validation profile `/root/x00827378/perflog2/rank0_3627303_20260620185516023_ascend_pt` has `TurboquantAttentionPaged4bit` count 84, avg 28.88 us, min 27.10 us, max 31.44 us. Smoke output for both prompts was normal, but performance regressed versus 28.78 us, so the code change was reverted.
- 2026-06-20: Rejected skipping the `packed` half cast for index rows when the matmul path is active. Validation profile `/root/x00827378/perflog2/rank0_3641530_20260620190747660_ascend_pt` has `TurboquantAttentionPaged4bit` count 84, avg 29.80 us, min 27.50 us, max 32.38 us. Smoke output for both prompts was normal, but performance regressed versus 28.78 us, so the code change was reverted.
- 2026-06-20: Rejected host tiling candidate A, which removed the fixed 16-core cap and computed data-parallel MIX groups from physical AIC/AIV counts. Validation profile `/root/x00827378/perflog2/rank0_3657127_20260620192426900_ascend_pt` has `TurboquantAttentionPaged4bit` count 84, avg 30.81 us, min 28.24 us, max 33.86 us. Smoke output for both prompts was normal, but performance regressed versus 28.78 us, so the code change was reverted.
- 2026-06-20: Rejected removing per-GQA-head vector pipe barriers between `Muls` and `Cast` in `WriteFinalOutput`. Validation profile `/root/x00827378/perflog2/rank0_3671116_20260620193546910_ascend_pt` has `TurboquantAttentionPaged4bit` count 84, avg 29.82 us, min 27.46 us, max 32.76 us. Smoke output for both prompts was normal, but performance regressed versus 28.78 us, so the code change was reverted.
- 2026-06-20: Clean rebuild after rejecting Candidate B confirmed the current accepted baseline. Validation profile `/root/x00827378/perflog2/rank0_3685298_20260620194734853_ascend_pt` has `TurboquantAttentionPaged4bit` count 84, avg 29.22 us, min 27.00 us, max 32.38 us. Smoke output for both prompts was normal.
- 2026-06-20: Accepted Candidate C, which reorders `VectorQk` m-major so each K row is cast once and reused across GQA heads. Validation profile `/root/x00827378/perflog2/rank0_3699532_20260620200008990_ascend_pt` has `TurboquantAttentionPaged4bit` count 84, avg 28.74 us, min 27.16 us, max 31.10 us. Smoke output for both prompts was normal.
- 2026-06-20: Accepted Candidate D, which splits softmax state update from PV and runs PV m-major so each V row is cast once and reused across GQA heads. Validation profile `/root/x00827378/perflog2/rank0_3714146_20260620201419192_ascend_pt` has `TurboquantAttentionPaged4bit` count 84, avg 28.25 us, min 26.46 us, max 30.12 us. Smoke output for both prompts was normal.
- 2026-06-20: Reverted Candidate E before accepting it. The only attempted change was removing the vector barrier after `outAcc *= alpha`; no valid smoke profile was produced because the device was stuck at HBM 60240/65536 MB and AICore 100% with no running process listed by `npu-smi`.
- 2026-06-20: Confirmed the smoke script overrides `source xrx_infoenvs` defaults from `ENCODE_OP=1, DECODE_OP=0` to `ENCODE_OP=0, DECODE_OP=1` before importing vLLM. This is required because fused 4-bit pack is still a separate correctness issue.
- 2026-06-20: Smoke/profile before the clean rebuild produced normal two-prompt output and restored the `TurboquantAttentionPaged4bit` profile entry. Profile `/root/x00827378/perflog2/rank0_3733860_20260620203745523_ascend_pt` has `TurboquantAttentionPaged4bit` count 84, avg 28.68 us, min 27.26 us, max 30.42 us. This run was used only to verify path correctness because the source had just reverted Candidate E and the kernel binary had not yet been rebuilt.
- 2026-06-20: Completed the required clean `build_ext` after reverting Candidate E. `tests/ut/ops/test_turboquant_attention_paged4bit_op.py` passed (`4 passed`). The following smoke/profile attempt was blocked by device state before generation: vLLM reported only 4.56 GiB free versus the 6.1 GiB requested, and `npu-smi` then showed HBM 60793/65536 MB and AICore 100% with no running process.
- 2026-06-20: Rechecked the user-provided bad profile `/root/x00827378/perflog2/rank0_3463944_20260620155619907_ascend_pt`. It has `TurboquantPackKvForCache4bit` count 84, avg 32.10 us, but `TurboquantAttentionPaged4bit` count 0. This confirms that profile did not exercise the 4-bit attention kernel and must not be used for attention performance decisions.
- 2026-06-20: Reconfirmed the pack rollback state: `TurboquantPackKVForCache4bitToCache::GetActiveWorkers()` returns 1, so the temporary physical-cache-group worker split is not active.
- 2026-06-20: Ran the required clean rebuild command again after reverting Candidate E: `source xrx_infoenvs && rm -rf build/temp.linux-aarch64-cpython-311 csrc/build && MAX_JOBS=32 COMPILE_CUSTOM_KERNELS=1 python setup.py build_ext --inplace`. Build completed successfully and regenerated `TurboquantAttentionPaged4bit_559403fcfdd98c9fcdd9ff2d96d4285d` and `TurboquantAttentionPaged4bit_5d4907a4205cb9daa18c855ed63ec34d`.
- 2026-06-20: Post-build UT passed again: `source xrx_infoenvs && pytest -q tests/ut/ops/test_turboquant_attention_paged4bit_op.py -s` reported `4 passed`.
- 2026-06-20: Smoke/profile is currently blocked by device state, not by source changes. `npu-smi info` repeatedly reports HBM `60558/65536 MB` and AICore `100%` with `No running processes found in NPU 7`, so vLLM cannot reserve the required smoke memory. Do not add or accept further attention optimization candidates until a clean smoke/profile can be produced from the current baseline.
