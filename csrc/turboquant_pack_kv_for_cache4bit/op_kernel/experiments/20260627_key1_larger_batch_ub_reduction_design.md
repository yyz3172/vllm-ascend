# Experiment Design: key1 UB Reduction For Larger Full-Group Batches

Date: 2026-06-27

## Hypothesis

Batch-count attribution showed that the long hot shape performs 470 key/value
`ComputeBatch` and rotate calls because the current full-group path covers
only 8 token rows per batch:

```text
TQ_MAX_BATCH_M = 64 vector rows
numHeads = 8
rowsPerBatch = 64 / 8 = 8 token rows
```

If key1 can process 16 token rows per full-group batch, the same hot shape
would reduce key/value compute calls from 470 to 236. This is a structural
target large enough to matter.

The blocker is UB usage. A constant-only increase from M=64 to M=128 would
push approximate UB from 180 KiB to 330 KiB. Therefore the experiment must
first reduce live key1 UB allocation before increasing effective batch size.

## Scope

Target only large contiguous full-group key1. Keep key0 and partial groups on
the existing route-only implementation.

Do not retry:

- shared-entry template flag that instantiates the full class twice in one
  kernel body;
- direct packed encode/copyout without increasing batch size;
- constant-only `TQ_MAX_BATCH_M` increase.

## Candidate Design

Use a truly separate key1 implementation shape if possible. The key1 path must
not instantiate another full pack class body inside the same kernel entry used
by key0. If the build system cannot provide separate key1 codegen, stop before
runtime implementation and report the blocker.

Within that key1-only implementation:

1. Reduce or remove `encodedBatchQue_` for physical full-group runs.
   - Encode one full 4-row group per head into `packedRowBuf_`.
   - Copy packed groups out directly.
   - Do not allocate `encodedBatchQue_` for this path.
2. Use the freed UB to raise the key1 full-group batch capacity target from
   64 vector rows to 128 vector rows.
   - For 8 heads, this gives 16 token rows per batch.
3. Preserve existing `NormalizeBatch`, `RotateBatchMatmul`, and quantization
   math in the first implementation.
4. Preserve partial-group and indexed fallback paths by keeping them on the
   route-only implementation.

## UB Budget Target

Approximate current M=64 usage:

| Buffer group | Bytes |
| --- | ---: |
| `x/a/y` queues, depth 2 | 98,304 |
| `encodedBatchQue_`, depth 2 | 36,864 |
| `rotateWorkBuf_` | 16,384 |
| norms | 2,048 |
| packed groups and scalar/vector work buffers | ~30,752 |
| total | ~184,352 |

Naive M=128 is ~337,952 bytes.

To make M=128 plausible, the key1 full-group path needs one or more of:

- remove `encodedBatchQue_` depth-2 allocation: save 73,728 bytes at M=128;
- reduce queue depth for full-group key1 if overlap is not being used:
  each of `x/a/y` depth reduction saves 32,768 bytes at M=128;
- reduce `packedRowBuf_` slot count if direct streaming can wait on writes
  earlier.

The minimum viable target is to get M=128 under the practical UB limit while
keeping enough buffers for `Normalize -> Rotate -> Encode`.

## Semantic Requirements

- Cache group layout is unchanged:
  - row0 index bits 0..3, row1 4..7, row2 8..11, row3 12..15;
  - norm words appended in row order.
- Norm bits match existing `EncodeBatch` behavior.
- Group boundary ownership remains unchanged:
  begin inside a group backs up to process the complete group; non-last end
  inside a group rolls back to the group start.
- Smoke output must match baseline exactly, including no new repetition.

## Validation Gates

1. Before coding, verify whether a truly separate key1 codegen path is
   available. If not, write a blocker report and do not change runtime.
2. Build with `tools/build_debug_perf.sh`.
3. Run smoke with correct profile variables:

```bash
XRX_TQ4BIT_PROFILE=1 \
TQ_SMOKE_MODEL_PATH=/root/x00827378/model/Qwen3-0.6B \
TQ_SMOKE_PROFILE_DIR=<dir> \
python tests/e2e/singlecard/xrx_turboquant4bit_smoke.py
```

4. Small shape must not regress from avg 46.59 us.
5. If smoke passes, run fixed long query. Hot shape must improve from avg
   820.39 us.
6. If long improves, run `tools/op.profile.sh` and compare with refreshed
   route-only `557.89 us / 287.50 us`.

## Reflection Before Coding

This is the first follow-up that directly attacks the measured batch-count
problem. It is also riskier than source-level micro-optimizations. The
experiment should not start by writing a large runtime patch unless key1 can
be codegen-isolated from key0. Without that isolation, the previous smoke
regression shows we are likely to reject the patch before learning anything
about long-shape performance.
