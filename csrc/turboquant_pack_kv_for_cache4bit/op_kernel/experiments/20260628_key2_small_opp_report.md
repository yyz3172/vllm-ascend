# Key2 Small-Shape OPP Report

Date: 2026-06-28

## Scope

Profile the current `tilingKey = 2` implementation without changing code.

The profiled path is the current AIV-only small-vector route:

- `TILING_KEY_IS(2)`
- no `REGIST_MATMUL_OBJ_STATIC`
- `RotateBatchVectorSmall()`
- existing 4-bit encode and cache CopyOut

## Command

```bash
source xrx_infoenvs
export ASCEND_CUSTOM_OPP_PATH=/root/x00827378/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend
export TQ4BIT_OP_PROFILE_APP="mytmp/flex_tq_4bit_perf/flex_tq_4bit_perf --pack-only --q-lens 2 --kv-lens 2 --pack-tokens 2 --heads 16 --kv-heads 8 --block-size 128 --warmup 1 --repeat 3"
export TQ4BIT_OP_PROFILE_METRICS=Source,PipeUtilization
export TQ4BIT_OP_PROFILE_LAUNCH_COUNT=1
export TQ4BIT_OP_PROFILE_WARM_UP=0
bash tools/op.profile.sh
```

OPP output:

- `mytmp/OPPROF_20260628073932_HAMRWXWPXODRZODT`
- kernel: `TurboquantPackKvForCache4bit_cf5c5d6d4ad385ca5cf1f7b534e9a391_2`
- debug source: ok

The pre-profile plain runner check reported:

- `pack4bit_to_cache_avg_us=143.66`

The OPP run reported a larger runner average:

- `pack4bit_to_cache_avg_us=374.37`

Use the OPP run for attribution only. Profiling overhead changes the runner
average.

## OPP Summary

`OpBasicInfo.csv`:

- op type: `vector`
- task duration: `133.000us`
- block dim: `20`

`PipeUtilization.csv`:

- block 1 is the real worker:
  - `aiv_time_us=132.373337`
  - `aiv_vec_time_us=101.386665`
  - `aiv_scalar_time_us=47.920555`
  - `aiv_mte2_time_us=8.188334`
  - `aiv_mte3_time_us=2.950556`
- other blocks are mostly 1-4us setup/early-return work.

This means the regression is not primarily from every block doing full work.
Most time is on one active worker.

## Source Hotspots

Function-level extraction:

```bash
python tools/extract_opprof_source_lines.py \
  mytmp/OPPROF_20260628073932_HAMRWXWPXODRZODT \
  --source turboquant_pack_kv_for_cache4bit.cpp \
  --group-by function --top 80 --hot-lines 12
```

Saved outputs:

- `mytmp/opprof_key2_small_function_top80_20260628073932.txt`
- `mytmp/opprof_key2_small_allsource_function_top100_20260628073932.txt`
- `mytmp/opprof_key2_small_compute_lines_20260628073932.txt`
- `mytmp/opprof_key2_small_rotate_lines_20260628073932.txt`
- `mytmp/opprof_key2_small_copyout_lines_20260628073932.txt`
- `mytmp/opprof_key2_small_flush_lines_20260628073932.txt`

Important source-attributed counts:

- `ComputeBatch`: `110,393`
  - line 1022, `RotateBatchVectorSmall(m)`: `95,216`
  - line 1026, `EncodeBatch(m)`: `12,300`
  - line 1020, `NormalizeBatch(m)`: `2,877`
- `RotateBatchVectorSmall`: `91,049`
- `EncodeBatch`: `12,238`
- `EncodeQuantCodesByCompare`: `9,696`
- `CopyOutResolvedTask`: `3,898`
- `CopyOutPhysicalGroupRowsFast`: `3,647`
- `NormalizeBatchScalarScale`: `2,842`

The high `FlushSequenceBatch`/`FlushVecBatch` lines are parent call-line
attribution:

- `FlushSequenceBatch` line 1605: `114,815`
- `FlushVecBatch` line 1595: `57,450`
- `FlushVecBatch` line 1596: `57,358`

Those lines call key and value pack tasks. They should not be read as an
independent flush bottleneck.

## Rotate Detail

Line-level extraction for `RotateBatchVectorSmall()` shows the hot region:

- line 634, row loop before `MulAddDst`: `20,480`
- line 635, `MulAddDst`: `24,576`
- line 636, destination row tensor: `12,288`
- line 638, coefficient block operand: `16,384`
- line 628, cast rotation row from local tile to fp32: `4,096`
- line 618/619, per-row `Brcb` setup: `3,072` each

All-source extraction also attributes cost to AscendC internals:

- `MulAddDst`: `20,480`
- `MulAddDstImpl`: `12,288`
- `MulAddDstIntrinsicsImpl`: `4,096`
- `Brcb`/`BrcbImpl`: `4,096`

## Interpretation

The current key2 implementation is dominated by the vector rotate matmul.

The 4-bit encode and cache CopyOut are not the primary reason this path is
slow for the tiny shape. In this OPP run, encode is about one eighth of the
source-attributed rotate count, and CopyOut is smaller.

The current vector rotate still performs a large nested loop:

- 16 `kBase` tiles
- 8 `kInner` rows per tile
- `m` row loop inside each `kInner`
- repeated `Cast` of rotation rows and `MulAddDst` accumulation

For this shape the path avoids KFC setup, but replaces Cube rotate with enough
vector work and barriers that it remains much slower than the fixed baseline
small-shape timing.

## Decision

Do not optimize encode or CopyOut first for key2. The next key2 experiment, if
any, must reduce the rotate body itself.

The most relevant v3 lesson is not the complete v3 op shape, because v3 is
8-bit pack and has a different output format. The reusable idea is only the
static-table strategy: avoid repeated GM rotation-tile loads if it can be done
without increasing inactive-worker overhead or UB pressure.
