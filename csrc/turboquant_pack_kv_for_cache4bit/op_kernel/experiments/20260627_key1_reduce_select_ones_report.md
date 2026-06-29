# Experiment Report: key1 Reduce Select Persistent Ones

Date: 2026-06-27

## Summary

Rejected and reverted.

The persistent-ones table preserved smoke output semantics, but it regressed
the guarded small shape from 46.59 us to 47.72 us average. Under the strict
performance gate, the runtime change cannot be kept and the long query/profile
gates were intentionally not run.

## Implementation Tested

The candidate increased `quantCodeBuf_` from one 2048-float table to two
tables:

- table 0: quant thresholds used by the reduce-sum encoder.
- table 1: persistent 1.0f values used as the tensor source of `Select`.

`EncodeQuantCodesByReduceSum` then removed the per-row:

```cpp
Duplicate(quantOnes, 1.0f, TQ_QUANT_TABLE_ELEMS);
```

and selected from the persistent one table instead.

## Commands And Logs

Build:

```bash
bash tools/build_debug_perf.sh
```

Build log:

- `mytmp/build_key1_reduce_select_ones_20260627162551.log`

Smoke:

```bash
XRX_TQ4BIT_ENABLE_PROFILE=1 \
XRX_TQ4BIT_PROFILE_DIR=mytmp/perflog_smoke_key1_reduce_select_ones_20260627163014 \
python tests/e2e/singlecard/xrx_turboquant4bit_smoke.py
```

Smoke log/profile:

- `mytmp/tq4bit_smoke_key1_reduce_select_ones_20260627163014.log`
- `mytmp/perflog_smoke_key1_reduce_select_ones_20260627163014`

## Correctness Result

Smoke output matched the route-only baseline exactly:

```text
" Shin. I'm a 25-year-old girl who is interested in the topic of 'Environmental Science - the sciences of of the environment. I need to find the question: I need help!"
"\n\nTurboQuant 是一种专注于GPU加速的高性能计算应用平台 (HPC-P) 核的软件加速框架，它基于 k88000 的0 和 NVIDIA 的 n"
```

The known baseline `"of of"` duplicate remained unchanged. The candidate did
not introduce new repetition or semantic drift.

## Performance Result

Guarded smoke shape:

```text
"2,8,128;2,8,128;16;128,128;2;3"
```

| Run | Count | Avg us | P50 us | P90 us | Max us |
| --- | ---: | ---: | ---: | ---: | ---: |
| route-only baseline | 56 | 46.59 | 46.44 | 47.86 | 48.72 |
| persistent ones | 56 | 47.72 | 47.52 | 48.72 | 50.76 |

The average regressed by 1.13 us, about 2.4%. P50 and max also worsened. This
fails the smoke latency gate, so the experiment stopped before long query and
`tools/op.profile.sh`.

## Analysis

The hypothesis targeted a real hot source line: the per-row 2048-float
`Duplicate` inside `EncodeQuantCodesByReduceSum`. However, the implementation
paid for the removal by increasing UB pressure and by changing the `Select`
source from the destination buffer to a second 2048-float table. Even though
the smoke workload does not dynamically take the reduce-sum branch for
`m <= 16`, the larger buffer and changed generated code layout were enough to
hurt the guarded small path.

This is also not a strong enough large-shape idea to justify isolating the
small regression with a separate key1 class. It removes one instruction group,
but it does not change the dominant algorithmic structure:

- `Brcb` still expands each row to 2048 floats.
- `Compare` still touches the full threshold table.
- `Select` still produces a full 2048-float mask table.
- `WholeReduceSum` still reduces the same expanded table.

The source-profile goal was to find a breakthrough in the reduce body. This
variant is only a source shuffle, and the measured smoke regression shows the
cost model is not favorable.

## Reflection

This implementation is the best reasonable version of the persistent-ones
idea because it uses one persistent table, fills it once during threshold table
preparation, and avoids any semantic change. Further variations such as moving
the one table to another UB buffer or making it key1-only could hide the small
regression but would not address the underlying issue that the reduce body
still materializes and reduces a 128 x 16 table per row.

Do not retry this exact persistent-one-table approach. The next useful work is
to quantify KFC/matmul setup overhead and then look for a larger structural
change, not another local `Select` source adjustment.

## Decision

Runtime code was reverted. The design and this report are kept in git history
as a failed experiment record.
