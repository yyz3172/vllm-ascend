# P18 spike — MTE2⇄VEC cross-run overlap

## Change

- Meta UB ping-pong: `dequantInt8Buf_` 2×768B (`BR_DEQUANT_UB_BYTES_PINGPONG`)
- After `SetFlag(V_MTE3)`, before `WaitFlag(V_MTE3)`: issue next-run codes+meta MTE2 into the other half/slot
- Drop mid-run `V_MTE2` gate; codes+meta one MTE2 burst per issue
- MTE2_V event IDs ping-pong (`eventIdCodesWait[2]`)

## Correctness

- UT `test_bit_residual_fia_dequant.py`: **9 passed**
- FIA Prefill/FD smoke: **all passed** (incl. bf16 / FD)

## Fair L6 A/B (same flags)

`msprof --kv=2000 --device=0 --source --warm-up=5 --launch-count=20`

| | A P17b-B HEAD | **B P18 spike** | Δ |
|--|--|--|--|
| Task Duration (µs) | 721.854 | **627.753** | **−94.1 (−13.0%)** |
| AIV vec_ratio | 0.517 | 0.591 | ↑ |
| AIV mte2_ratio | 0.304 | 0.223 | ↓ (hidden in wall) |
| AIV mte3_ratio | 0.128 | 0.151 | |
| Vec FOPS (avg) | 11.210M | 11.219M | ≈ flat |

- A: `OPPROF_20260723132658_LLFUIFGSLHQZOPYU`
- B: `OPPROF_20260723132815_UBYYNXXPMHZAVDDZ`
- Logs: `/root/l00856060/perflog2/p18_mte2_vec_ab_20260723_132612`

## Verdict

**Keep.** Wall −13% clears the ≥3% go gate; pipe ratios match successful MTE2∥VEC overlap (mte2 fraction ↓, vec ↑, FOPS unchanged).

