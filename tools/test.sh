#pytest tests/ut/attention/test_turboquant_functional.py >log.test 2>&1 | tail -f log.test
#msprof op pytest -v -s tests/ut/ops/test_turboquant_decode_paged_8bit_op.py::test_decode_paged_op_matches_pytorch_golden[0] > log.test 2>&1 | tail -f log.test
pytest -v -s tests/ut/ops/test_turboquant_decode_paged_8bit_op.py::test_decode_paged_op_matches_pytorch_golden[0]
