unset ASCEND_GLOBAL_LOG_LEVEL
unset ASCEND_SLOG_PRINT_TO_STDOUT
#export KENEL_DEBUG_LINE=on
export COMPILE_CUSTOM_KERNELS=1
rm -rf build csrc/build csrc/output
python3 setup.py build_ext
