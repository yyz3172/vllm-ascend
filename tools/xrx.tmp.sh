cd csrc
rm -rf build output 
bash build.sh -n bit_residual_pack_k8v4 -c ascend910b
cd .. 
./csrc/output/CANN-custom_ops--linux.aarch64.run --install-path=/root/x00827378/vllm-ascend/vllm_ascend/_cann_ops_custom 
python tests/e2e/singlecard/xrx_bit_residual_k8v4_key1_unaligned.py
