cd csrc
rm -rf build output
bash build.sh -n "bit_residual_pack_k8v4;bit_residual_attention_paged_k8v4" -c ascend910b
cd ..
./csrc/output/CANN-custom_ops--linux.aarch64.run --install-path=/root/x00827378/vllm-ascend/vllm_ascend/_cann_ops_custom

# Runtime env: the custom opapi symbols (aclnnBitResidualAttentionPagedK8v4)
# live in libcust_opapi.so under the installed vendor dir, which is only on
# the loader path via the vendor's set_env.bash. Without it the decode op
# reports "not in libopapi.so".
source /root/x00827378/env.ascend.sh
source /root/x00827378/vllm-ascend/xrx_infoenvs
source /root/x00827378/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/bin/set_env.bash

python tests/e2e/singlecard/xrx_bit_residual_k8v4_key1_unaligned.py
python tests/e2e/singlecard/xrx_bit_residual_k8v4_golden.py
python tests/e2e/singlecard/xrx_bit_residual_k8v4_batch_tiers.py
