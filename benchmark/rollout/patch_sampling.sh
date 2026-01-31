

VLLM_PATH="/home/yak/myenv_v14/lib/python3.12/site-packages/vllm"

patch $VLLM_PATH/sampling_params.py < sampling_params.patch
patch $VLLM_PATH/v1/engine/parallel_sampling.py < parallel_sampling.patch
