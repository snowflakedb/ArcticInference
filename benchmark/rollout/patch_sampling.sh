
VLLM_PATH="$(pip show vllm | awk '/^Location: /{print $2}')"

if [ -z "$VLLM_PATH" ]; then
  echo "Error: could not find VLLM in current env"
  exit 1
else
  echo "VLLM path is: $VLLM_PATH"
fi

patch $VLLM_PATH/vllm/sampling_params.py < sampling_params.patch
patch $VLLM_PATH/vllm/v1/engine/parallel_sampling.py < parallel_sampling.patch
