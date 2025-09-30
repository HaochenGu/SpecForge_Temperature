SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# train eagle3 for llama3.1-8b
NUM_GPUS=${1:-8}

# space-separated list or pass via TEMPERATURES env var
TEMPERATURES=${TEMPERATURES:-"0.5 0.7 1.0 1.3 1.5 2.0"}

for T in $TEMPERATURES; do
OUTPUT_DIR="$ROOT_DIR/outputs/llama3-8b-eagle3/temp-${T}"
mkdir -p "$OUTPUT_DIR"
torchrun \
	--standalone \
	--nproc_per_node $NUM_GPUS \
	$ROOT_DIR/scripts/train_eagle3_online.py \
	--target-model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
	--draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
	--train-data-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
	--output-dir "$OUTPUT_DIR" \
	--num-epochs 5 \
	--batch-size 4 \
	--learning-rate 1e-4 \
	--max-length 2048 \
	--chat-template llama3 \
	--cache-dir $ROOT_DIR/cache \
	--attention-backend flex_attention \
	--train-temperature $T
done
