PAXML_CKPT_PATH="$PWD/model"
EXTERNAL_MODEL_CHECKPOINT_DIR="$PWD/converted_checkpoint"

python convert_paxml_to_megatron_distributed.py -gckpt $PAXML_CKPT_PATH -o $EXTERNAL_MODEL_CHECKPOINT_DIR --dtype bf16 -p 1
