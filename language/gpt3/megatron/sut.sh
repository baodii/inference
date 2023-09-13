
# mode=$1
model=./model/
dataset=./data/cnn_eval.json

ipdb3 main.py \
    --scenario=Offline \
    --dataset-path=${dataset} \
    --max_examples=1 \
    --accuracy
