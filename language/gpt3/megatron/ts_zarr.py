import tensorstore as ts
import zarr
import numpy as np
import torch

import os
import sys
sys.path.append(os.environ["MEGATRON_PATH"])

from megatron.arguments import core_transformer_config_from_args
from megatron import get_args
from megatron import print_rank_0
from megatron.training import get_model
from megatron.model import GPTModel
from megatron.initialize import initialize_megatron

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--use-beam-search", action = "store_true")
    return parser

initialize_megatron(
    extra_args_provider=add_text_generate_args,
    args_defaults={
        "tokenizer_type": "SentencePieceTokenizer",
        "no_load_rng": True,
        "no_load_optim": True,
    }
)

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0("building GPT model ...")
    config = core_transformer_config_from_args(get_args())
    config.init_method =None
    config.gradient_accumulation_fusion = False
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process,
    )
    model = model.eval()
    return model

def open_with_ts(layer_dir):
    spec = {'driver': 'zarr',
            'metadata_key': '.zarray',
            'kvstore': {'driver': 'file', 'path': layer_dir}}
    return ts.open(ts.Spec(spec), open=True).result().read().result()

def open_with_zarr(layer_dir):
    return zarr.open(layer_dir)[:]

model = get_model(model_provider, wrap_with_ddp=False)
model = model[0]

def replace_module(module, name_prefix = ""):
    for name, child in module.named_children():
        if len(name_prefix) == 0:
            name = name
        else:
            name = name_prefix + "." + name
        replace_module(child, name)
        if hasattr(child, 'weight'):
            print(name, ":", child.__class__, child.weight.shape)
        elif hasattr(child, 'bias'):
            print(name, ":", child.__class__, child.bias.shape)

def replace_module_2(module):
    for key, weight in module.named_children():
        if isinstance(weight, str):
            continue

        weight_size = weight.numel() * 2
        print(key, ":", weight_size)

replace_module_2(model)

# model_to_save = PreTrainedModel(model)
# model_to_save.save_pretrained('./test_model', max_shard_size = "400M")

