import os
import sys

sys.path.append(os.environ["MEGATRON_PATH"])

import datetime
import torch
import json
import threading
from flask import Flask, request, jsonify, current_app
from flask_restful import Resource, Api
from megatron import get_args
from megatron.text_generation.generation import (
    generate_tokens_probs_and_return_on_first_stage,
    beam_search_and_return_on_first_stage
)


from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.arguments import core_transformer_config_from_args
from megatron.model import GPTModel
from megatron.training import get_model
# from megatron.core.dist_checkpointing import load as megatron_load
import torch
import deepspeed


GENERATE_NUM = 0
BEAM_NUM = 1
lock = threading.Lock()

def mapping_impi_to_torch():
    os.environ['LOCAL_RANK'] = os.environ['MPI_LOCALRANKID']
    os.environ['LOCAL_SIZE'] = os.environ['MPI_LOCALNRANKS']
    os.environ['RANK'] = os.environ['PMI_RANK']
    os.environ['WORLD_SIZE'] = os.environ['PMI_SIZE']
    os.environ['CROSS_RANK'] = '0'
    os.environ['CROSS_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '44434'

if 'MPI_LOCALRANKID' in os.environ: # MPI launcher
    mapping_impi_to_torch()


class MegatronGenerate(Resource):
    def __init__(self, model, gen_kwargs, log=None):
        self.model = model
        self.log = log
        self.gen_kwargs = gen_kwargs
        self.use_beam_search = self.gen_kwargs.get("use_beam_search", None)

    @staticmethod
    def send_do_generate():
        choice = torch.xpu.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(choice, 0)

    @staticmethod
    def send_do_beam_search():
        choice = torch.xpu.LongTensor([BEAM_NUM])
        torch.distributed.broadcast(choice, 0)

    @staticmethod
    def sync_input(input_ids, input_length):
        input_length_tensor = torch.xpu.LongTensor(input_length)
        torch.distributed.broadcast(input_length_tensor, 0)
        input_ids_tensor = torch.xpu.LongTensor(input_ids)
        torch.distributed.broadcast(input_ids_tensor, 0)
        return input_ids_tensor, input_length_tensor

    def put(self):
        args = get_args()
        # import pdb
        # pdb.set_trace()
        # print('---------------------megatron generate--------------------------------------')
        import os
        print('---------------------------------server pid: -------------------------------------------', os.getpid())
        if not "input_ids" in request.get_json():
            return "input_ids argument required", 400

        if not "input_length" in request.get_json():
            return "input_length is required", 400

        input_ids = request.get_json()["input_ids"]
        input_length = request.get_json()["input_length"]

        with lock:  # Need to get lock to keep multiple threads from hitting code

            if self.log:
                print("request IP: " + str(request.remote_addr))
                print("start time: ", datetime.datetime.now())

            try:
                if self.use_beam_search:
                    try:
                        MegatronGenerate.send_do_beam_search()  # Tell other ranks we're doing beam_search
                        input_ids_tensor, input_length_tensor = MegatronGenerate.sync_input(
                            input_ids, input_length
                        )
                        (
                            output_tokens,
                            _,
                        ) = beam_search_and_return_on_first_stage(
                            self.model,
                            input_ids_tensor,
                            input_length_tensor,
                            beam_size=self.gen_kwargs.get("beam_size", 4),
                            stop_token = self.gen_kwargs.get("beam_stop_token", 1),
                            num_return_gen = self.gen_kwargs.get("beam_num_return_gen", 1),
                            length_penalty = self.gen_kwargs.get("beam_length_penalty", 1),
                            min_length = self.gen_kwargs.get("min_new_tokens", 30),
                        )
                        output_batch_truncated = []
                        for data, source_len in zip(output_tokens, input_length_tensor):
                            output_batch_truncated.append(
                                data[source_len:].cpu().numpy().tolist()
                            )
                        if self.log:
                            print("end time: ", datetime.datetime.now())
                        return jsonify({"output": output_batch_truncated})
                    except Exception as e:
                        print(str(e))
                        print("ERROR")
                        return jsonify({"output": [[]], "is_error": True})
                else:
                    # try:
                    MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
                    input_ids_tensor, input_length_tensor = MegatronGenerate.sync_input(
                        input_ids, input_length
                    )
                    input_ids_tensor = input_ids_tensor.to('xpu:{}'.format(torch.xpu.current_device()))
                    input_length_tensor = input_length_tensor.to('xpu:{}'.format(torch.xpu.current_device()))
                    (
                        output_tokens,
                        _,
                        _,
                    ) = generate_tokens_probs_and_return_on_first_stage(
                        self.model,
                        input_ids_tensor,
                        input_length_tensor,
                        top_k=self.gen_kwargs.get("top_k", 4),
                        temperature=self.gen_kwargs.get("temperature", 0.0),
                    )
                    output_batch_truncated = []
                    for data, source_len in zip(output_tokens, input_length_tensor):
                        output_batch_truncated.append(
                            data[source_len:].cpu().numpy().tolist()
                        )
                    if self.log:
                        print("end time: ", datetime.datetime.now())
                    return jsonify({"output": output_batch_truncated})
                    # except Exception as e:
                    #     print(str(e))
                    #     print("ERROR")
                    #     return jsonify({"output": [[]], "is_error": True})

            except ValueError as ve:
                return ve.args[0]


class MegatronServer(object):
    def __init__(self, model, gen_kwargs):
        self.app = Flask(__name__, static_url_path="")
        api = Api(self.app)
        api.add_resource(
            MegatronGenerate, "/api", resource_class_args=[model, gen_kwargs]
        )

    def run(self, url):
        self.app.run(url, threaded=True, debug=False)


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
    # model = model.eval()

    return model


def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--use-beam-search", action = "store_true")
    return parser

def ds_inference(model, args):
    import megatron.model as mm
    engine = deepspeed.init_inference(model=model,
                                      mp_size=args.tensor_model_parallel_size,
                                      tensor_parallel={"mpu": mpu},
                                      dtype=torch.half if args.fp16 == True else torch.bfloat16,
                                      replace_with_kernel_inject=True if args.fp16 == True else False,
                                      moe_experts=args.num_experts,
                                      moe_type=args.mlp_type)

    return engine.module

if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=add_text_generate_args,
        args_defaults={
            "tokenizer_type": "SentencePieceTokenizer",
            "no_load_rng": True,
            "no_load_optim": True,
        }
    )

    args = get_args()
    gen_kwargs = {
        "early_stopping": True,
        "max_new_tokens": 128,
        "min_new_tokens": 30,
        "top_k": 40,
        "temperature": 0.5,
        "use_beam_search": args.use_beam_search,
        "beam_size": 4,
        "beam_stop_token": 1,
        "beam_num_return_gen": 1,
        "beam_length_penalty": 1
    }
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)
    # print_rank_0(model[0])
    # exit(0)
    # if args.load is not None:
    #     _ = load_checkpoint(model, None, None)
        # static_dic = megatron_load(None, './model')


    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # model = ds_inference(model, args)

    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        import os
        print('-------------------------------------rank0 pid: ---------------------------------------', os.getpid())
        server = MegatronServer(model, gen_kwargs)
        server.run("127.0.0.1")
    else:
        while True:
            import os
            print('-------------------------------------rank1 pid: ---------------------------------------', os.getpid())
            choice = torch.xpu.LongTensor([GENERATE_NUM])
            input_length_tensor = torch.xpu.LongTensor(1)
            torch.distributed.broadcast(choice, 0)
            if choice[0].item() == 0:
                # Greedy or top-k
                try:
                    torch.distributed.broadcast(input_length_tensor, 0)
                    input_ids_tensor = torch.xpu.LongTensor(
                        [
                            [
                                0
                                for _ in range(
                                    input_length_tensor[0].item()
                                    + gen_kwargs.get("max_new_tokens")
                                )
                            ]
                        ]
                    )
                    torch.distributed.broadcast(input_ids_tensor, 0)
                    generate_tokens_probs_and_return_on_first_stage(
                        model,
                        input_ids_tensor,
                        input_length_tensor,
                        top_k=gen_kwargs.get("top_k", 4),
                        temperature=gen_kwargs.get("temperature", 1.0),
                    )
                except ValueError as ve:
                    pass
            elif choice[0].item() == 1:
                # Beam search
                try:
                    torch.distributed.broadcast(input_length_tensor, 0)
                    input_ids_tensor = torch.xpu.LongTensor(
                        [
                            [
                                0
                                for _ in range(
                                    input_length_tensor[0].item()
                                    + gen_kwargs.get("max_new_tokens")
                                )
                            ]
                        ]
                    )
                    torch.distributed.broadcast(input_ids_tensor, 0)
                    beam_search_and_return_on_first_stage(
                        model,
                        input_ids_tensor,
                        input_length_tensor,
                        beam_size=gen_kwargs.get("beam_size", 4),
                        stop_token = gen_kwargs.get("beam_stop_token", 1),
                        num_return_gen = gen_kwargs.get("beam_num_return_gen", 1),
                        length_penalty = gen_kwargs.get("beam_length_penalty", 1),
                        min_length = gen_kwargs.get("min_new_tokens", 30),
                    )
                except ValueError as ve:
                    pass
