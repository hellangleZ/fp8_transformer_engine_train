import os
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
import quickstart_utils as utils

def init_distributed_mode():
    global rank, world_size
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f'RANK and WORLD_SIZE in env: {rank}/{world_size}')
    else:
        print('Not using distributed mode')
        return False

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'  # 确保端口设置正确

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(rank)
    dist.barrier()
    print(f'Process {rank} initialized and barrier passed')
    return True

if init_distributed_mode():
    hidden_size = 4096
    sequence_length = 2048
    batch_size = 4
    ffn_hidden_size = 16384
    num_attention_heads = 32
    dtype = torch.float16

    print(f'Process {rank} creating synthetic data')
    # Synthetic data
    x = torch.rand(sequence_length, batch_size, hidden_size).cuda().to(dtype=dtype)
    dy = torch.rand(sequence_length, batch_size, hidden_size).cuda().to(dtype=dtype)
    print(f'Process {rank} synthetic data created')

    # Assuming you have 8 GPUs
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    print(f'Process {rank} creating parallel groups')
    data_parallel_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    tensor_parallel_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    print(f'Process {rank} parallel groups created')
    dist.barrier()
    print(f'Process {rank} passed parallel groups barrier')

    # Construct layer
    print(f'Process {rank} constructing transformer layer')
    parallel_transformer = te.TransformerLayer(
        hidden_size,
        ffn_hidden_size,
        num_attention_heads,
        set_parallel_mode=True,
        tp_group=tensor_parallel_group,
        sequence_parallel=True,
    )
    parallel_transformer.to(dtype=dtype).cuda()
    parallel_transformer = torch.nn.parallel.DistributedDataParallel(
        parallel_transformer,
        process_group=data_parallel_group,
    )
    print(f'Process {rank} transformer layer constructed')
    dist.barrier()
    print(f'Process {rank} passed transformer layer barrier')

    # fp8
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(
        fp8_format=fp8_format,
        amax_history_len=16,
        amax_compute_algo="max",
    )

    # Training step
    print(f'Process {rank} starting training step')
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, fp8_group=data_parallel_group):
        y = parallel_transformer(x, attention_mask=None)
    y.backward(dy)
    print(f'Process {rank} training step completed')
    dist.barrier()
    print(f'Process {rank} passed training step barrier')

    # Measure step time
    print(f'Process {rank} measuring step time')
    utils.speedometer(
        parallel_transformer,
        x,
        dy,
        forward_kwargs={"attention_mask": None},
        fp8_autocast_kwargs={
            "enabled": True,
            "fp8_recipe": fp8_recipe,
            "fp8_group": data_parallel_group,
        },
    )
    print(f'Process {rank} completed')
else:
    print("Distributed mode not initialized")