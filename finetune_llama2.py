import torch.distributed as dist
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments as TrainArgs, HfArgumentParser as HfArgs
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from tensor_parallel import get_tensor_sharded_model
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from data_utils import (DataCollatorForSupervisedDataset, 
                         setup_distributed_dataloader,
                         all_reduce_mean,
                         tokenize_batch_for_finetune,
                         tokenize_batch_for_finetune_tp,
                         jload)
import os

def setup():
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(world_size=world_size, rank=rank,
                            init_method="env://", backend="nccl")
    torch.cuda.set_device(local_rank)
    device_id = torch.cuda.current_device()
    print("Process running on GPU:", device_id)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(TrainArgs):
    cache_dir: Optional[str] = field(default=None)
    optimizer: str = field(default="adam") 
    auto_cast: bool = True
    lazy_init: bool = True 
    lr_scheduler: str = field(default="cosine") 
    learning_rate: float = field(default=2e-5)
    adam_beta1: float = field(default=0.9) 
    adam_beta2: float = field(default=0.95)
    weight_decay: float = field(default=0.1)
    num_train_epochs: int = field(default=1) 
    per_device_train_batch_size: int = field(default=1)
    tp: int = field(default=4)
    use_ddp: bool = True
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    tensorboard_path: str = field(default="tensorboard")  

def print_rank0(msg):
    rank = dist.get_rank()
    if rank == 0:
        print(msg)

def get_model_size(model: torch.nn.Module):
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel

def get_tflops(grad_checkpoint, model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * (8 if grad_checkpoint else 6) / 1e12 / (step_time + 1e-12)

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def train():
    parser = HfArgs((ModelArguments, DataArguments, TrainingArguments))
    setup()
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device=torch.cuda.current_device()
    model.to(device)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizer.pad_token = tokenizer.unk_token
    train_dataset = jload(data_args.data_path)
    data_loader = setup_distributed_dataloader(train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=partial(tokenize_batch_for_finetune, tokenizer=tokenizer, max_length=training_args.model_max_length),
    )

    optimizer = torch.optim.Adam(model.parameters(), training_args.learning_rate, 
                                    [training_args.adam_beta1, training_args.adam_beta2],
                                    weight_decay=training_args.weight_decay)
    
    if training_args.lr_scheduler== "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                         T_0=training_args.num_train_epochs, 
                                                                         T_mult=3, 
                                                                         eta_min=0.1 * training_args.learning_rate)
    elif training_args.lr_scheduler == "linear": 
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                      start_factor=0.5,
                                                      total_iters=training_args.num_train_epochs * len(data_loader))

    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    if training_args.use_ddp:
        model = model.to(device_id)
        model = DDP(model, device_ids=[device_id], find_unused_parameters = True) 
    
    # TODO: TP now is imcompatiable with gradient_checkpoint, lazy_linear. 
    if training_args.tp > 1:
        get_tensor_sharded_model(model, training_args.tp) 
        dataset = []
        for data in train_dataset: 
            dataset.append(tokenize_batch_for_finetune_tp(data, tokenizer, training_args.model_max_length))   
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, max_length=training_args.model_max_length)
        data_loader = setup_distributed_dataloader(
            dataset=dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=data_collator,
            use_tp=True
        )

    # training
    writer = SummaryWriter(training_args.tensorboard_path)
    epoch = 0
    step = 0

    num_steps_per_epoch = len(data_loader)
    train_epoches  = training_args.num_train_epochs

    for _epoch in range(epoch, training_args.num_train_epochs):
        for batch in data_loader:
            step += 1
            torch.cuda.synchronize()
            start_time = time.time()
            batch = to_device(batch, device)
            if training_args.auto_cast:
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                with torch.autocast(device_type=device_type):
                    outputs = model(**batch)
                    loss = outputs[0]
            else:
                outputs = model(**batch)
                loss = outputs[0]
            if training_args.use_ddp:
                loss.to(dist.get_rank())

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            end_time = time.time()
            step_time = end_time - start_time

            tflops = get_tflops(training_args.gradient_checkpointing,
                                get_model_size(model), 
                                training_args.per_device_train_batch_size, 
                                training_args.model_max_length,
                                step_time)
            
            all_reduce_mean(loss) 
            print_rank0(f"Epoch:{_epoch} " +
                        f"step_{step} / total_steps_{train_epoches * num_steps_per_epoch} " +
                        f"loss: {loss.item()} " +
                        f"learning_rate: {scheduler.get_lr()[0]} " +
                        f"step_time: {step_time:.2f}s " +
                        f"current_allocated_mem: {(torch.cuda.memory_allocated() / 1024**2):.2f}MB " +
                        f"max_allocated_mem: {(torch.cuda.max_memory_allocated() / 1024**2):.2f}MB " +
                        f"tflops: {tflops:.2f}")
            writer.add_scalar("loss", loss.item(),step)
            torch.cuda.empty_cache()
    
    model.eval()
    writer.close()

if __name__ == "__main__":
    train()