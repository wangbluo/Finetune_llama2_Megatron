import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase

device = "cuda" if torch.cuda.is_available() else "cpu" 

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def test(data_loader, model, model_tp):
    # unit test for TP
    torch.cuda.empty_cache()
    _check_module(model, model_tp)
    testcase = TestCase()

    data_iter = iter(data_loader)
    batch = to_device(next(data_iter), device)
    output = model(**batch)
    output_tp = model_tp(**batch)
    testcase.assertEqual(output[0], output_tp[0])

    output[0].backward()
    output_tp[0].backward()
    torch.cuda.empty_cache()
    _check_module_bwd(model, model_tp, check_grad=True)
    optimizer = torch.optim.Adam(model.parameters(), 2e-5)
    optimizer_tp = torch.optim.Adam(model_tp.parameters(), 2e-5) 
    optimizer.step()
    optimizer_tp.step()
    torch.cuda.empty_cache()
    _check_module_bwd(model, model_tp, check_grad=True)
    batch = to_device(next(data_iter), device)
    output_1 = model(**batch)
    output_tp_1 = model_tp(**batch)
    testcase.assertEqual(output_1[0], output_tp_1[0])
    torch.cuda.empty_cache()
    _check_module_bwd(model, model_tp, check_grad=True)

def _check_module(m1, m2, check_grad=False):
        testcase = TestCase()
        named_parameters = dict(m1.named_parameters())
        for name, param_m2 in m2.named_parameters():
            testcase.assertTrue(name in named_parameters)
            param_m1 = named_parameters[name]
            with torch.no_grad():
                if "q_proj" in name or "k_proj" in name or "v_proj" in name or "gate_proj" in name or "up_proj" in name:
                #if "q_proj" in name or "k_proj" in name or "v_proj" in name:    
                    param_m2 = param_m2.contiguous() 
                    process_group = dist.group.WORLD
                    tensor_list = [torch.empty_like(param_m2) for _ in range(dist.get_world_size())]
                    rank = dist.get_rank()
                    tensor_list[rank] = param_m2      
                    dist.all_gather(tensor_list, param_m2, group=process_group)
                    param_m2 = torch.cat(tensor_list, dim = 0).contiguous()
                elif "o_proj" in name or "down_proj" in name:
                #elif "o_proj" in name: 
                    param_m2 = param_m2.contiguous()
                    process_group = dist.group.WORLD
                    tensor_list = [torch.empty_like(param_m2).contiguous() for _ in range(dist.get_world_size())]
                    rank = dist.get_rank()
                    tensor_list[rank] = param_m2    
                    dist.all_gather(tensor_list, param_m2, group=process_group)
                    param_m2 = torch.cat(tensor_list, dim = -1).contiguous() 
                    
            testcase.assertEqual(param_m2, param_m1)

def _check_module_bwd(m1, m2, check_grad=False):
    testcase = TestCase()
    named_parameters = dict(m1.named_parameters())
    for name, param_m2 in m2.named_parameters():
        testcase.assertTrue(name in named_parameters)
        param_m1 = named_parameters[name]
        if check_grad:
            param_m2 = param_m2.grad
            param_m1 = param_m1.grad
            if "q_proj" in name or "k_proj" in name or "v_proj" in name or "gate_proj" in name or "up_proj" in name:
            #if "q_proj" in name or "k_proj" in name or "v_proj" in name:      
                param_m2 = param_m2.contiguous() 
                process_group = dist.group.WORLD
                tensor_list = [torch.empty_like(param_m2) for _ in range(dist.get_world_size())]
                rank = dist.get_rank()
                tensor_list[rank] = param_m2     
                dist.all_gather(tensor_list, param_m2, group=process_group)
                param_m2 = torch.cat(tensor_list, dim = 0).contiguous()
            elif "o_proj" in name or "down_proj" in name:
            #elif "o_proj" in name: 
                param_m2 = param_m2.contiguous() 
                process_group = dist.group.WORLD
                tensor_list = [torch.empty_like(param_m2) for _ in range(dist.get_world_size())]
                rank = dist.get_rank()
                tensor_list[rank] = param_m2     
                dist.all_gather(tensor_list, param_m2, group=process_group)
                param_m2 = torch.cat(tensor_list, dim = -1).contiguous()
                
        testcase.assertEqual(param_m2, param_m1)

from transformers import LlamaForCausalLM, AutoTokenizer

instruction = """Write a code that optimizes a given mathematical equation using aHill Climb algorithm\nEquation: f(x, y) = (x - 3)^2 + (y - 3)^2"""
output = """Here's a python code implementation using Hill Climb algorithm:\n\n```python\nimport random\n\n# Define objective function\ndef f(x, y):\n    return (x - 3)**2 + (y - 3)**2\n\n# Define hill climbing function\ndef hill_climb(x_start, y_start, max_iters):\n    x_current = x_start\n    y_current = y_start\n\n    # Start iterations\n    for i in range(max_iters):\n        # Generate new candidate solutions\n        x_new = x_current + random.uniform(-1, 1)\n        y_new = y_current + random.uniform(-1, 1)\n\n     
   # If new solution is better, move to new solution\n        if f(x_new, y_new) < f(x_current, y_current):\n            x_current = x_new\n            y_current = y_new\n\n    
# Return the best solution \n    return x_current, y_current\n\n# Initialize parameters\nx_start = random.uniform(-10, 10)\ny_start = random.uniform(-10, 10)\nmax_iters = 10000\
n\n# Call the hill climbing function\nx_opt, y_opt = hill_climb(x_start, y_start, max_iters)\n\nprint(f\"The optimized value (x, y) is ({x_opt}, {y_opt})\")\n```\n\nThis algorit
hm starts with a random (x, y) point. Then, in each iteration, it generates a new point near the current one (here, randomly in the [-1, 1] range), and if this new point gives a
 lower value of the function f (i.e., it is closer to the (3, 3) point), then this new point is considered as the current one. This process continues for a specified number of i
terations ('max_iters'). At the end, the algorithm returns the best (x, y) found."""

model = LlamaForCausalLM.from_pretrained("/home/zhongyuting/model/Sheared-LLaMA-1.3B")
tokenizer = AutoTokenizer.from_pretrained("/home/zhongyuting/model/Sheared-LLaMA-1.3B")

tokenized = tokenizer([instruction+'\n'+output], return_tensors="pt", max_length=1024, truncation=True)

input_ids = tokenized['input_ids']
attention_mask = tokenized['attention_mask']
labels = input_ids.clone()
label_len = tokenizer([output], return_tensors="pt", max_length=1024, truncation=True, add_special_tokens=False)['input_ids'].shape[-1]
labels[0,:-label_len] = 1
print(tokenizer.decode(input_ids[0]))
print('###############')
print(tokenizer.decode(labels[0]))
labels[0,:-label_len] = -100
model.train()
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
print(loss) 