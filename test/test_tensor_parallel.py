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
