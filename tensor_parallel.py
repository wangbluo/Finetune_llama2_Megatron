from transformers.cache_utils import Cache
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase
from typing import List, Optional, Tuple, Union
from transformers import logging

logger = logging.get_logger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu" 
tp_size = 1

def set_tp_size(tp):
    tp_size = tp
    
def _check_module(m1, m2, check_grad=False):
        testcase = TestCase()
        named_parameters = dict(m1.named_parameters())
        for name, param_m2 in m2.named_parameters():
            testcase.assertTrue(name in named_parameters)
            param_m1 = named_parameters[name]
            if check_grad:
                param_m2 = param_m2.grad
                param_m1 = param_m1.grad
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
            if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                # cancat the splited param_m2.grad
                process_group = dist.group.WORLD
                tensor_list = [torch.empty_like(param_m2) for _ in range(dist.get_world_size())]
                rank = dist.get_rank()
                tensor_list[rank] = param_m2     
                dist.all_gather(tensor_list, param_m2, group=process_group)
                # Note: torch.cat already creates a contiguous tensor.
                first_dim = param_m2.shape[0]
                param_m2 = torch.cat(tensor_list, dim = 0).contiguous()
            if "o_proj" in name:
                # cancat the splited param_m2.grad
                process_group = dist.group.WORLD
                tensor_list = [torch.empty_like(param_m2) for _ in range(dist.get_world_size())]
                rank = dist.get_rank()
                tensor_list[rank] = param_m2     
                dist.all_gather(tensor_list, param_m2, group=process_group)
                # Note: torch.cat already creates a contiguous tensor.
                first_dim = param_m2.shape[0]
                param_m2 = torch.cat(tensor_list, dim = -1).contiguous()
                
        testcase.assertEqual(param_m2, param_m1)
 
#before q, k, v_proj
class InputTpLinear(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        process_group = dist.group.WORLD
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=process_group)
        return grad_output 

def input_tp_linear(input):
    return InputTpLinear.apply(input)

# for the q k v_proj output
class ColTpLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output):
        #split qkv_weight have already done in get_tensor_sharded_model() 
        
        return output
        
    @staticmethod
    def backward(ctx, grad):
        #all_gather the grad, [384, 768] -> [768, 768]
        print(grad)
        print(grad.shape)
        process_group = dist.group.WORLD
        tensor_list = [torch.empty_like(grad) for _ in range(dist.get_world_size())]
        rank = dist.get_rank()
        tensor_list[rank] = grad     
        dist.all_gather(tensor_list, grad, group=process_group)
        # Note: torch.cat already creates a contiguous tensor.
        first_dim = grad.shape[0]
        grad = torch.cat(tensor_list, dim=first_dim).contiguous()
        return grad

def col_tp_linear(output):
    return ColTpLinear.apply(output)  
  
# for the o_proj output
class RowTpLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output):
        # Use c10d collectselfops
        process_group = dist.group.WORLD
        if not output[0].is_contiguous():
            output[0] = output[0].contiguous()
        dist.all_reduce(output[0], op=dist.ReduceOp.SUM, group=process_group)
        return output[0]
        
    @staticmethod
    def backward(ctx, grad):
        bs = 1
        grad = grad.view(bs, 2048, 768)
        return grad
    
def row_tp_linear(output):
    return RowTpLinear.apply(output)
        
def bind(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the 
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method
    
def get_tensor_sharded_model(model, tp_size):

    # Parallelize the Attention and MLP submodules.
    for layer in model.model.layers:
        # nn.linear will transpose the weight, so we should do row split for qkv and do col split for o.
        # For example, we should make q shape from [2048, 2048] -> [1024, 2048]
        set_tp_size(tp_size)
        
        q_proj = layer.self_attn.q_proj.weight
        k_proj = layer.self_attn.k_proj.weight 
        v_proj = layer.self_attn.v_proj.weight 
        o_proj = layer.self_attn.o_proj.weight  
        row_split_dim = q_proj.shape[0] // tp_size
        col_split_dim = o_proj.shape[-1] // tp_size
        
        layer.self_attn.tp = tp_size
        
        split_q_tensors = torch.split(q_proj, row_split_dim, dim = 0)
        split_k_tensors = torch.split(k_proj, row_split_dim, dim = 0)
        split_v_tensors = torch.split(v_proj, row_split_dim, dim = 0)
        split_o_tensors = torch.split(o_proj, col_split_dim, dim = -1)
        # tp_size is no need larger than device_count
        for i in range(len(split_q_tensors)):
            if dist.get_rank()==i:
                layer.self_attn.q_proj.weight.data = split_q_tensors[i].to(f"{device}:{i}")
                layer.self_attn.k_proj.weight.data = split_k_tensors[i].to(f"{device}:{i}")
                layer.self_attn.v_proj.weight.data = split_v_tensors[i].to(f"{device}:{i}")
                layer.self_attn.o_proj.weight.data = split_o_tensors[i].to(f"{device}:{i}")
   
        bind(layer.self_attn, forward, "forward")

        # Manually adjust the number of heads after sharding the self attention modules.
        # For Llama2 models, your should adjust the number of heads separately.
        assert model.model.config.num_attention_heads % dist.get_world_size() == 0
        layer.self_attn.num_heads = model.model.config.num_attention_heads // dist.get_world_size()
        layer.self_attn.num_key_value_heads = model.model.config.num_key_value_heads // dist.get_world_size()
        layer.self_attn.hidden_size = model.model.config.hidden_size // dist.get_world_size()
        
    
# bind the new forward function with llama2 model's self_attn

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()
    if hasattr(self, 'tp') and self.tp > 1:
        hidden_states = input_tp_linear(hidden_states)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
        

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal=self.is_causal and attention_mask is None and q_len > 1,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)
        
    if hasattr(self, 'tp') and self.tp > 1:
        attn_output = row_tp_linear(attn_output)

    return attn_output, None, past_key_value    
      
        
