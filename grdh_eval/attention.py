"""Optional PyTorch SDPA implementation of the existing cross-attention math."""

import types


def sdpa_forward(module, x, context=None, mask=None):
    import torch.nn.functional as functional

    context = x if context is None else context
    batch, length, _ = x.shape
    heads = module.heads
    q = module.to_q(x).reshape(batch, length, heads, -1).transpose(1, 2)
    k = module.to_k(context).reshape(batch, context.shape[1], heads, -1).transpose(1, 2)
    v = module.to_v(context).reshape(batch, context.shape[1], heads, -1).transpose(1, 2)
    attention_mask = None if mask is None else mask.reshape(batch, -1)[:, None, None, :].bool()
    output = functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
    return module.to_out(output.transpose(1, 2).reshape(batch, length, -1))


def enable_sdpa(model):
    from ldm.modules.attention import CrossAttention

    count = 0
    for module in model.modules():
        if isinstance(module, CrossAttention):
            module.forward = types.MethodType(sdpa_forward, module)
            count += 1
    return count
