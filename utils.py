import torch, os
import torch.distributed as dist

def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))
    if ckp_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            ckp_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(ckp_path, map_location='cpu')
        
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            if key == "model_ema":
                value.ema.load_state_dict(checkpoint[key])
            else:
                value.load_state_dict(checkpoint[key])
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]



def load_pretrained_weights(model, pretrained_weights, checkpoint_key=None, prefixes=None,drop_head="head"):
    """load vit weights"""
    if pretrained_weights == '':
        return
    elif pretrained_weights.startswith('https'):
        state_dict = torch.hub.load_state_dict_from_url(
            pretrained_weights, map_location='cpu', check_hash=True)
    else:
        state_dict = torch.load(pretrained_weights, map_location='cpu')
    
    epoch = state_dict['epoch'] if 'epoch' in state_dict else -1
    if not checkpoint_key: 
        for key in ['model', 'teacher', 'encoder']:
            if key in state_dict: checkpoint_key = key
            
    print("Load pre-trained checkpoint from: %s[%s] at %d epoch" % (pretrained_weights, checkpoint_key, epoch))
    
    state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    if prefixes is None: prefixes= ["module.","backbone."]
    for prefix in prefixes:
        state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if not drop_head in k }
    # remove `backbone.` prefix induced by multicrop wrapper
    checkpoint_model = state_dict
    
    
    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] ) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    # print('debug:', pos_embed_checkpoint.shape,orig_size,new_size,num_extra_tokens)
    
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed
    
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))