from openaimodel import UNetModel

import torch
import numpy as np
import torch.nn as nn

batch_size = 64

# Define model parameters
image_size = 32
in_channels = 3
model_channels = 160
out_channels = 3
num_res_blocks = 2
attention_resolutions = (4, 2, 1)
dropout = 0.2
channel_mult = (1, 2, 4, 4)
conv_resample = True
dims = 2
num_classes = None
use_checkpoint = False
use_fp16 = False
num_heads = 1
num_head_channels = -1
num_heads_upsample = -1
use_scale_shift_norm = False
resblock_updown = False
use_new_attention_order = False
use_spatial_transformer = True
transformer_depth = 1
context_dim = 1280
n_embed = None
legacy = True

# Create an instance of the UNetModel
model = UNetModel(
    image_size=image_size,
    in_channels=in_channels,
    model_channels=model_channels,
    out_channels=out_channels,
    num_res_blocks=num_res_blocks,
    attention_resolutions=attention_resolutions,
    dropout=dropout,
    channel_mult=channel_mult,
    conv_resample=conv_resample,
    dims=dims,
    num_classes=num_classes,
    use_checkpoint=use_checkpoint,
    use_fp16=use_fp16,
    num_heads=num_heads,
    num_head_channels=num_head_channels,
    num_heads_upsample=num_heads_upsample,
    use_scale_shift_norm=use_scale_shift_norm,
    resblock_updown=resblock_updown,
    use_new_attention_order=use_new_attention_order,
    use_spatial_transformer=use_spatial_transformer,
    transformer_depth=transformer_depth,
    context_dim=context_dim,
    n_embed=n_embed,
    legacy=legacy
)


model.to('cuda:0')


# summary(model, input_size=[(in_channels, image_size, image_size),
#                            (1,), (1, 1280)], batch_size=batch_size, device="cuda")

# Generate some dummy data to run through the model
dummy_data = torch.randn(batch_size, in_channels, image_size, image_size).to('cuda:0')

# create a sample timesteps tensor of size 64 x 1000
timesteps = torch.randint(0, 1000, (batch_size,)).long().to('cuda:0')

cont = torch.randn(batch_size, 1, context_dim).to('cuda:0')

# Pass the data through the model
output = model(dummy_data, timesteps, cont)

# Print the output shape
print(output.shape)


