from thop import profile
from segmentation_models_pytorch import DeepLabV3Plus
import torch
from torchvision.models import resnet34

BATCH_SIZE = 2
PATCH_SIZE = 256

device = 'cuda'

for CHANNELS in range(1, 9):
    # Instantiate model
    model = DeepLabV3Plus(in_channels=CHANNELS,
                          classes=1,
                          activation='sigmoid').to(device)
    # Profile a single forward pass
    input_sample = torch.randn(BATCH_SIZE, CHANNELS, PATCH_SIZE,
                               PATCH_SIZE).to(device)
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        output = model(input_sample)

    # Get FLOPs and MACs from the profiler
    flops, params = profile(model, inputs=(input_sample, ), verbose=False)
    print(f"Channels: {CHANNELS}, GFLOPs: {flops / 1e9 : .2f}")

    print(64 * '=')

    # Optionally, print detailed profiling information
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
