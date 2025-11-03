import torch

from torch.profiler import profile, ProfilerActivity, record_function

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(input)

prof.export_chrome_trace