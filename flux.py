import matplotlib.pyplot as plt

import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("PATH_TO_MODEL", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload() # Offloads modules to the CPU on a submodular level - needed for GPUs with little RAM

prompt="A diverse group of founding fathers"

image = pipe(
    prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=4,
    max_sequence_length=250,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

plt.imshow(image)
plt.show()
image.save("image.png")