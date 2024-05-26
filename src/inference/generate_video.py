import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda"
dtype = torch.float16

def generate_video(prompt, step, output_path, repo="ByteDance/AnimateDiff-Lightning", base="emilianJR/epiCRealism"):
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    
    adapter = MotionAdapter().to(device, dtype)
    adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
    pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
    
    output = pipe(prompt=prompt, guidance_scale=1.0, num_inference_steps=step)
    export_to_gif(output.frames[0], output_path)

if __name__ == "__main__":
    text_prompt = "A girl smiling"
    step = 4  # Options: [1, 2, 4, 8]
    output_path = "animation.gif"
    
    generate_video(text_prompt, step, output_path)
