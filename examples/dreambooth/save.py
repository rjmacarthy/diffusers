import os
import json
import torch

from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel

def save_weights(args, accelerator, unet):
    if accelerator.is_main_process:
        text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=text_enc_model,
            vae=AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="vae",
                revision=args.revision,
            ),
            safety_checker=None,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision=None,
        )
        model_name = args.instance_prompt.split(" ")[-1]
        save_dir = os.path.join(args.output_dir, f"{model_name}")
        pipeline.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=2)
        print(f"[*] Weights saved at {save_dir}")