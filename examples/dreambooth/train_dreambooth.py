import argparse
import random
import json
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

torch.backends.cudnn.benchmark = True

logger = get_logger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help=(
            "The size for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--save_interval", type=int, default=10_000, help="Save weights every N steps.")
    parser.add_argument("--save_min_steps", type=int, default=0, help="Start saving weights after N steps.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    return parser.parse_args()

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_prompt,
        instance_data_dir,
        tokenizer,
        size=512,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.pad_tokens = False

        self.instance_images_paths = []

        inst_img_path = [(x, instance_prompt) for x in Path(instance_data_dir).iterdir() if x.is_file()]
        self.instance_images_paths.extend(inst_img_path)

        random.shuffle(self.instance_images_paths)

        self._length = len(self.instance_images_paths)

        self.image_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_path, instance_prompt = self.instance_images_paths[index % self._length]
        instance_image = Image.open(instance_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return example

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index]

class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def load_pretrained_models(args):
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        torch_dtype=torch.float32
    )

    vae.requires_grad_(False)

    return tokenizer, text_encoder, vae, unet

def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=Path(args.output_dir, "0", args.logging_dir),
    )

    tokenizer, text_encoder, vae, unet = load_pretrained_models(args)

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_dataset = DreamBoothDataset(
        instance_prompt=args.instance_prompt,
        instance_data_dir=args.instance_data_dir,
        tokenizer=tokenizer,
        size=args.size,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        image_ids = [example["instance_images"] for example in examples]

        image_ids = torch.stack(image_ids)
        image_ids = image_ids.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "image_ids": image_ids,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True
    )

    weight_dtype = torch.float32

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    latents_cache = []
    text_encoder_cache = []
    for batch in tqdm(train_dataloader, desc="Caching latents"):
        with torch.no_grad():
            batch["image_ids"] = batch["image_ids"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
            batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
            latents_cache.append(vae.encode(batch["image_ids"]).latent_dist)
            text_encoder_cache.append(text_encoder(batch["input_ids"])[0])

    train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)

    del vae

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")

    print("Training...please wait")

    def save_weights(step):
        if accelerator.is_main_process:
            text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=text_enc_model,
                vae=vae,
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

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    loss_avg = AverageMeter()
    for _ in range(args.num_train_epochs):
        unet.train()
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):

                with torch.no_grad():
                    latent_dist = batch[0][0]
                    latents = latent_dist.sample() * 0.18215

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = batch[0][1]
                prediction = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(prediction.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                loss_avg.update(loss.detach_(), bsz)

            if not global_step % args.log_interval:
                logs = {"loss": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step > 0 and not global_step % args.save_interval and global_step >= args.save_min_steps:
                save_weights(global_step)

            progress_bar.update(1)
            global_step += 1

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    save_weights(global_step)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
