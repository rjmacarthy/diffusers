from torch.utils.data import Dataset
import random
from pathlib import Path
from image_utils import get_transformed_images
from PIL import Image

class DreamBoothDataset(Dataset):
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

        self.image_transforms = get_transformed_images(size)

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