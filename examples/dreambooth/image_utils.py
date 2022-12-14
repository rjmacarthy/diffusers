from torchvision import transforms

def get_transformed_images(size):
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )