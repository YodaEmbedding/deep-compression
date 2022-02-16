from compressai.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms


def get_data_transforms(conf):
    return {
        "train": transforms.Compose(
            [
                transforms.RandomCrop(conf.data.patch_size),
                transforms.ToTensor(),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.CenterCrop(conf.data.patch_size),
                transforms.ToTensor(),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    }


def get_datasets(conf, data_transforms):
    return {
        "train": ImageFolder(
            conf.dataset, split="train", transform=data_transforms["train"]
        ),
        "valid": ImageFolder(
            conf.dataset, split="valid", transform=data_transforms["valid"]
        ),
        "test": ImageFolder(
            conf.dataset, split="test", transform=data_transforms["test"]
        ),
    }


def get_dataloaders(conf, device, datasets):
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=conf.data.batch_size,
            num_workers=conf.data.num_workers,
            shuffle=True,
            pin_memory=(device == "cuda"),
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=conf.data.test_batch_size,
            num_workers=conf.data.num_workers,
            shuffle=False,
            pin_memory=(device == "cuda"),
        ),
        "infer": DataLoader(
            datasets["test"],
            batch_size=1,
            num_workers=conf.data.num_workers,
            shuffle=False,
            pin_memory=(device == "cuda"),
        ),
    }
