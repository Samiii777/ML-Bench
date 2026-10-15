"""Paths to the small reference inputs in data/ (committed; re-downloaded if deleted)."""

import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCES = {
    "imagenet_classes.txt": "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "dog.jpg": "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
}


def data_file(name: str) -> str:
    path = DATA_DIR / name
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(SOURCES[name], path)
    return str(path)


def get_imagenet_classes_path() -> str:
    return data_file("imagenet_classes.txt")


def get_sample_image_path() -> str:
    return data_file("dog.jpg")
