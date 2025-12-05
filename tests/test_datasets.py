
import os
import shutil
from PIL import Image
import torch

from src.dataset import create_dataloaders, TestDataset


def create_fake_dataset(root):
    os.makedirs(os.path.join(root, "classA"), exist_ok=True)
    os.makedirs(os.path.join(root, "classB"), exist_ok=True)

    for i in range(5):
        img = Image.new("RGB", (400, 400), color=(i*20, i*20, i*20))
        img.save(os.path.join(root, "classA", f"a{i}.jpg"))
        img.save(os.path.join(root, "classB", f"b{i}.jpg"))


def test_dataloaders_creation(tmp_path):

    train_dir = tmp_path / "train"
    test_dir  = tmp_path / "test"
    extra_dir = tmp_path / "extra"

    create_fake_dataset(train_dir)
    create_fake_dataset(test_dir)
    create_fake_dataset(extra_dir)

    loaders = create_dataloaders(str(train_dir), str(test_dir), str(extra_dir), batch_size=2)

    
    assert "train_loader" in loaders
    assert "valid_loader" in loaders
    assert "test_loader" in loaders

   
    batch = next(iter(loaders["train_loader"]))
    images, labels = batch

    assert isinstance(images, torch.Tensor)
    assert images.shape[0] == 2  


def test_testdataset(tmp_path):
    img_dir = tmp_path / "test_images"
    os.makedirs(img_dir, exist_ok=True)

    for i in range(3):
        img = Image.new("RGB", (300, 300), color=(100, 100, 100))
        img.save(img_dir / f"img{i}.jpg")

    dataset = TestDataset(str(img_dir))

    assert len(dataset) == 3

    img, path = dataset[0]
    assert isinstance(path, str)
