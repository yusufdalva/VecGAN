import os
import sys
from importlib import import_module
from easydict import EasyDict
from PIL import Image

from torchvision import transforms
import torchvision.utils as vutils

from core.networks import Generator

def parse_config(config_path):
    assert os.path.isfile(config_path), f"{config_path} is not a file"
    config_dir = os.path.dirname(config_path)
    config_file = os.path.basename(config_path)
    config_module_name, extension = os.path.splitext(config_file)
    assert extension == ".py", "File specified by config_path is not a Python file"
    sys.path.insert(0, config_dir)
    module = import_module(config_module_name)
    sys.path.pop(0)
    config = EasyDict()
    for key, value in module.__dict__.items():
        if key.startswith("__"):
            continue
        config[key] = value
    del sys.modules[config_module_name]
    return config

def get_img_transform(img_res):
    return transforms.Compose([transforms.Resize((img_res, img_res)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def read_img(img_path, img_res):
    assert os.path.isfile(img_path), f"The input {img_path} is not a file"
    img_transform = get_img_transform(img_res)
    img = Image.open(img_path).convert("RGB")
    return img_transform(img).unsqueeze(0)

def save_img(img_tensor, save_dir, img_name):
    assert os.path.isdir(save_dir), f"The input {save_dir} is not a directory"
    img = (img_tensor + 1) / 2 # Pixel values between [0,1]
    img_path = os.path.join(save_dir, img_name)
    vutils.save_image(img, img_path)

def build_model(config_path, ckpt_path):
    hyperparameters = parse_config(config_path)
    model = Generator(hyperparameters, random_init=True)
    model.load(ckpt_path)
    return model