import os
import torch
import numpy as np

from core.networks import Generator
from core.utils import read_img, save_img

IMG_RES = 256

def infer_single_image(model: Generator, x: torch.Tensor, mode: str, tag: int, attribute: int = -1, z: float = -1, ref_img_path: str = None, device: str = "cuda:0"):
    e, s = model.encode(x)
    alpha_x = model.extract(e, tag)
    if mode == "l": # Latent-guided
        if z == -1:
            z = torch.rand(1,1).to(device)
        alpha_trg = model.map(z, tag, attribute)
    elif mode == "r": # Reference-guided
        x_ref = read_img(ref_img_path, IMG_RES)
        e_ref, _ = model.encode(x_ref)
        alpha_trg = model.extract(e_ref, tag)

    alpha = alpha_trg - alpha_x
    e_trg = model.translate(e, tag, alpha)
    x_trg = model.decode(e_trg, s)
    return x_trg

def infer_image_folder(model: Generator, folder_path: str, out_path: str, mode: str, tag: str, attribute: str = None, z: float = -1, ref_path: str = None, device:str = "cuda:0"):
    img_paths = [img_path for img_path in os.listdir(folder_path) if os.path.isfile(img_path) and (img_path.endswith(".jpg") or img_path.endswith(".png"))]
    if mode == "r":
        ref_img_paths = [ref_img_path for ref_img_path in os.listdir(ref_path) if os.path.isfile(ref_img_path) and (ref_img_path.endswith(".jpg") or ref_img_path.endswith(".png"))]
        ref_idxs = np.random.permutation(len(ref_img_paths))

    for img_idx in range(len(img_paths)):
        x = read_img(img_paths[img_idx], img_res).to(device)
        file_name = img_paths[img_idx].split(ps.sep)[-1].split(".")[0]
        if mode == "l":
            x_trg = infer_single_image(model, x, mode, tag, attribute, z, device)
        elif mode == "r":
            ref_img_path = ref_img_paths[ref_idxs[img_idx]]
            x_trg = infer_single_image(model, x, mode, tag, ref_img_path, device)
        save_img(x, out_path, f"{file_name}_out.jpg")
