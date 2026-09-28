"""This module contains simple helper functions"""

from __future__ import print_function
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import torch.distributed as dist
import os


def _ao_on(name, default="1"):
    return os.environ.get(name, default) not in ("0", "false", "False", "")


AUTOOPTM_OPT_1 = _ao_on("AUTOOPTM_OPT_1")

_ao_opt_13 = {}


def _ao_opt_11(image_tensor):
    x = image_tensor[0].float().permute(1, 2, 0).contiguous()
    x = ((x + 1) / 2.0 * 255.0).to(torch.uint8)
    key = tuple(x.shape)
    buf = _ao_opt_13.get(key)
    if buf is None:
        buf = torch.empty(key, dtype=torch.uint8, device="cpu", pin_memory=True)
        _ao_opt_13[key] = buf
    buf.copy_(x, non_blocking=True)
    torch.cuda.synchronize()
    return buf.numpy().copy()


def _ao_opt_12(image_numpy):
    import struct
    import zlib

    h, w, _ = image_numpy.shape
    rows = np.empty((h, w * 3 + 1), dtype=np.uint8)
    rows[:, 0] = 0                                   # filter type 0 (None)
    rows[:, 1:] = image_numpy.reshape(h, w * 3)
    co = zlib.compressobj(0)
    idat = co.compress(rows.tobytes()) + co.flush()

    def _chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", idat) + _chunk(b"IEND", b""))


def tensor2im(input_image, imtype=np.uint8):
    """ "Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        if AUTOOPTM_OPT_1 and image_tensor.is_cuda and image_tensor.dim() == 4 and image_tensor.shape[1] == 3 and imtype is np.uint8:
            return _ao_opt_11(image_tensor)
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name="network"):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


# initialize ddp
def init_ddp():
    # Initialize DDP if LOCAL_RANK is set
    is_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

    if is_ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    print(f"Initialized with device {device}")
    return device


# cleanup ddp
def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    if AUTOOPTM_OPT_1 and aspect_ratio == 1.0 and image_numpy.ndim == 3 and image_numpy.shape[2] == 3 and str(image_path).endswith(".png"):
        with open(image_path, "wb") as fh:
            fh.write(_ao_opt_12(np.ascontiguousarray(image_numpy)))
        return

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print("shape,", x.shape)
    if val:
        x = x.flatten()
        print("mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f" % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)
