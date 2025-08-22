import torch
import contextlib
import numpy as np
import random
import shutil
import os

import torch

def check_tensor(tensor, dtype):
    """
    Convert :code:`tensor` into a :code:`dtype` torch.Tensor.

    :param numpy.array/torch.tensor tensor: Input data.
    :param str dtype: PyTorch dtype string.
    :return: A :code:`dtype` torch.Tensor.
    """
    return torch.tensor(tensor, dtype=dtype)


def reset_params(model):
    """
    Reset all parameters in :code:`model`.
    :param torch.nn.Module model: Pytorch model.
    """
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    else:
        for layer in model.children():
            reset_params(layer)


class NumpyDataLoader:
    """
    Convert numpy arrays into a dataloader.

    :param numpy.array *inputs: Numpy arrays.
    """
    def __init__(self, *inputs):
        self.inputs = inputs
        self.n_inputs = len(inputs)

    def __len__(self):
        return self.inputs[0].shape[0]

    def __getitem__(self, item):
        if self.n_inputs == 1:
            return self.inputs[0][item]
        else:
            return [array[item] for array in self.inputs]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, checkpoint_path, filename="checkpoint.pt"):
    filename = os.path.join(checkpoint_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_path, "model_best.pt"))


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint["state_dict"])

def log_metrics(set_name, metrics, logger):
    logger.info(
        "{}: Loss: {:.5f} | depth_acc: {:.5f}, rgb_acc: {:.5f}".format(
            set_name, metrics["loss"], metrics["depth_acc"], metrics["rgb_acc"]
        )
    )


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
def check_tensor(tensor, dtype):
    """
    Convert :code:`tensor` into a :code:`dtype` torch.Tensor.

    :param numpy.array/torch.tensor tensor: Input data.
    :param str dtype: PyTorch dtype string.
    :return: A :code:`dtype` torch.Tensor.
    """
    return torch.tensor(tensor, dtype=dtype)


def reset_params(model):
    """
    Reset all parameters in :code:`model`.
    :param torch.nn.Module model: Pytorch model.
    """
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    else:
        for layer in model.children():
            reset_params(layer)


class NumpyDataLoader:
    """
    Convert numpy arrays into a dataloader.

    :param numpy.array *inputs: Numpy arrays.
    """
    def __init__(self, *inputs):
        self.inputs = inputs
        self.n_inputs = len(inputs)

    def __len__(self):
        return self.inputs[0].shape[0]

    def __getitem__(self, item):
        if self.n_inputs == 1:
            return self.inputs[0][item]
        else:
            return [array[item] for array in self.inputs]