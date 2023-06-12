import torch

def get_device() -> torch.device:
    """Returns the device on which the model is trained."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_device_name() -> str:
    """Returns the name of the device on which the model is trained."""
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def get_device_count() -> int:
    """Returns the number of GPUs available."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def model_to_gpu(model) -> bool:
    """Returns True if the model is attached to a GPU."""
    device = get_device()
    model.to(device)
    return next(gpumodel.parameters()).is_cuda


def data_to_gpu(X_train, y_train, X_test, y_test) -> tuple:
    """Returns the data in the GPU."""
    device = get_device()
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    return X_train, y_train, X_test, y_test
