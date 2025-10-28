import torch


def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True

    return False


def init_generator(device: torch.device, fallback: torch.Generator = None):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback


def split_inverse_concat(tensor: torch.Tensor) -> torch.Tensor:
    """
    Split the input tensor along the batch dimension into two matrices,
    take the inverse of each matrix, and concatenate them back along the batch dimension.

    Parameters:
    tensor (torch.Tensor): The input tensor of shape (batch_size, n, n), assumed to be invertible.

    Returns:
    torch.Tensor: A tensor where each split matrix has been inverted and concatenated back along the batch dimension.
    """
    # Split the tensor along the batch dimension into two parts
    batch_size = tensor.shape[0]
    mid_point = batch_size // 2

    # First and second split tensors
    tensor1 = tensor[:mid_point, :, :]
    tensor2 = tensor[mid_point:, :, :]

    # Take the inverse of each split tensor
    tensor1_inv = torch.inverse(tensor1)
    tensor2_inv = torch.inverse(tensor2)

    # Concatenate them back along the batch dimension
    result = torch.cat([tensor1_inv, tensor2_inv], dim=0)

    return result


# QR-based pseudoinverse function
def qr_based_pseudoinverse(A: torch.Tensor) -> torch.Tensor:
    """
    Computes the pseudoinverse of a matrix using QR factorization.

    Parameters:
    A (torch.Tensor): The input matrix.

    Returns:
    torch.Tensor: The pseudoinverse of the input matrix.
    """
    A = A.float()
    _, m, n = A.shape

    Q, R = torch.linalg.qr(A)
    R1 = R[:n, :n]  # Get the square upper triangular matrix
    R1_inv = torch.inverse(R1)
    A_pinv = R1_inv @ Q[:, :n].transpose(-1, -2)
    return A_pinv.half()


def conjugate_transpose_inverse(A: torch.Tensor) -> torch.Tensor:
    """
    Computes a regularized pseudoinverse using the formula (A^T A)^(-1) A^T.
    Suitable for matrices with linearly independent columns.

    Parameters:
    A (torch.Tensor): The input matrix.

    Returns:
    torch.Tensor: The regularized pseudoinverse of the input matrix.
    """
    A = A.float()
    A_T = A.transpose(-1, -2)
    A_pinv = split_inverse_concat(A_T @ A + torch.eye(A.shape[-1], device=A.device).unsqueeze(0) * 1e-4) @ A_T
    # A_pinv = torch.inverse(A_T @ A) @ A_T
    return A_pinv.half()


def do_nothing(x: torch.Tensor, mode: str = None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(input.unsqueeze(-1), dim - 1 if dim < 0 else dim, index.unsqueeze(-1)).squeeze(-1)
    else:
        return torch.gather(input, dim, index)
