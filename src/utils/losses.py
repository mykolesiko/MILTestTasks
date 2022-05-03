import torch


def dice_coeff(input, target):
    smooth = 1.

    input_flat = input.view(-1)
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat).sum()
    union = input_flat.sum() + target_flat.sum()

    return (2. * intersection + smooth) / (union + smooth)


def jaccard_coeff(input, target):
    smooth = 1.
    input_flat = input.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (input_flat * target_flat).sum()
    union = input_flat.sum() + target_flat.sum()
    return (intersection + smooth) / (union - intersection + smooth)


def jaccard_loss(input, target):
    return -torch.log(jaccard_coeff(input[:, 0, :, :], target[:, 0, :, :])) \
           - torch.log(jaccard_coeff(input[:, 1, :, :], target[:, 1, :, :]))
