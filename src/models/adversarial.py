import torch
import torch.nn.functional as F
from copy import deepcopy


def fgsm_attack(model, data, epsilon=0.005):
    model.eval()

    data_adv = deepcopy(data)
    data_adv.x.requires_grad = True

    out = model(data_adv)
    loss = F.cross_entropy(
        out[data_adv.train_mask],
        data_adv.y[data_adv.train_mask]
    )

    model.zero_grad()
    loss.backward()

    data_grad = data_adv.x.grad.data
    sign_data_grad = data_grad.sign()

    perturbed_x = data_adv.x + epsilon * sign_data_grad
    data_adv.x = perturbed_x.detach()

    return data_adv