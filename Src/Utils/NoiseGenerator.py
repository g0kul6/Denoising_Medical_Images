import torch
import skimage


def add_gaussian(img,variance):
    for i in img:
        i = torch.tensor(skimage.util.random_noise(i, mode="gaussian", var=variance, clip=True ))
    return i

def s_p(img,amt):
    for i in img:
        i=torch.tensor(skimage.util.random_noise(i, mode="s&p", amount=amt, clip=True ))
    return i

