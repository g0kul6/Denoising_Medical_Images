import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
from PIL import Image

def add_gaussian(img,variance):
    gaussian_img = skimage.util.random_noise(img, mode="gaussian", var=variance, clip=True )
    return gaussian_img

def s_p(img,amt):
    s_p_img=skimage.util.random_noise(img, mode="s&p", amount=amt, clip=True )
    return s_p_img

