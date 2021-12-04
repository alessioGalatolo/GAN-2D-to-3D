import matplotlib.pyplot as plt
import numpy as np

def plot_originals(images, im_idx=0):
    image = images[im_idx][0].view(128,128,3)
    image = image.numpy()

    plt.imshow(image)
    plt.show(block=False)

    return 0

def plot_reconstructions(recon_dict, im_idx=0):
    image = recon_dict['images'][im_idx][0].view(128,128,3)
    image = image.numpy()

    plt.imshow(image)
    plt.show(block=False)

    return 0
