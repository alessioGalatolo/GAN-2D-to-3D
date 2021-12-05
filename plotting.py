import matplotlib.pyplot as plt
import numpy as np

def plot_originals(images, im_idx=0):
    image = images[im_idx][0].transpose(0,2).transpose(0,1)
    image = image.numpy()

    plt.imshow(image)
    plt.show(block=False)

    return

def plot_reconstructions(recon_dict, im_idx=0):
    image = recon_dict['images'][im_idx][0].transpose(0,2).transpose(0,1)
    image = image.numpy()

    plt.imshow(image)
    plt.show(block=False)
    plt.savefig("results/plots/reconstructed_test.png")

    return
