import matplotlib.pyplot as plt
import numpy as np

def plot_originals(images, im_idx=0):
    image = images[im_idx][0].transpose(0,2).transpose(0,1)
    image = image.numpy()

    plt.imshow(image)
    plt.show(block=False)

    return

def plot_reconstructions(recon_im, recon_depth, total_it="", im_idx="", stage=""):
    # image = recon_dict['images'][im_idx][0].transpose(0,2).transpose(0,1)
    image = recon_im[0].transpose(0,2).transpose(0,1)
    image = image.numpy()

    plt.imshow(image, aspect='auto')
    plt.show(block=False)
    plt.savefig("results/plots/recon_im_number_" + im_idx + "_" \
                + total_it + "_it_" \
                + "stage_" + stage \
                + ".png")

    depth = recon_depth[0]
    plt.imshow(depth, aspect='auto')
    plt.show(block=False)
    plt.savefig("results/plots/recon_im_depth_" + im_idx + "_" \
                + total_it + "_it_" \
                + "stage_" + stage \
                + ".png")
    return
