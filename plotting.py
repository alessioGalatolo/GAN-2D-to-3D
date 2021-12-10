import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def plot_predicted_depth_map(depth, image_size, img_idx=0):
    x = np.arange(0, image_size, 1)
    y = np.arange(0, image_size, 1)
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, depth[0], cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    plt.show(block=False)


def plot_originals(images, im_idx=0):
    image = images[im_idx][0].transpose(0, 2).transpose(0, 1)
    image = image.numpy()

    plt.imshow(image)
    plt.show(block=False)

    return


def plot_3d_depth(recon_depth, image_size):
    depth = recon_depth[0]
    x = np.arange(0, image_size, 1)
    y = np.arange(0, image_size, 1)
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, depth.numpy(), cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    plt.show(block=False)
    plt.savefig("results/plots/recon_3d_depth.png")
    return


def plot_reconstructions(recon_im, recon_depth, total_it="", im_idx="", stage=""):
    # image = recon_dict['images'][im_idx][0].transpose(0,2).transpose(0,1)
    image = recon_im[0].transpose(0, 2).transpose(0, 1)
    image = image.numpy()

    plt.imshow(image, aspect='auto')
    plt.show(block=False)
    plt.savefig("results/plots/recon_im_number_" + im_idx + "_"
                + total_it + "_it_"
                + "stage_" + stage
                + ".png")

    depth = recon_depth[0]
    plt.imshow(depth, aspect='auto')
    plt.show(block=False)
    plt.savefig("results/plots/recon_im_depth_" + im_idx + "_"
                + total_it + "_it_"
                + "stage_" + stage
                + ".png")
    return
