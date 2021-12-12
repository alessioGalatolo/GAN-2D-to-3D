import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
plt.axis('equal')

def plot_predicted_depth_map(depth, image_size, img_idx=0, block=False):
    x = np.arange(0, image_size, 1)
    y = np.arange(0, image_size, 1)
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, depth[0], cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    plt.show(block=block)


def plot_originals(image, block=False):
    image = image[0].transpose(0, 2).transpose(0, 1)
    image = image.numpy()

    plt.imshow(image)
    plt.axis('off')
    plt.title('Original')
    plt.show(block=block)


def plot_3d_depth(recon_depth, image_size, block=False):
    depth = recon_depth[0]
    x = np.arange(0, image_size, 1)
    y = np.arange(0, image_size, 1)
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, depth.numpy(), cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    plt.axis('off')
    plt.title('3D depth')
    plt.show(block=block)    
    plt.savefig("results/plots/recon_3d_depth.png")


def plot_reconstructions(recon_im, recon_depth, total_it="", im_idx="", stage="", block=False):
    # image = recon_dict['images'][im_idx][0].transpose(0,2).transpose(0,1)
    image = recon_im[0].transpose(0, 2).transpose(0, 1)
    image = image.numpy()

    plt.imshow(image, aspect='auto')
    plt.axis('off')
    plt.title('Reconstructed image')
    plt.show(block=block)
    plt.savefig("results/plots/recon_im_number_" + im_idx + "_"
                + total_it + "_it_"
                + "stage_" + stage
                + ".png")

    depth = recon_depth[0]
    plt.imshow(depth, aspect='auto')
    plt.axis('off')
    plt.title('Reconstructed depth map')
    plt.show(block=block)
    plt.savefig("results/plots/recon_im_depth_" + im_idx + "_"
                + total_it + "_it_"
                + "stage_" + stage
                + ".png")
