from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import numpy as np
import plotly.graph_objs as go
import gif
plt.axis('equal')


def plot_originals(image, block=False):
    image = image[0].transpose(0, 2).transpose(0, 1)
    image = image.numpy()

    plt.imshow(image)
    plt.axis('off')
    plt.title('Original')
    plt.show(block=block)


def plotly_3d_depth(recon_depth, texture=None, save=False, filename="", img_idx=None, show=True):
    depth = recon_depth[0].numpy()
    if texture is not None:
        tex = texture[0, 0].numpy()
        # tex = np.flip(tex, axis=1)
        fig = go.Figure(data=[go.Surface(z=-1*depth, surfacecolor=tex, cmin=0)])
    else:
        fig = go.Figure(data=[go.Surface(z=-1*depth)])
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
            ),
        scene_camera=dict(
            up=dict(x=0.05, y=-1, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=0, z=2)
            ),
        # scene_camera = dict(
        #     up=dict(x=0.2, y=-1.5, z=1),
        #     center=dict(x=0, y=0.125, z=0),
        #     eye=dict(x=0, y=-0.5, z=1.8)
        #     ),
        margin=dict(l=1, r=1, t=1, b=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    fig.update_traces(showscale=False)
    if save:
        im_nr_str = "" if img_idx is None else "_im_" + str(img_idx)
        fig.write_image("results/plots/plotly_" + filename + im_nr_str + ".png")
        fig.write_html("results/htmls/plotly_" + filename + im_nr_str + ".html")
    if show:
        fig.show()


def plotly_3d_animate(recon_depth, texture=None, save=False, filename="", img_idx=None, show=True):
    depth = recon_depth[0].numpy()
    if texture is not None:
        tex = texture[0, 0].numpy()
        fig = go.Figure(data=[go.Surface(z=-1*depth, surfacecolor=tex, cmin=0)])
    else:
        fig = go.Figure(data=[go.Surface(z=-1*depth)])

    x_eye, y_eye, z_eye = 0, 0, 1.5
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False,
                       backgroundcolor="rgb(255, 255, 255)",
                       gridcolor="white",
                       showbackground=True,
                       zerolinecolor="white",
                       title=''),
            yaxis=dict(showticklabels=False,
                       backgroundcolor="rgb(255, 255, 255)",
                       gridcolor="white",
                       showbackground=True,
                       zerolinecolor="white",
                       title=''),
            zaxis=dict(showticklabels=False,
                       backgroundcolor="rgb(255, 255, 255)",
                       gridcolor="white",
                       showbackground=True,
                       zerolinecolor="white",
                       title='')
            ),
        scene_camera=dict(
            up=dict(x=0.05, y=-1, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=x_eye, y=y_eye, z=2)
            ),
        margin=dict(l=1, r=1, t=1, b=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    fig.update_traces(showscale=False)

    # for animation
    frames = []
    max_dist = 0.75
    min_dist = -0.75
    step_dist = 0.025
    for t in np.arange(min_dist, max_dist, step_dist):
        frames.append(dict(x=t, y=y_eye, z=z_eye))
    for t in np.arange(min_dist, max_dist, step_dist):
        frames.append(dict(x=max_dist+min_dist-t, y=y_eye, z=z_eye))

    if save:
        im_nr_str = "" if img_idx is None else "_im_" + str(img_idx)
        fig.write_image("results/plots/plotly_" + filename + im_nr_str + ".png")
        fig.write_html(f"results/htmls/plotly_{filename}{im_nr_str}.html")

        # save animation frames
        @gif.frame
        def plot(i):
            layout = fig.layout
            layout['scene']['camera']['eye'] = frames[i]
            fig_i = go.Figure(data=fig.data, layout=layout)
            return fig_i
        gif_frames = []
        for i in range(len(frames)):
            gif_frame = plot(i)
            gif_frames.append(gif_frame)

        gif.save(gif_frames, f'results/plots/plotly_{filename}{im_nr_str}.gif', duration=50)
    if show:
        fig.show()


def plt_3d_depth(depth, image_size, block=False):
    matplotlib.style.use('seaborn')
    x = np.arange(0, image_size, 1)
    y = np.arange(0, image_size, 1)
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, depth[0, 0], cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.view_init(-105, -90)
    plt.show(block=block)


def plot_predicted_depth_map(depth, image_size, img_idx=None, block=False, save=False, filename=""):
    plt_3d_depth(depth, image_size)
    if save:
        im_nr_str = "" if img_idx is None else "_im_" + str(img_idx)
        plt.savefig("results/plots/" + filename + im_nr_str +  ".png")
    plt.close()


def plot_reconstructions(recon_im, recon_depth, total_it="", im_idx="", stage="", block=False):
    # image = recon_dict['images'][im_idx][0].transpose(0,2).transpose(0,1)
    image = recon_im[0].transpose(0, 2).transpose(0, 1)
    image = image.numpy()

    plt.imshow(image, aspect='auto')
    plt.axis('off')
    plt.title('Reconstructed image')
    plt.show(block=block)
    plt.savefig("results/plots/recon_im_number_" + str(im_idx) + "_"
                + total_it + "_it_"
                + "stage_" + stage
                + ".png")
    plt.close()

    depth = recon_depth[0]
    plt.imshow(depth, aspect='auto')
    plt.axis('off')
    plt.title('Reconstructed depth map')
    plt.show(block=block)
    plt.savefig("results/plots/recon_im_depth_" + str(im_idx) + "_"
                + total_it + "_it_"
                + "stage_" + stage
                + ".png")
    plt.close()

    plt_3d_depth(recon_depth.unsqueeze(0).numpy(), recon_depth.shape[-1])
    plt.axis('off')
    plt.title('Reconstructed depth map')
    plt.show(block=block)
    plt.savefig("results/plots/recon_3d_depth_" + str(im_idx) + "_"
                + total_it + "_it_"
                + "stage_" + stage
                + ".png")
    plt.close()


def statistical_box_plot(loss_list, savename=None):
    a=0
    plt.boxplot(loss_list)
    plt.show()

    if savename is not None:
        plt.savefig(savename)
