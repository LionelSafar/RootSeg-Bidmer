import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def plot_multiclass_segmentation(image_list: list, predicted_list: list,
    savepath: str, class_names: list, segmented_list: list=None, savename: int=None):
    """
    Transforms a single-channel segmentation map (0-6) into a colored image,
    plots it with a legend, and saves the figure.

    Args:
        seg_map (np.ndarray): The 2D array of class indices (H, W), with values 0 to 6.
        save_path (str): The full path to save the output image (e.g., "colored_seg.png").
        class_names (list, optional): List of class names corresponding to indices 0-6.
                                      Defaults to ["Class 0", ..., "Class 6"].
    """

    # Include Soil class to plotting as well
    if class_names is None:
        return

    classes = ["Soil"] + class_names
    num_classes = len(classes)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:olive", "tab:pink"]
    cmap = plt.matplotlib.colors.ListedColormap(colors[:num_classes])

    N_plots = len(image_list)
    if segmented_list is not None:
        title_names = ["Input", "Label", "Prediction"]
        scaling_factor = 2.5 # scale factor for text
        fig, axs = plt.subplots(N_plots, 3, figsize=(17, 25))
        for i in range(N_plots):
            img = image_list[i]
            img = img[92:-92, 92:-92, :]
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            seg = segmented_list[i]
            pred = predicted_list[i]
            axs[i, 0].imshow(img)
            axs[i, 1].imshow(seg, cmap=cmap, 
                vmin=-0.5, vmax=num_classes - 0.5, interpolation="none")
            axs[i, 2].imshow(pred, cmap=cmap, 
                vmin=-0.5, vmax=num_classes - 0.5, interpolation="none")
            if i==0:
                for k, title in enumerate(title_names):
                    axs[i, k].set_title(title, fontsize=12*scaling_factor)
            patches = []
            for j in range(num_classes):
                color = cmap(j)
                patches.append(mpatches.Patch(color=color, label=classes[j]))

    else: # No ground truth given
        title_names = ["Input", "Prediction"]
        pad = 512-388
        scaling_factor = 2.5 # scale factor for text
        fig, axs = plt.subplots(N_plots, 2, figsize=(14, 25))
        for i in range(N_plots):
            img = image_list[i]
            img = img[pad:-pad, pad:-pad, :]
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            pred = predicted_list[i]
            axs[i, 0].imshow(img)
            axs[i, 1].imshow(pred, cmap=cmap, 
                vmin=-0.5, vmax=num_classes - 0.5, interpolation="none")
            if i==0:
                for k, title in enumerate(title_names):
                    axs[i, k].set_title(title, fontsize=12*scaling_factor)
            patches = []
            for j in range(num_classes):
                color = cmap(j)
                patches.append(mpatches.Patch(color=color, label=classes[j]))

    fig.legend(handles=patches, 
           title="Category", 
           loc="lower center", 
           bbox_to_anchor=(0.5, -0.05), 
           ncol=4, 
           frameon=False,
           fontsize=12 * scaling_factor,
           title_fontsize=(12 + 2) * scaling_factor, 
           handlelength=scaling_factor//2, 
           handletextpad=scaling_factor//2
         )
    for ax in axs.ravel():
        ax.axis("off")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if savename is None:
        plt.savefig(savepath+"/model_inf.pdf", format="pdf", dpi=300, bbox_inches="tight")
    else:
        plt.savefig(savepath+savename, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)



def visualize_multiclass(segmentation, class_names, savepath):
    """
    Visualises multiclass segmentation output as labeled image.
    """
    # Set Background as Soil
    classes = ["Soil"] + class_names
    num_classes = len(classes)
    class_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:olive", "tab:pink"]
    cmap = plt.matplotlib.colors.ListedColormap(class_colors[:num_classes])
    fig, ax = plt.subplots(figsize=(8, 8))
    img_plot = ax.imshow(segmentation, cmap=cmap, vmin=-0.5, vmax=len(classes) - 0.5, interpolation="none")
    ax.set_title("Multiclass Segmentation Result")
    ax.axis("off")
    patches = []
    for i in range(num_classes):
        color = cmap(i)
        patches.append(mpatches.Patch(color=color, label=classes[i]))
    ax.legend(handles=patches, 
              title="Classes", 
              loc="center left", 
              bbox_to_anchor=(1.05, 0.5),
              frameon=False)
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def vis_seg_rgb(segmentation, savepath=None):
    """
    Converts a grayscale segmentation map to an RGB image. Allows for up to 6 different classes + background
    """

    if not isinstance(segmentation, np.ndarray):
        segmentation = segmentation.cpu().numpy() if hasattr(segmentation, "cpu") else np.array(segmentation)
    segmentation = segmentation.squeeze().astype(np.uint8)

    # Define color map 
    colors = {
        0: (0, 0, 0),         # background
        1: (0, 0, 255),       # blue
        2: (255, 0, 0),       # red
        3: (0, 255, 0),       # green
        4: (255, 255, 0),     # yellow
        5: (255, 0, 255),     # magenta
        6: (0, 255, 255),     # cyan
    }

    # Create new RGB image and apply mapping
    h, w = segmentation.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    for k, color in colors.items():
        mask = segmentation == k
        rgb_img[mask] = color

    if savepath is not None:
        Image.fromarray(rgb_img).save(savepath)
