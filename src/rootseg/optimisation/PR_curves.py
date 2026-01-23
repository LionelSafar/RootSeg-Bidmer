"""
Plot PR Curves of a trained binary segmenter model (only UNet class supported).
NOTE: The performance of threshold between ~0.2 and 0.8 is almost identical, which led to keep 0.5 as default
for the final model.
"""
import os
import glob
import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

from rootseg.training.models import UNet
from rootseg.training.datasets import TrainDataset_torch, ValDataset_torch, PrefetchWrapper, seed_worker

@torch.no_grad()
def plot_multiclass_pr_curves(
    model: torch.nn.Module, 
    loader: PrefetchWrapper, 
    savedir: str,
    label: str
) -> None:
    model.eval()
    all_probs = []
    all_labels = []

    # Gather all output probabilities and labels on cpu
    for images, labels in loader:
        logits = model(images)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)  # (N, C, H, W)
    all_labels = np.concatenate(all_labels, axis=0)  # (N, H, W)
    C = all_probs.shape[1]

    # Get PR curve
    y_true = all_labels.ravel().astype(np.uint8)
    y_score = all_probs.ravel()
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # Plot PR curve
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recalls, precisions, label=f"Avg. Precision (AP)={ap:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{savedir}/PR_{label}.pdf", format="pdf", dpi=300, bbox_inches="tight")

    # Get F1 as function of threshold
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)

    # Remove last -- no threshold case
    f1_scores = f1_scores[:-1] 

    # Get the threshold, precision, and recall at index of best F1
    best_f1_index = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_index]
    best_threshold = thresholds[best_f1_index]
    best_precision = precisions[best_f1_index]
    best_recall = recalls[best_f1_index]

    print(50*"-")
    print(f"Best F1-Score for {label}: {best_f1:.3f}")
    print(f"  Achieved at threshold: {best_threshold:.3f}")
    print(f"  With Precision: {best_precision:.3f}")
    print(f"  And Recall: {best_recall:.3f}")

    # Plot the F1 vs threshold value with maximum marked
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1_scores, label="F1 Score")
    ax.plot(best_threshold, best_f1, "ro", markersize=8, 
            label=f"Max F1 ({best_f1:.3f}) at Thresh={best_threshold:.3f}")
    ax.axvline(x=best_threshold, color="tab:red", linestyle="--", alpha=0.7)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.legend(loc="lower center")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{savedir}/PR_{label}_f1.pdf", format="pdf", dpi=300, bbox_inches="tight")


def main(args):
    # Torch initialisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator()
    torch.multiprocessing.set_start_method("spawn", force=True)
    g.manual_seed(42)
    output_size = 1

    # Load Datasets and Dataloader
    if args.basedir.endswith("/"):
        args.basedir = args.basedir[:-1]    
    X_train_dir = os.path.join(args.basedir, "train", "images")
    y_train_dir = os.path.join(args.basedir, "train", "annotations")
    X_val_dir = os.path.join(args.basedir, "val", "images")
    y_val_dir = os.path.join(args.basedir, "val", "annotations")

    X_train = sorted(glob.glob(os.path.join(X_train_dir, "*")))
    y_train = sorted(glob.glob(os.path.join(y_train_dir, "*")))
    X_val = sorted(glob.glob(os.path.join(X_val_dir, "*")))
    y_val = sorted(glob.glob(os.path.join(y_val_dir, "*")))
    
    train_dataset = TrainDataset_torch(
        X_train, y_train, N_subimgs=40, multiclass=output_size>1, imgsize=576, outsize=388
    )
    val_dataset = ValDataset_torch(
        X_val, y_val, multiclass=output_size>1, imgsize=576, outsize=388
    )
    train_loader = DataLoader(train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=15,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=15,
        pin_memory=True
    )
    train_loader = PrefetchWrapper(train_loader, device, 2)
    val_loader = PrefetchWrapper(val_loader, device, 2)

    # load model
    model = UNet(64, 1, 4)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    save_dir = os.path.abspath(os.path.join(os.path.dirname(args.model_path), "..", "figures", "PR_curves"))
    os.makedirs(save_dir, exist_ok=True)

    # Plot PR curves
    plot_multiclass_pr_curves(model, val_loader, save_dir, "train")
    plot_multiclass_pr_curves(model, train_loader, save_dir, "val")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--basedir", 
        type=str, 
        default="data/train/",
        help="Path to the training and validation data"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="trained model (.pth file) to evaluate PR curve on"
    )
    args = parser.parse_args()
    
    main(args)


