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
def plot_multiclass_pr_curves(model, val_loader, label):
    model.eval()
    all_probs = []
    all_labels = []
    for images, labels in val_loader:
        logits = model(images)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)  # (N, C, H, W)
    all_labels = np.concatenate(all_labels, axis=0)  # (N, H, W)
    C = all_probs.shape[1]

    plt.figure(figsize=(7, 6))
    y_true = all_labels.ravel().astype(np.uint8)
    y_score = all_probs.ravel()
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.plot(recalls, precisions, label=f"(AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-class Precision–Recall Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    savedir = f'figures/PR_{label}.pdf'
    plt.savefig(savedir, format='pdf', dpi=300, bbox_inches='tight')

    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)

    # We only care about the F1 scores that correspond to a threshold
    f1_scores = f1_scores[:-1] 

    # 4. Find the index of the best F1-score
    best_f1_index = np.argmax(f1_scores)

    # 5. Get the threshold, precision, and recall at that index
    best_f1 = f1_scores[best_f1_index]
    best_threshold = thresholds[best_f1_index]
    best_precision = precisions[best_f1_index]
    best_recall = recalls[best_f1_index]

    print(f"Best F1-Score: {best_f1:.3f}")
    print(f"  Achieved at threshold: {best_threshold:.3f}")
    print(f"  With Precision: {best_precision:.3f}")
    print(f"  And Recall: {best_recall:.3f}")

    # Plot the F1 vs threshold value with maximum marked
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.plot(best_threshold, best_f1, 'ro', markersize=8, 
            label=f'Max F1 ({best_f1:.3f}) at Thresh={best_threshold:.3f}')
    plt.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    savedir = f'figures/PR_{label}_f1.pdf'
    plt.savefig(savedir, format='pdf', dpi=300, bbox_inches='tight')


def main(args):
    # Torch initialisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator()
    torch.multiprocessing.set_start_method('spawn', force=True)
    g.manual_seed(42)
    output_size = 1


    # Load Datasets and Dataloader
    if args.basedir.endswith('/'):
        args.basedir = args.basedir[:-1]    
    X_train_dir = os.path.join(args.basedir, 'train', 'images')
    y_train_dir = os.path.join(args.basedir, 'train', 'annotations')
    X_val_dir = os.path.join(args.basedir, 'val', 'images')
    y_val_dir = os.path.join(args.basedir, 'val', 'annotations')

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


    print('Train data len', len(train_dataset))
    print('val data len', len(val_dataset))
    print('Train dataloader len', len(train_loader))
    print('val dataloader len', len(val_loader))

    # load the model
    model = UNet(64, 1, 4)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    # Plot PR curves
    plot_multiclass_pr_curves(model, val_loader, 'val')
    plot_multiclass_pr_curves(model, train_loader, 'train')

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
        help="Use a trained model to finetune instead of training from scratch"
    )
    args = parser.parse_args()
    
    main(args)


