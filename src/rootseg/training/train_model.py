"""
Training script to train a single-, or multiclass-segmenter

"""
import os
import glob

import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from rootseg.training.logger import DataLogger
from rootseg.training.datasets import TrainDataset_torch, ValDataset_torch, PrefetchWrapper, seed_worker
from rootseg.training.models import UNet, SwinT_UNet, SwinB_UNet
from rootseg.training.training import training_loop, CosineAnnealingWarmRestartsDecay


def train_model(args):

    # Torch initialisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator()
    torch.multiprocessing.set_start_method("spawn", force=True)
    g.manual_seed(42)

    # select class and corresponding model output size
    if args.class_selection == "roots":
        class_names = None
        output_size = 1
    elif args.class_selection == "carex":
        class_names = ["Carex", "Non-Carex"]
        output_size = 2
    elif  args.class_selection == "multispecies":
        class_names = ["Anthox", "Geum", "Carex", "Leontodon", "Potentilla", "Helictotrichon"]
        output_size = 7
    elif args.class_selection == "classes":
        class_names = ["Carex", "Graminoids", "Herbs"]
        output_size = 3
    
    # Load train and val images and labels as list of sorted paths (corresponding X, y pairs)
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

    # Set model input and output size, based on whether basic U-Net or Swin-Transformer backed U-Net is used
    # NOTE: For Transformer-based models the increased receptive field leads to performance improvement
    if args.model.lower() == "unet":
        in_size = 572
        outsize = 388
    else: # multiclass networks
        in_size = 768
        outsize = 644

    # If using transfer learning, don't load pretrained backbones, since trained weights will be overwritten
    if args.pretrained_model:
        use_pretrained_backbone = False
    else:
        use_pretrained_backbone = True
        

    # Initialise model
    if args.model.lower() == "unet":
        model = UNet(64, output_size, 4)
    elif args.model.lower() == "swin_b":
        model = SwinB_UNet(output_size, pretrained=use_pretrained_backbone)
    elif args.model.lower() == "swin_t":
        model = SwinT_UNet(output_size, pretrained=use_pretrained_backbone)


     # Initialise training and validation datasets
    train_dataset = TrainDataset_torch(
        X_train, y_train, N_subimgs=30, multiclass=output_size>1, 
        imgsize=in_size, outsize=outsize, crop_annot=True
    )
    val_dataset = ValDataset_torch(
        X_val, y_val, multiclass=output_size>1, imgsize=in_size, 
        outsize=outsize, crop_annot=True
    )

    # Transfer learning case -- load pretrained model and merge train data with finetune dataset
    # Uses a reduced lr and no warm restarts in the learning
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))
        X_transfer_dir = os.path.join(args.basedir, "transfer", "images")
        y_transfer_dir = os.path.join(args.basedir, "transfer", "annotations")

        X_transfer = sorted(glob.glob(os.path.join(X_transfer_dir, "*")))
        y_transfer = sorted(glob.glob(os.path.join(y_transfer_dir, "*")))
        transfer_dataset = TrainDataset_torch(
            X_transfer, y_transfer, N_subimgs=30, multiclass=output_size>1, 
            imgsize=in_size, outsize=outsize, crop_annot=True
        )

        # Create a sampler that takes new and old data for 50% of the time - however we allow replacement here
        n_old = len(train_dataset)
        n_new = len(transfer_dataset)
        weights = ([0.5 / n_old] * n_old + [0.5 / n_new] * n_new)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=2 * n_new,
            replacement=True
        )

        train_dataset = ConcatDataset([train_dataset, transfer_dataset]) #concatenate new and old data

        # Use sampler instead of shuffle
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=15,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True
        )
        epochs = args.epochs
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate//5, # take 1/5 of the lr for finetuning
            weight_decay=args.weight_decay 
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs, # use a single decay -- no annealing
            eta_min=args.min_lr
        )

    else:
        # Initialise dataloader and wrap them
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=15,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True
        )
        # Initialise optimizer and LR scheduler in standard cosine annealing with given lr
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay 
        )
        scheduler = CosineAnnealingWarmRestartsDecay(
            optimizer, 
            T_0=10, 
            T_mult=2, 
            eta_min=args.min_lr,
            lr_decay_factor=0.8
        )
        epochs = args.epochs

    val_loader = DataLoader(val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=15,
        pin_memory=True
    )
    train_loader = PrefetchWrapper(train_loader, device, 2)
    val_loader = PrefetchWrapper(val_loader, device, 2)
    
    # Initialise logger and saving directories
    logger = DataLogger(class_names=class_names)
    os.makedirs(args.save_path, exist_ok=True)
    if args.identifier:
        save_path = f"{args.save_path}/{args.model.lower()}_{args.identifier}/"
        model_name = args.model.lower()+"_"+args.identifier
    else:
         save_path = f"{args.save_path}/{args.model.lower()}/"
         model_name = args.model.lower()
    checkpointpath = os.path.join(save_path, "checkpoints")
    figpath = os.path.join(save_path, "figures")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(figpath, exist_ok=True)
    os.makedirs(checkpointpath, exist_ok=True)

    # Train model
    model = model.to(device)
    _, logger = training_loop(
        model, 
        optimizer, 
        scheduler, 
        train_loader, 
        val_loader, 
        epochs=epochs, 
        alpha=args.alpha, 
        save_path=checkpointpath, 
        logger=logger, 
        device=device, 
        gamma=args.gamma,
        N_classes=max(output_size-1, 1) # If output == 1, take 1 class, else subtract soil class
    )

    # save final model
    torch.save(model.state_dict(), checkpointpath + "/model.pth")
    
    # Save training curves and training metrics
    logger.plot_metrics(path=figpath)
    logger.save(checkpointpath + "/metrics.h5")
    
    print("Training successfully finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--basedir", 
        type=str, 
        default="data/train/",
        help="Path to the training and validation data"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default="trained_models/",
        help="Path to save the model checkpoints"
    )

    parser.add_argument(
        "--class_selection",
        choices=["roots", "carex", "multispecies", "classes"],
        default="roots",
        help="""Choose operational mode for segmentation.
                Default is 'roots' to detect roots only.
                Options are 'roots', 'carex', and 'multispecies'."""
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="Unet",
        help="Model to train, choices are 'unet', 'swin_t', 'swin_b'"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50,
        help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1.5e-4,
        help="Initial base learning rate for cosine annealing"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=5e-6,
        help="weight decay for the optimizer (AdamW used)"
    )
    parser.add_argument(
        "--min_lr", 
        type=float, 
        default=5e-7,
        help="minimum learning rate for cosine annealing"
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.35,
        help="Weight for the combined loss, specifically the weight for cross-entropy"
    )
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=1.2,
        help="exponent for focal loss"
    )
    parser.add_argument(
        "--identifier", 
        type=str, 
        default="default",
        help="identifier label for the trained network and figure saving"
    )

    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="if transfer learning, indicate the path of the pretrained model. It is assumed that the model is" \
             "the same type as indicated in 'model'."
    )
    args = parser.parse_args()

    train_model(args)