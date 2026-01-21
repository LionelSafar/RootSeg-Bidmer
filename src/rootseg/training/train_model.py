"""
Training script to train a single-, or multiclass-segmenter

"""
import os
import glob
import argparse

import torch
from torch.utils.data import DataLoader

from rootseg.training.datasets import TrainDataset_torch, ValDataset_torch, PrefetchWrapper, seed_worker
from rootseg.training.models import UNet, SwinT_UNet, SwinB_UNet
from rootseg.training.logger import DataLogger
from rootseg.training.training import training_loop, CosineAnnealingWarmRestartsDecay


def train_model(args):

    # Torch init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator()
    torch.multiprocessing.set_start_method('spawn', force=True)
    g.manual_seed(42)

    # select class and corresponding model output size
    if args.class_selection == 'roots':
        class_names = None
        output_size = 1
    elif args.class_selection == 'carex':
        class_names = ['Carex', 'Non-Carex']
        output_size = 2
    elif  args.class_selection == 'multispecies':
        class_names = ['Anthox', 'Geum', 'Carex', 'Leontodon', 'Potentilla', 'Helictotrichon']
        output_size = 7
    elif args.class_selection == 'classes':
        class_names = ['Carex', 'Graminoids', 'Herbs']
        output_size = 3
    
    # Load train and val images and labels as list of sorted paths (corresponding X, y pairs)
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

    # Set model input and output size, based on whether basic U-Net or Swin-Transformer backed U-Net is used
    # NOTE: For Transformer-based models the increased receptive field leads to performance improvement
    if args.model.lower() == 'unet':
        in_size = 572
        outsize = 388
    else: # multiclass networks
        in_size = 768
        outsize = 644

    # Initialise model
    if args.model.lower() == 'unet':
        model = UNet(64, output_size, 4)
        crop_annot = True
    elif args.model.lower() == 'swin_b':
        model = SwinB_UNet(output_size)
        crop_annot = True
    elif args.model.lower() == 'swin_t':
        model = SwinT_UNet(output_size)
        crop_annot = True
    
    # Initialise training and validation datasets
    train_dataset = TrainDataset_torch(
        X_train, y_train, N_subimgs=30, multiclass=output_size>1, 
        imgsize=in_size, outsize=outsize, crop_annot=crop_annot
    )
    val_dataset = ValDataset_torch(
        X_val, y_val, multiclass=output_size>1, imgsize=in_size, 
        outsize=outsize, crop_annot=crop_annot
    )

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
    val_loader = DataLoader(val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=15,
        pin_memory=True
    )
    train_loader = PrefetchWrapper(train_loader, device, 2)
    val_loader = PrefetchWrapper(val_loader, device, 2)
    
    # Initialise logger and saving directorier
    logger = DataLogger(class_names=class_names)
    os.makedirs(args.save_path, exist_ok=True)
    if args.identifier:
        save_path = f'{args.save_path}/{args.model.lower()}_{args.identifier}/'
        model_name = args.model.lower()+'_'+args.identifier
    else:
         save_path = f'{args.save_path}/{args.model.lower()}/'
         model_name = args.model.lower()
    current_dir = os.getcwd()
    figpath = os.path.abspath(os.path.join(current_dir, 'figures', model_name))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(figpath, exist_ok=True)


    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    # Initialise optimizer and LR scheduler
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

    # Train model
    model = model.to(device)
    _, logger = training_loop(
        model, 
        optimizer, 
        scheduler, 
        train_loader, 
        val_loader, 
        epochs=args.epochs, 
        alpha=args.alpha, 
        save_path=save_path, 
        logger=logger, 
        device=device, 
        gamma=args.gamma,
        N_classes=max(output_size-1, 1) # If output == 1, take 1 class, else subtract soil class
    )

    # save final model
    torch.save(model.state_dict(), save_path+"model.pth")
    
    # Save training curves and training metrics
    logger.plot_metrics(model_name=model_name)
    logger.save(save_path + '/metrics.h5')
    
    # Additionally save some of the validation images
    #N_samples = 5
    #idxs = np.random.randint(0, len(val_dataset), N_samples)
    #img_list = []
    #to_model = []
    #seg_list = []
    #pred_list = []

    #for idx in idxs:
    #    X, y_gt = val_dataset[idx] 
    #    img_list.append(X.permute(1, 2, 0).numpy())
    #    to_model.append(X)
    #    seg_list.append(y_gt.permute(1, 2, 0).squeeze(2).numpy()) #to (H, W)
    #X_batch = torch.stack(to_model).to(device)
    #with torch.no_grad():
        # Take GT mask for plotting here..
    #    if X_batch.shape[-1] == seg_list[0].shape[-1]:
            #seg_batch = torch.stack(seg_list).to(device)
    #        logits = model(X_batch) 
    #    else:
    #        logits = model(X_batch) 
    #    y_pred_batch = torch.argmax(logits, dim=1)
    #pred_list = [y_pred.detach().cpu().numpy() for y_pred in y_pred_batch]
    #plot_multiclass_segmentation(img_list, pred_list, figpath, class_names, segmented_list=seg_list)
    print('Training successfully finished!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--basedir", 
        type=str, 
        default="data/train/",
        help="Path to the training and validation data"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default="checkpoints/",
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
        default=5,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-2,
        help="Initial base learning rate for cosine annealing"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0,
        help='weight decay for the optimizer (AdamW used)'
    )
    parser.add_argument(
        "--min_lr", 
        type=float, 
        default=5e-2,
        help="minimum learning rate for cosine annealing"
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.3,
        help="Weight for the combined loss, specifically the weight for cross-entropy"
    )
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=0.5,
        help='factor for elastic deformation within [0, 1]'
    )
    parser.add_argument(
        "--identifier", 
        type=str, 
        default='',
        help="identifier label for the trained network and figure saving"
    )
    args = parser.parse_args()

    train_model(args)