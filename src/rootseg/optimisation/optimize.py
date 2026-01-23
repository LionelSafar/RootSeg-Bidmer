"""
Model Optimisation script. Current version only supports optimisation of the multiclass models, 
need hyperparameter adjustments + range of variables for single-class model.

Currently Optimizes for
    - learning rate
    - weight decay
    - alpha for focal loss weighting
    - gamma for focal loss
"""

import os
import glob
import argparse

import optuna
import torch
from torch.utils.data import DataLoader

from rootseg.training.models import UNet, SwinT_UNet, SwinB_UNet
from rootseg.training.training import CosineAnnealingWarmRestartsDecay, training_loop
from rootseg.training.logger import DataLogger
from rootseg.training.datasets import TrainDataset_torch, ValDataset_torch, PrefetchWrapper, seed_worker


def main(args):
    g = torch.Generator()
    torch.multiprocessing.set_start_method("spawn", force=True)
    g.manual_seed(42) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mode selection
    if args.class_selection == "roots":
        class_names = None
        output_size = 1
    elif args.class_selection == "carex":
        class_names = ["Carex", "Non-Carex"]
        output_size = 2
    elif  args.class_selection == "multispecies":
        class_names = ["Anthox", "Geum", "Carex", "Leontodon", "Potentilla", "Helictotrichon"]
        output_size = 6
    elif args.class_selection == "classes":
        class_names = ["Carex", "Graminoids", "Herbs"]
        output_size = 3

    if args.basedir.endswith("/"):
        args.basedir = args.basedir[:-1]
        
    # Load datasets
    X_train_dir = os.path.join(args.basedir, "train", "images")
    y_train_dir = os.path.join(args.basedir, "train", "annotations")
    X_val_dir = os.path.join(args.basedir, "val", "images")
    y_val_dir = os.path.join(args.basedir, "val", "annotations")

    X_train = sorted(glob.glob(os.path.join(X_train_dir, "*")))
    y_train = sorted(glob.glob(os.path.join(y_train_dir, "*")))
    X_val = sorted(glob.glob(os.path.join(X_val_dir, "*")))
    y_val = sorted(glob.glob(os.path.join(y_val_dir, "*")))

    train_dataset = TrainDataset_torch(X_train, y_train, N_subimgs=25, multiclass=output_size>1, imgsize=768, outsize=644)
    val_dataset = ValDataset_torch(X_val, y_val, multiclass=output_size>1, imgsize=768, outsize=644)

    def objective(trial: optuna.Trial):
        """Optuna objective function"""
        # Initialise new logger
        logger = DataLogger(class_names=class_names)

        # Select hyperparameters
        batchsize = trial.suggest_categorical("batchsize", [2, 4])
        lr = trial.suggest_float("learning rate", 1e-4, 1e-3, log=True)
        wd = trial.suggest_float("weight decay", 1e-7, 1e-4, log=True)
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        gamma = trial.suggest_float("gamma", 0.0, 4.0)

        # Initialise model with new weight initialisation
        if args.model == "swin_b":
            model = SwinB_UNet(output_size)
        elif args.model == "swin_t":
            model = SwinT_UNet(output_size)
        else:
            raise ValueError(f"Invalid model name '{args.model}'. Expected 'swin_b' or 'swin_t'.")
        model.to(device)

        # Initialise Dataloader based on selected batchsize
        train_loader = DataLoader(
            train_dataset,
            batch_size=batchsize,
            shuffle=True,
            num_workers=15,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batchsize,
            shuffle=False,
            num_workers=15,
            pin_memory=True
        )
        train_loader = PrefetchWrapper(train_loader, device, 2)
        val_loader = PrefetchWrapper(val_loader, device, 2)

        # Initialise optimiser and LR scheduler based on selected base learning rate and weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=wd 
        )
        scheduler = CosineAnnealingWarmRestartsDecay(
            optimizer, 
            T_0=10, 
            T_mult=2, 
            eta_min=1e-6,
            lr_decay_factor=0.8)
        
        # Perform trial
        model = model.to(device)
        _, logger = training_loop(
            model, 
            optimizer, 
            scheduler, 
            train_loader, 
            val_loader, 
            epochs=args.epochs, 
            alpha=alpha, 
            gamma=gamma, 
            save_path=None, 
            logger=logger, 
            device=device,
            N_classes=max(output_size-1, 1), 
            trial=trial
        )

        savepath = f"trained_models/optimisation/trial_{trial.number}"
        os.makedirs(savepath, exist_ok=True)
        logger.plot_metrics(path=savepath)

        return logger.get_last_F1("Val")

    # Create or load study
    path = "trained_models/optimisation/"
    os.makedirs(path, exist_ok=True)
    storage_path = path + f"{args.name}"
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{args.name}",
        storage=f"sqlite:///{storage_path}.db",
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=5),
        load_if_exists=True
    )
    # Perform trials
    study.optimize(objective, n_trials=args.n_trials-len(study.trials), show_progress_bar=True)
    print("finished")
    # Evaluation
    all_trials = study.get_trials(
        states=(optuna.trial.TrialState.COMPLETE,)
    )
    sorted_trials = sorted(
        all_trials,
        key=lambda t: t.value,
        reverse=True
    )
    best_5_trials = sorted_trials[:5]
    print(f"Study Direction: {study.direction.name}")
    print("\n--- Top 5 Trials ---")
    for i, trial in enumerate(best_5_trials):
        print(f"Rank {i+1}:")
        print(f"  Value: {trial.value:.4f}")
        print(f"  Params: {trial.params}")
        print("-" * 15)
    print("Number of finished trials:", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"Best Validation F1: {trial.value:4.2}")
    print(f"Best Params:")
    for key, value in trial.params.items():
        print(f"-> {key}: {value}")
    print("Best trial number:", trial.number)

    path = f"trained_models/optimisation"

    # Save parameter importance
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(path + f"/param_importance_{args.name}.pdf", scale=3)

    # Save parallel coordinate
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.update_layout(
        title_text="",
        font=dict(size=16)
    )
    fig.write_image(path + f"/parallel_coordinate_{args.name}.pdf", scale=3)

    # Visualise slice
    fig = optuna.visualization.plot_slice(study)
    fig.write_image(path + f"/slice_{args.name}.pdf", scale=3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        default="swin_b", 
        choices=["swin_b", "swin_t"],
        help="Model to optimise: swin_b or swin_t backboned U-Net"
    )
    parser.add_argument(
        "--class_selection",
        choices=["roots", "carex", "multispecies", "classes"],
        default="roots",
        help="""Choose operational mode for segmentation.
        Default is "roots" to detect roots.
        Options are "roots", "carex", and "multispecies"."""
    )
    parser.add_argument(
        "--basedir", 
        type=str, 
        default="data", 
        help="Path to the train and val data folders"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=30, 
        help="Number of max epochs per trial"
    )
    parser.add_argument(
        "--n_trials", 
        type=int, 
        default=50, 
        help="Number of total trials"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="swin_DB", 
        help="Name of optuna study"
    )
    args = parser.parse_args()

    main(args)