import torch
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from pytorch_lightning.loggers import WandbLogger

# Custom packages
from train.dataloader import Dataset
from train.train_configs import get_cfg
from train.trainer import TrainingModule
from models.traversability_net import TravNet
from adversarial_attacks.shadow_attacker_optimized import ShadowAttack
import tqdm
import wandb




# RUN COMMAND:  python3 wayfaster/eval_shadow_attack.py

# Parameters. For quick testing, just evaluate on val
BATCH_SIZE = 10
TRAIN_BATCH_END_IDX = 1
LOGGER_RUN_NAME = "shadow_attack_eval"
EVALUATE_ON_TRAIN = False
EVALUATE_ON_VAL = True

# Note: I wanted to log images at step = data index but its impossible with wandb nor WandbLogger. I give up (I tried, but its their internal step tracker that fucks it up)
# Therefore this evaluation script contains no index information, although you can still see results on wandb.ai, just step not related to index,


def visualize_input_output(
    model,
    image,
    trav_map,
    pred_depth,
    prefix,
):
    """It logs a batch of images, with first datapoint logged on step = `start_data_idx`"""

    # Prepare depth_prediction
    n_d = (model.grid_bounds["dbound"][1] - model.grid_bounds["dbound"][0]) / model.grid_bounds["dbound"][2]
    depth_pred = torch.argmax(pred_depth, dim=1, keepdim=True) / (n_d - 1)
    shape = depth_pred.shape
    depth_pred = depth_pred.unsqueeze(0).view(BATCH_SIZE, 6, *shape[1:])

    for i in range(BATCH_SIZE):
        # Visualize the camera inputs
        wandb.log({prefix + "_images": wandb.Image(image[i])})
        # Visualize the traversability map
        wandb.log({prefix + "_mu": wandb.Image(trav_map[i, 0])})
        wandb.log({prefix + "_nu": wandb.Image(trav_map[i, 1])})
        # Visualize the depth prediction
        wandb.log({prefix + "_depth_pred": wandb.Image(depth_pred[i])})


def load_data(configs):
    train_dataset, train_loader, valid_dataset, valid_loader = None, None, None, None
    
    # Create indices to represent subset of dataset (get every sixth elements due to sequence length)
    indices = list(range(0, len(dataset), 6))

    if EVALUATE_ON_TRAIN:
        train_dataset = Dataset(configs, configs.DATASET.TRAIN_DATA)
        train_dataset = Subset(train_dataset, indices)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )
    if EVALUATE_ON_VAL:
        valid_dataset = Dataset(configs, configs.DATASET.VALID_DATA)
        valid_dataset = Subset(valid_dataset, indices)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )
    return train_dataset, train_loader, valid_dataset, valid_loader


def evaluate_attacker(trav_model, dataloader, attacker, dataloader_prefix: str):
    """
    Evaluate adversarial attacks on the pretrained traversability network model for some dataloader.
    It will log results on WANDB with step (x-axis) being the datapoint index (not epoch).
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with torch.no_grad():
        trav_model.eval()

        for batch_idx, batch in tqdm.tqdm(enumerate(dataloader)):
            if batch_idx == TRAIN_BATCH_END_IDX:
                break
            # Unperturbed data:
            color_img, pcloud, inv_intrinsics, extrinsics, path, target_trav, trav_weights, depth_target, depth_mask = batch  # fmt: skip

            # Load them to the right device
            color_img = color_img.to(device)
            pcloud = pcloud.to(device)
            inv_intrinsics = inv_intrinsics.to(device)
            extrinsics = extrinsics.to(device)

            trav_map, pred_depth, _ = trav_model(color_img, pcloud, inv_intrinsics, extrinsics)
            visualize_input_output(
                trav_model,
                color_img,
                trav_map,
                pred_depth,
                prefix=dataloader_prefix + "_unperturbed",
            )

            # Perturb data by shadow attack
            # mean_abs_error is of shape (batch_size, ) as it simply compares unperturbed output and perturbed output.
            mean_abs_error, perturbed_color_img = attacker.generate_attack(
                model_inputs=[color_img, pcloud, inv_intrinsics, extrinsics]
            )
            print("length of mean abs error", len(mean_abs_error))
            trav_map, pred_depth, _ = trav_model(perturbed_color_img, pcloud, inv_intrinsics, extrinsics)
            visualize_input_output(
                trav_model,
                perturbed_color_img,
                trav_map,
                pred_depth,
                prefix=dataloader_prefix + "_perturbed",
            )

            # Log mean_abs_error.
            for i in range(BATCH_SIZE):
                wandb.log({dataloader_prefix + "_mean_abs_error": mean_abs_error[i].item()})


def main():
    # Parse config file
    CONFIG_FILE_PATH = "configs/temporal_model.yaml"
    configs = get_cfg(CONFIG_FILE_PATH)

    # Set seed and device
    pl.seed_everything(configs.SEED, workers=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load data
    train_dataset, train_loader, valid_dataset, valid_loader = load_data(configs)

    # Load pretrained model and logger (it's pl.lightning module, model.model is the actual traversability network)
    model = TrainingModule(configs)
    # wandb_logger = WandbLogger(project=LOGGER_RUN_NAME)
    wandb.init(project=LOGGER_RUN_NAME)

    if configs.MODEL.LOAD_NETWORK is not None:
        print("Loading saved network from {}".format(configs.MODEL.LOAD_NETWORK))
        pretrained_dict = torch.load(configs.MODEL.LOAD_NETWORK, map_location="cpu")["state_dict"]
        model.load_state_dict(pretrained_dict)
        model = model.to(device)  # Load model to cuda

    # Extract actual traversability model
    trav_model: TravNet = model.model
    trav_model = trav_model.to(device)

    # Set up adversarial attacker (Don't change parameters, its fast but at the expense of a lot of memory... will crash your PC, I will run this on cluster.)
    attacker = ShadowAttack(trav_model)
    attacker.PSO_params["num_iters"] = 2
    attacker.PSO_params["n_particles"] = 2
    print("Shadow Attack PSO Parameters: ")
    print(attacker.PSO_params)

    # Evaluate adversarial attacks on train set (results are logged)
    if EVALUATE_ON_TRAIN:
        evaluate_attacker(trav_model, train_loader, attacker, dataloader_prefix="train")

    # Evaluate adversarial attacks on validation set (results are logged)
    if EVALUATE_ON_VAL:
        evaluate_attacker(trav_model, valid_loader, attacker, dataloader_prefix="val")


if __name__ == "__main__":
    main()
