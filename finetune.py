# -*- coding: utf-8 -*-
# @Author : liang
# @File : finetune.py


import os, yaml
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from utils.dataset import CrystalDataset, CrystalDataLoader
from models.dgm import DGModel
import warnings

warnings.filterwarnings('ignore')


def train_fn(data_loader, model, optimizer, device, criterion):
    model.train()  # Put the model in training mode.
    lr_list = []
    train_losses = []
    predictions = []
    labels = []

    print('Training...')

    for batch_step, (se3_graph_prompt, so3_graph_prompt, y) in tqdm(enumerate(data_loader), total=len(data_loader)):  # Loop over all batches.
        optimizer.zero_grad()  # To zero out the gradients.

        se3_graph_prompt, so3_graph_prompt, y = se3_graph_prompt.to(device), so3_graph_prompt.to(device), y.to(device)
        y_pred = model(se3_graph_prompt, so3_graph_prompt)

        loss = criterion(y_pred, y)

        loss.backward()                         # To backpropagate the error (gradients are computed).
        optimizer.step()                        # To update parameters based on current gradients.

        train_losses.append(loss.item())
        lr_list.append(optimizer.param_groups[0]["lr"])

        if device == 'cpu':
            predictions.extend(y_pred.detach().numpy())
            labels.extend(y.detach().numpy())

        else:
            predictions.extend(y_pred.cpu().detach().numpy())
            labels.extend(y.cpu().detach().numpy())


        # Print the loss when batch_step reaches 300
        if batch_step % 300 == 0 or batch_step % len(data_loader) == 0:
            # mae = mean_absolute_error(np.array(labels) * 1.0, np.array(predictions) * 1.0)
            print(f"Step {batch_step}, Current Train Loss: {np.mean(train_losses)}")

    # mae = mean_absolute_error(np.array(labels) * 1.0, np.array(predictions) * 1.0)
    print('Train Loss is:', np.mean(train_losses), 'Learning rate is:', np.mean(lr_list))
    return np.mean(train_losses), np.mean(lr_list)


def validate_fn(data_loader, model, device, criterion):
    model.eval()  # Put model in evaluation mode.
    val_losses = []
    predictions = []
    labels = []

    print('Validating...')
    with torch.no_grad():  # Disable gradient calculation.
        for batch_step, (se3_graph_prompt, so3_graph_prompt, y) in tqdm(enumerate(data_loader), total=len(data_loader)):  # Loop over all batches.
            # optimizer.zero_grad()  # To zero out the gradients.

            se3_graph_prompt, so3_graph_prompt, y = se3_graph_prompt.to(device), so3_graph_prompt.to(device), y.to(device)
            y_pred = model(se3_graph_prompt, so3_graph_prompt)

            loss = criterion(y_pred, y)

            val_losses.append(loss.item())

            if device == 'cpu':
                predictions.extend(y_pred.detach().numpy())
                labels.extend(y.detach().numpy())

            else:
                predictions.extend(y_pred.cpu().detach().numpy())
                labels.extend(y.cpu().flatten().numpy())

    # predictions = np.array(predictions)
    # labels.append(labels)
    mae = mean_absolute_error(np.array(labels) * 1.0, np.array(predictions) * 1.0)
    print('Valid loss:', np.mean(val_losses), 'Valid MAE:', mae)

    return np.mean(val_losses), mae


def pred_fn(data_loader, model, datawrapper, device, criterion):
    mean, std = datawrapper.mean, datawrapper.std

    model.eval()  # Put model in evaluation mode.
    test_losses = []
    predictions = []
    labels = []

    print('Testing...')
    with torch.no_grad():  # Disable gradient calculation.
        for batch_step, (se3_graph_prompt, so3_graph_prompt, y) in tqdm(enumerate(data_loader), total=len(data_loader)):  # Loop over all batches.
            # optimizer.zero_grad()  # To zero out the gradients.

            se3_graph_prompt, so3_graph_prompt, y = se3_graph_prompt.to(device), so3_graph_prompt.to(device), y.to(device)
            y_pred = model(se3_graph_prompt, so3_graph_prompt)

            loss = criterion(y_pred, y)

            test_losses.append(loss.item())

            if device == 'cpu':
                predictions.extend(y_pred.detach().numpy())
                labels.extend(y.detach().numpy())

            else:
                predictions.extend(y_pred.cpu().detach().numpy())
                labels.extend(y.cpu().flatten().numpy())

    # predictions = np.array(predictions)
    # labels.append(labels)
    labels = (np.array(labels) * std) + mean
    predictions = (np.array(predictions) * std) + mean

    mae = mean_absolute_error(labels, predictions)
    print('Test loss:', np.mean(test_losses), 'Test MAE:', mae)

    return np.mean(test_losses), mae

def return_layers(model):
    layer_list = []
    for name, param in model.named_parameters():
        if 'pred_head' in name:
            print(name, param.requires_grad)
            layer_list.append(name)
    return layer_list


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    config = yaml.load(open("/home/zl/DGM/config/finetune.yml", "r"), Loader=yaml.FullLoader)
    print(config)

    datawrapper = CrystalDataLoader(
        root = config['root'],
        name =config['name'],
        target = config['target'],
        batch_size = config['batch_size'],
        num_workers = config['num_workers'],
        train_size = config['train_size'],
        valid_size = config['val_size'],
        test_size = config['test_size'],
        normalize = config['normalize'],
        idx_save_file = config['idx_save_file'],
        random_seed = config['random_seed']
    )

    (train_loader, 
     val_loader, _) = datawrapper.get_data_loaders()

    model = DGModel(config=config,
                     config_model=config['model']
                     ).to(config['device'])

    # if torch.cuda.device_count() > 1:
    #     print(f"Using GPUs: {config['device']}")
    #     model = model.cuda(config['device'][0])
    #     model = nn.DataParallel(model, device_ids=config['device'])
    # model.cuda()

    if config['load_pretrained_model_path']:
        # print('Loading pre-trained model from', config['load_pretrained_model_path'])
        try:
            state_dict = torch.load(os.path.join(str(config['load_pretrained_model_path']),
                                                 config['pretrained_model_pt']),
                                    map_location=config['device'])

            model.load_state_dict(state_dict['model_state_dict'], strict=False)
            print('Loading pre-trained model from', config['load_pretrained_model_path'], 'with success.')

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'] - config['warmup_steps'],
        eta_min=0,
        last_epoch=-1)
    )

    # criterion = nn.HuberLoss(delta=config['delta'], reduction= 'mean')
    # criterion = nn.L1Loss(reduction = 'mean')
    criterion = nn.MSELoss(reduce='mean')

    lr_list = []
    train_losses = []

    ckpt_save_path = os.path.join(config['ckpt_save_path'], config['target'])
    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path, exist_ok=True)


    best_epoch, best_MAE = 0, 100
    for epoch_counter in range(1, config['epochs'] + 1):
        print(f'Epoch: {epoch_counter}')

        # Call the train function and get the training loss
        train_loss, lr = train_fn(train_loader,
                                  model,
                                  optimizer,
                                  config['device'],
                                  criterion)

        # Perform validation and get the validation loss
        val_loss, val_MAE = validate_fn(val_loader,
                                        model,
                                        config['device'],
                                        criterion)

        if epoch_counter >= config['warmup_steps']:
            scheduler.step()

        # if not DEBUG:
        #     wandb.log({"train_loss": train_loss, "val_loss": val_loss, 'lr': lr})
        # If there's improvement on the validation loss, save the model checkpoint.

        if val_MAE < best_MAE:
            # save best MAE value

            best_MAE = val_MAE
            best_epoch = epoch_counter

            checkpoint_filename = str(config['target']) + f'_checkpoint_{best_epoch}_val_MAE_{round(best_MAE, 5)}.pt'

            torch.save(
                model.state_dict(), os.path.join(str(ckpt_save_path), checkpoint_filename))

            print(
                f"Epoch: {best_epoch}, val MAE = {round(best_MAE, 5)}, checkpoint saved.")

    print('Best epoch is:', best_epoch, 'Best val MAE is:', round(best_MAE, 5))

