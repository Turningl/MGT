# -*- coding: utf-8 -*-
# @Author : liang
# @File : pretraining.py


import os, yaml, datetime
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.dataset import CrystalDataset, CrystalDataLoader
from models.dgm import DGModel
import warnings

warnings.filterwarnings('ignore')


def train_fn(data_loader, model, optimizer, device):
    model.train()  # Put the model in training mode.
    lr_list = []
    train_losses = []

    print('Training...')

    for batch_step, (se3_graph_prompt, so3_graph_prompt) in tqdm(enumerate(data_loader), total=len(data_loader)):  # Loop over all batches.
        optimizer.zero_grad()  # To zero out the gradients.

        loss = model(se3_graph_prompt.to(device), so3_graph_prompt.to(device))

        loss.backward()                         # To backpropagate the error (gradients are computed).
        optimizer.step()                        # To update parameters based on current gradients.

        train_losses.append(loss.item())

        lr_list.append(optimizer.param_groups[0]["lr"])

        # if device == 'cpu':
        #     predictions.extend(y_pred.detach().numpy())
        #     labels.extend(y.detach().numpy())
        #
        # else:
        #     predictions.extend(y_pred.cpu().detach().numpy())
        #     labels.extend(y.cpu().detach().numpy())


        # Print the loss when batch_step reaches 500
        if batch_step % 500 == 0 or batch_step % len(data_loader) == 0:
            # mae = mean_absolute_error(np.array(labels) * 1.0, np.array(predictions) * 1.0)
            print('Step', batch_step, 'Current Train Loss', np.mean(train_losses))

    # mae = mean_absolute_error(np.array(labels) * 1.0, np.array(predictions) * 1.0)
    print('Train Loss', np.mean(train_losses), 'Learning Rate', np.mean(lr_list))

    return np.mean(train_losses), np.mean(lr_list)


def validate_fn(data_loader, model, device):
    model.eval()  # Put model in evaluation mode.
    val_losses = []

    print('Validating...')
    # with torch.no_grad():  # Disable gradient calculation.
    for batch_step, (se3_graph_prompt, so3_graph_prompt) in tqdm(enumerate(data_loader), total=len(data_loader)):  # Loop over all batches.
        #optimizer.zero_grad()  # To zero out the gradients.

        loss = model(se3_graph_prompt.to(device), so3_graph_prompt.to(device))

        val_losses.append(loss.item())

            # if device == 'cpu':
            #     predictions.extend(y_pred.detach().numpy())
            #     labels.extend(y.detach().numpy())
            #
            # else:
            #     predictions.extend(y_pred.cpu().detach().numpy())
            #     labels.extend(y.cpu().flatten().numpy())

    # predictions = np.array(predictions)
    # labels.append(labels)
    print('Val Loss', np.mean(val_losses))

    return np.mean(val_losses)


def pred_fn(data_loader, model, device):
    model.eval()  # Put model in evaluation mode.
    test_losses = []

    print('Testing...')
    # with torch.no_grad():  # Disable gradient calculation.
    for batch_step, (se3_graph_prompt, so3_graph_prompt) in tqdm(enumerate(data_loader), total=len(data_loader)):  # Loop over all batches.
        #optimizer.zero_grad()  # To zero out the gradients.

        loss = model(se3_graph_prompt.to(device), so3_graph_prompt.to(device))

        test_losses.append(loss.item())

            # if device == 'cpu':
            #     predictions.extend(y_pred.detach().numpy())
            #     labels.extend(y.detach().numpy())
            #
            # else:
            #     predictions.extend(y_pred.cpu().detach().numpy())
            #     labels.extend(y.cpu().flatten().numpy())

    # predictions = np.array(predictions)
    # labels.append(labels)
    print('Test Loss:', np.mean(test_losses))

    return np.mean(test_losses)

# def return_layers(model):
#     layer_list = []
#     for name, param in model.named_parameters():
#         if 'pred_head' in name:
#             print(name, param.requires_grad)
#             layer_list.append(name)
#     return layer_list

if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    config = yaml.load(open("/home/zl/DGM/config/pretraining.yml", "r"), Loader=yaml.FullLoader)
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
        idx_save_file = config['idx_save_file']
    )

    (train_loader,
     val_loader,
     test_loader) = datawrapper.get_data_loaders()

    # model = GTMTModel(tokenizer = tokenizer,
    #                   config=config,
    #                   config_model=config['model']).to(config['device'])

    # if torch.cuda.device_count() > 1:
    #     print(f"Using GPUs: {config['device']}")
    #     model = model.cuda(config['device'][0])
    #     model = nn.DataParallel(model, device_ids=config['device'])
    # model.cuda()

    model = DGModel(config=config,
                     config_model=config['model']
                     ).to(config['device'])

    if config['resume_ckpt_path']:
        try:
            print('Loading resumed model from', config['resume_ckpt_path'])
            # print(RESUME_CKPT_PATH)
            state_dict = torch.load(os.path.join(config['load_pretrained_model_path'], 'pretraining_checkpoint_best.pt'),
                                    map_location=config['device'])

            model.load_my_state_dict(state_dict)

        except FileNotFoundError:
            print("Resume-trained weights not found. Training from pretraining_model.")

    # train_losses = []
    # val_losses = []
    # test_losses = []

    best_epoch, best_loss = 0, 100

    ckpt_save_path = os.path.join(config['ckpt_save_path'])
    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path, exist_ok=True)

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

    for epoch_counter in range(1, config['epochs'] + 1):

        print(f'Epoch: {epoch_counter}')

        # Call the train function and get the training loss
        train_loss, lr = train_fn(train_loader,
                                  model,
                                  optimizer,
                                  config['device'])

        # Perform validation and get the validation loss
        val_loss = validate_fn(val_loader,
                                model,
                                config['device'])

        if epoch_counter >= config['warmup_steps']:
            scheduler.step()

        # if not DEBUG:
        #     wandb.log({"train_loss": train_loss, "val_loss": val_loss, 'lr': lr})
        # If there's improvement on the validation loss, save the model checkpoint.

        if val_loss < best_loss:
            # save best val loss
            best_loss = val_loss
            best_epoch = epoch_counter

            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            checkpoint_filename = f'pretraining_checkpoint_{best_epoch}_{current_time}_loss_{best_loss}.pt'

            torch.save({'epoch': best_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'learning_rate': lr},

                         os.path.join(str(ckpt_save_path), checkpoint_filename))

            print(
                f"Epoch: {epoch_counter}, Best Val Loss = {round(best_loss, 5)}, checkpoint saved.")

            # # test loss to observe whether the model is overfitting
            # test_loss = pred_fn(test_loader, model, config['device'])
            #
            # train_losses.append(train_loss)
            # val_losses.append(val_loss)
            # test_losses.append(test_loss)

    # df = pd.DataFrame(columns=['train_losses', 'val_losses', 'test_losses'])
    #
    # df['train_losses'] = train_losses
    # df['val_losses'] = val_losses
    # df['test_losses'] = test_losses
    #
    # df.to_pickle(str(config['result_save_path']) + '_loss.pkl')

    print('Best Epoch is:', best_epoch, 'Best Loss is:', best_loss)
