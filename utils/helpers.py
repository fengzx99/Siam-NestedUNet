import logging
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import os
from utils.dataloaders import (full_path_loader, full_test_loader, CDDloader)
from utils.metrics import jaccard_loss, dice_loss
from utils.losses import hybrid_loss
from models.Models import Siam_NestedUNet_Conc, SNUNet_ECAM
from models.siamunet_dif import SiamUnet_diff
from collections.abc import Iterable
logging.basicConfig(level=logging.INFO)


def initialize_metrics():
    """Generates a dictionary of metrics with metrics as keys
       and empty lists as values

    Returns
    -------
    dict
        a dictionary of metrics

    """
    metrics = {
        'cd_losses': [],
        'cd_corrects': [],
        'cd_precisions': [],
        'cd_recalls': [],
        'cd_f1scores': [],
        'learning_rate': [],
    }

    return metrics


def get_mean_metrics(metric_dict):
    """takes a dictionary of lists for metrics and returns dict of mean values

    Parameters
    ----------
    metric_dict : dict
        A dictionary of metrics

    Returns
    -------
    dict
        dict of floats that reflect mean metric value

    """
    return {k: np.mean(v) for k, v in metric_dict.items()}


def set_metrics(metric_dict, cd_loss, cd_corrects, cd_report, lr):
    """Updates metric dict with batch metrics

    Parameters
    ----------
    metric_dict : dict
        dict of metrics
    cd_loss : dict(?)
        loss value
    cd_corrects : dict(?)
        number of correct results (to generate accuracy
    cd_report : list
        precision, recall, f1 values

    Returns
    -------
    dict
        dict of  updated metrics


    """
    metric_dict['cd_losses'].append(cd_loss.item())
    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])
    metric_dict['learning_rate'].append(lr)

    return metric_dict

def set_test_metrics(metric_dict, cd_corrects, cd_report):

    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])

    return metric_dict


def get_loaders(opt):


    logging.info('STARTING Dataset Creation')

    train_full_load, val_full_load = full_path_loader(opt.dataset_dir)


    train_dataset = CDDloader(train_full_load, aug=opt.augmentation)
    val_dataset = CDDloader(val_full_load, aug=False)

    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return train_loader, val_loader

def get_test_loaders(opt, batch_size=None):

    if not batch_size:
        batch_size = opt.batch_size

    logging.info('STARTING Dataset Creation')

    test_full_load = full_test_loader(opt.dataset_dir)

    test_dataset = CDDloader(test_full_load, aug=False)

    logging.info('STARTING Dataloading')


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return test_loader


def get_criterion(opt):
    """get the user selected loss function

    Parameters
    ----------
    opt : dict
        Dictionary of options/flags

    Returns
    -------
    method
        loss function

    """
    if opt.loss_function == 'hybrid':
        criterion = hybrid_loss
    if opt.loss_function == 'bce':
        criterion = nn.CrossEntropyLoss()
    if opt.loss_function == 'dice':
        criterion = dice_loss
    if opt.loss_function == 'jaccard':
        criterion = jaccard_loss

    return criterion


def load_pretrained_model(model, pretrained_model):
    if pretrained_model is not None:
        logging.info('Loading pretrained model from {}'.format(pretrained_model))


        if os.path.exists(pretrained_model):
            para_state_dict = torch.load(pretrained_model)

            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    logging.warning("{} is not in pretrained model".format(k))
                elif list(para_state_dict[k].shape) != list(model_state_dict[k]
                                                            .shape):
                    logging.warning(
                        "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                        .format(k, para_state_dict[k].shape, model_state_dict[k]
                                .shape))
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)
            logging.info("There are {}/{} variables loaded into {}.".format(
                num_params_loaded,
                len(model_state_dict), model.__class__.__name__))

        else:
            raise ValueError('The pretrained model directory is not Found: {}'.
                             format(pretrained_model))
    else:
        logging.info(
            'No pretrained model to load, {} will be trained from scratch.'.
            format(model.__class__.__name__))




'''
freeze the layer
'''
def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)


def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)


def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)


def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)

def load_model(opt, device, preTrainedModel=None):
    """Load the model

    Parameters
    ----------
    opt : dict
        User specified flags/options
    device : string
        device on which to train model

    """

    # device_ids = list(range(opt.num_gpus))
    model = SNUNet_ECAM(opt.num_channel, 2).to(device)
    if preTrainedModel:
        #model = torch.load(opt.pretrained_model_weight_dir+preTrainedModel)
        model = torch.load("tmp/checkpoint_epoch_68.pt")
        #model = load_pretrained_model(model,opt.pretrained_model_weight_dir+preTrainedModel)
    # model = nn.DataParallel(model, device_ids=device_ids)

    for k,v in model.named_children():
        #print(k)
        freeze_by_names(model,k)

    # unfreeze_layers = ['conv0_1','conv1_1', 'conv0_2','conv2_1','conv1_2',
    #                 'conv0_3', 'conv_3_1','conv_2_2','conv1_3','conv0_4',
    #                    'ca', 'ca1','conv_final'
    #                    ]
    unfreeze_layers = [
                        'conv1_3','conv0_4',
                       'ca', 'ca1', 'conv_final'
                       ]
    for layer in unfreeze_layers:
        unfreeze_by_names(model,layer)

    return model
