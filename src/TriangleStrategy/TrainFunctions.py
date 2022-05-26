# -*- coding: utf-8 -*-

'''These functions are helpful in the training process.
'''

import torch
import numpy as np
import datetime
import os


def get_DataLoader(data_path, batch_size):
    '''Wrapping a data set into data loader.

    Args:
        data_path (str): The ``.npz`` data file path.
        batch_size (int): The ``batch_size`` while training.
    '''
    dataset_npz = np.load(data_path)
    input_data = torch.tensor(dataset_npz["input_data"], dtype=torch.float)
    buy_price = torch.tensor(dataset_npz["buy_price"], dtype=torch.float)
    sell_price = torch.tensor(dataset_npz["sell_price"], dtype=torch.float)
    print(input_data.shape, buy_price.shape, sell_price.shape)
    dataset = torch.utils.data.TensorDataset(
        input_data, buy_price, sell_price)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return data_loader


def train_model_until_converge(
        model, data_loader_train, data_loader_dev, data_loader_test, optimizer, trainer,
        epoch_max, epoch_window, epoch_window_win, **kwargs):
    '''Train a model until converge.

    Args:
        model (torch.nn.Module): The neural network model.
        data_loader_train (torch.utils.data.DataLoader): The data loader which contains the **train** dataset.
        data_loader_dev (torch.utils.data.DataLoader): The data loader which contains the **dev** dataset.
        data_loader_test (torch.utils.data.DataLoader): The data loader which contains the **test** dataset.
        optimizer (torch.optim.Adam): The optimizer provided by PyTorch.
        trainer (object): The trainer of reinforcement learning methods.
        epoch_max (int): The parameter that for convergence determination. The training process will continue if the reward is increased in at least ``epoch_window_win`` epochs in the recent ``epoch_window`` epochs, and stop otherwise. If the model has been trained for ``epoch_max`` epochs, the peocess will also stop.
        epoch_window (int): The parameter that for convergence determination.
        epoch_window_win (int): The parameter that for convergence determination.
        kwargs (dict): Other parameters used in ``trainer.train_epoch``, for example, ``DecisionMaker``.

    Returns:
        reward_list (list): The reward value in every epoch.
    '''
    reward_list = []
    for epoch_id in range(epoch_max):
        train_loss, train_reward = trainer.train_epoch(
            model, data_loader_train, optimizer, **kwargs)
        if data_loader_dev is not None:
            dev_loss, dev_reward = trainer.test_epoch(
                model, data_loader_dev, **kwargs)
        else:
            dev_reward = train_reward
        if data_loader_test is not None:
            test_loss, test_reward = trainer.test_epoch(
                model, data_loader_test, **kwargs)
        else:
            test_reward = train_reward
        print(
            "epoch: %d" % epoch_id,
            "difficulty: %.6f" % trainer.difficulty,
            "train_reward: %.6f" % train_reward,
            "dev_reward: %.6f" % dev_reward,
            "test_reward: %.6f" % test_reward,
        )
        reward_list.append(dev_reward)
        if len(reward_list) < kwargs.get("epoch_ignore", 0) + epoch_window:
            continue
        if len(reward_list) >= epoch_window+1:
            d_reward_list = [i-j for i, j in zip(
                reward_list[-epoch_window:], reward_list[-epoch_window-1:-1])]
            if sum([i > 0 for i in d_reward_list]) >= epoch_window_win:
                continue
            else:
                break
    return reward_list


def heuristic_train_model(model, data_loader_train, data_loader_dev, data_loader_test, optimizer, trainer, model_save_folder=None, **kwargs):
    '''Train the model with curriculum learning. The ``difficulty`` (transaction cost) is set to 0 initially and then is increased linearly.

    Args:
        model (torch.nn.Module): The neural network model.
        data_loader_train (torch.utils.data.DataLoader): The data loader which contains the **train** dataset.
        data_loader_dev (torch.utils.data.DataLoader): The data loader which contains the **dev** dataset.
        data_loader_test (torch.utils.data.DataLoader): The data loader which contains the **test** dataset.
        optimizer (torch.optim.Adam): The optimizer provided by PyTorch.
        trainer (object): The trainer of reinforcement learning methods.
        model_save_folder (str): The folder that models are saved to. Models will not be saved if ``model_save_folder`` is ``None``.
        steps (str): The steps while increasing the ``difficulty``.
        kwargs (dict): Other parameters used in ``trainer.train_epoch``, for example, ``DecisionMaker``.

    Returns:
        save_path (str): Save path for the last model.
    '''
    steps = kwargs["steps"]
    for it in range(steps+1):
        trainer.difficulty = it/steps
        train_model_until_converge(
            model, data_loader_train, data_loader_dev, data_loader_test, optimizer, trainer,
            **kwargs
        )
        if model_save_folder is not None:
            save_path = "%s/model_step_%d.pth" % (model_save_folder, it)
            print("save model to %s" % save_path)
            if type(model) is list or type(model) is tuple:
                torch.save(model[0].state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
    return save_path


def quick_save(model, save_path=None):
    '''Quickly save a model.

    Args:
        model (torch.nn.Module): The neural network model.
        save_path (str): Save path for the model. If ``save_path`` is ``None``, a folder will be created automatically.

    Returns:
        save_path (str): Save path for the model.
    '''
    if save_path is None:
        model_save_folder = "models/%s" % datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S_%f')
        os.makedirs(model_save_folder, exist_ok=True)
        save_path = "%s/model.pth" % (model_save_folder)
    print("save model to %s" % save_path)
    torch.save(model.state_dict(), save_path)
    return save_path
