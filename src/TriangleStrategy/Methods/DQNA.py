# -*- coding: utf-8 -*-

'''DQN-A (Deep Q Network-Action Augmentation).
'''

from ..Environment import EnvX
import torch
import numpy as np


class DecisionMakerDQNA():
    '''The decision maker of DQN-A.
    
    Args:
        head_mask (int): ``head_mask``.
        tail_mask (int): ``tail_mask``.
        device (torch.device): The computing device.

    Attributes:
        training (bool): If ``training=True``, ``model.train()`` will be called. If ``training=False``, ``model.eval()`` will be called.
    '''
    def __init__(self, head_mask, tail_mask, device=torch.device("cuda")):
        self.head_mask = head_mask
        self.tail_mask = tail_mask
        self.training = True
        self.device = device

    def get_model_output(self, model, input_data, action_data):
        '''Get the raw output of a model.
    
        Args:
            model (torch.nn.Module): The neural network model.
            input_data (torch.tensor): A batch of sequencial environment states. The shape is ``(batch_size, sequence_length, input_size)``.
            action_data (torch.tensor): A batch of sequencial actions. The shape is ``(batch_size, sequence_length)``. This tensor is generated in trainer.
        '''
        if self.training:
            model.train()
        else:
            model.eval()
        seq_num = input_data.shape[0]
        q_value_data = []
        info = (None, None)
        for i in range(input_data.shape[1]):
            if i == 0:
                action = torch.ones(seq_num).to(torch.int64).to(self.device)
            else:
                action = action_data[:, i-1]
            state_data = torch.nn.functional.one_hot(
                action, num_classes=3)
            state_data = state_data.reshape((
                state_data.shape[0], 1, state_data.shape[1]
            ))
            q_value, info = model(
                input_data[:, i:i+1, :],
                (state_data, info)
            )
            q_value_data.append(q_value)
        q_value_data = torch.stack(q_value_data, dim=1)
        return q_value_data

    def __call__(self, model, input_data):
        '''Get the decisions of a model.
    
        Args:
            model (torch.nn.Module): The neural network model.
            input_data (torch.tensor): A batch of sequencial environment states. The shape is ``(batch_size, sequence_length, input_size)``.

        Returns:
            signal_series (torch.tensor): A batch of sequencial actions.
        '''
        model.eval()
        if len(input_data.shape) == 2:
            input_data = input_data.reshape(
                1, input_data.shape[0], input_data.shape[1]
            )
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(
                input_data, dtype=torch.float).to(self.device)
        with torch.no_grad():
            seq_num = input_data.shape[0]
            signal_series = []
            action = torch.ones(seq_num).to(torch.int64).to(self.device)
            info = (None, None)
            for i in range(input_data.shape[1]):
                state_data = torch.nn.functional.one_hot(
                    action, num_classes=3)
                state_data = state_data.reshape((
                    state_data.shape[0], 1, state_data.shape[1]
                ))
                q_value, info = model(
                    input_data[:, i:i+1, :],
                    (state_data, info)
                )
                if i < self.head_mask or i >= self.tail_mask:
                    action = torch.ones(seq_num).to(
                        torch.int64).to(self.device)
                else:
                    action = torch.argmax(q_value, dim=-1, keepdim=False)
                signal_series.append(action)
            signal_series = torch.stack(signal_series, dim=1)
            return signal_series

class TrainerDQNA():
    '''The trainer of DQN-A.
    
    Args:
        head_mask (int): ``head_mask``.
        tail_mask (int): ``tail_mask``.
        buy_cost_pct (float): The cost ratio when the agent chooses to buy.
        sell_cost_pct (float): The cost ratio when the agent chooses to sell.
        gamma (float): The discount factor in the reward value function.
        huber_loss_delta (float): The delta value in huber loss function.
        clamp_bound (float): The upper bound on the absolute value of the gradients.
        target_tao (float): The coefficient for target model updating.
        device (torch.device): The computing device.

    Attributes:
        difficulty (float): ``difficulty``.
    '''

    def __init__(
        self,
        head_mask,
        tail_mask,
        buy_cost_pct=0.0001,
        sell_cost_pct=0.0001,
        gamma=0.99,
        huber_loss_delta=1.0,
        clamp_bound=0.5,
        target_tao=0.1,
        device=torch.device("cuda")
    ):
        self.head_mask = head_mask
        self.tail_mask = tail_mask
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.gamma = gamma
        self.loss_fn = torch.nn.HuberLoss(delta=huber_loss_delta)
        self.clamp_bound = clamp_bound
        self.target_tao = target_tao
        self.difficulty = 1.0
        self.device = device

    def update_money(
        self,
        action_last,
        action_now,
        buy_price,
        sell_price,
        buy_cost_pct,
        sell_cost_pct,
        initial_money
    ):
        money = initial_money
        position_delta = action_now-action_last
        if position_delta > 0:
            money = money-position_delta*buy_price
            money = money-position_delta*buy_price*buy_cost_pct
        else:
            position_delta = -position_delta
            money = money+position_delta*sell_price
            money = money-position_delta*sell_price*sell_cost_pct
        return money

    def completely_explore(
        self,
        action_series,
        buy_price_series,
        sell_price_series,
        buy_cost_pct,
        sell_cost_pct
    ):
        # check
        for i in action_series:
            if i != 1 and i != 0 and i != -1:
                raise ValueError("position signal must be 0, 1 or -1.")
        # simulation
        money = 0
        stock = 0
        money_series, asset_series = [], []
        for position_signal, buy_price, sell_price in zip(
            action_series, buy_price_series, sell_price_series
        ):
            position_delta = position_signal-stock
            if position_delta > 0:
                money = money-position_delta*buy_price
                money = money-position_delta*buy_price*buy_cost_pct
            else:
                position_delta = -position_delta
                money = money+position_delta*sell_price
                money = money-position_delta*sell_price*sell_cost_pct
            stock = position_signal
            money_series.append(money)
            asset_series.append(money + stock*(buy_price+sell_price)/2)
        initial_price = (buy_price_series[0]+sell_price_series[0])/2
        reward_series = [0]+[i-j for i, j in zip(
            asset_series[1:], asset_series[:-1])]
        final_reward = sum(reward_series) / initial_price
        # explore
        d_reward = [[0, 0, 0] for i in range(len(action_series))]
        for i in range(self.head_mask, self.tail_mask):
            for action_explore in [-1, 0, 1]:
                money_explore = money_series[i-1]
                asset_before = asset_series[i-1]
                money_explore = self.update_money(
                    action_series[i-1],
                    action_explore,
                    buy_price_series[i],
                    sell_price_series[i],
                    buy_cost_pct,
                    sell_cost_pct,
                    money_explore
                )
                asset_now = money_explore + action_explore\
                    * (buy_price_series[i]+sell_price_series[i])/2
                money_explore = self.update_money(
                    action_explore,
                    action_series[i+1],
                    buy_price_series[i+1],
                    sell_price_series[i+1],
                    buy_cost_pct,
                    sell_cost_pct,
                    money_explore
                )
                asset_after = money_explore + action_series[i+1]\
                    * (buy_price_series[i+1]+sell_price_series[i+1])/2
                d_reward[i][action_explore+1] = (
                    (asset_now-asset_before)
                    + (asset_after-asset_now)*self.gamma
                )/initial_price
        return d_reward, final_reward

    def calculate_loss(self, model, model_target, decision_maker, input_data_batch, buy_price_batch, sell_price_batch):
        # price
        mid_price_batch = (buy_price_batch + sell_price_batch) / 2
        buy_price_batch_ = mid_price_batch + \
            (buy_price_batch - mid_price_batch) * self.difficulty
        sell_price_batch_ = mid_price_batch + \
            (sell_price_batch - mid_price_batch) * self.difficulty
        # action
        action_batch = decision_maker(model, input_data_batch)
        # reward
        reward_action = []
        reward_value = 0
        for position_signal_series, buy_price_series, sell_price_series in zip(
                action_batch.tolist(), buy_price_batch_.tolist(), sell_price_batch_.tolist()):
            d_reward, final_reward = self.completely_explore(
                [i-1 for i in position_signal_series],
                buy_price_series,
                sell_price_series,
                self.buy_cost_pct*self.difficulty,
                self.sell_cost_pct*self.difficulty
            )
            reward_action.append(d_reward)
            reward_value += final_reward
        reward_action = torch.tensor(reward_action).to(self.device)
        # q_value
        q = decision_maker.get_model_output(
            model, input_data_batch, action_batch)
        with torch.no_grad():
            q_target = decision_maker.get_model_output(
                model_target, input_data_batch, action_batch)
        # q_future=r_i+max_a Q(s',a)
        q_now = q[:, self.head_mask:self.tail_mask, :]
        q_future = torch.max(
            q_target.detach(), dim=2, keepdim=False
        )[0][:, self.head_mask+2:self.tail_mask+2]
        q_future = q_future.reshape(
            q_future.shape[0], q_future.shape[1], 1).repeat((1, 1, 3))
        q_future = reward_action[:, self.head_mask:self.tail_mask, :]\
            + self.gamma*self.gamma*q_future
        # loss and reward
        loss_value = self.loss_fn(q_now, q_future)
        return loss_value, reward_value

    def update_target_network(self, model, model_target):
        state_dict = model.state_dict()
        target_dict = model_target.state_dict()
        for i in state_dict:
            target_dict[i] = target_dict[i]*(1-self.target_tao)\
                + state_dict[i]*self.target_tao
        model_target.load_state_dict(target_dict)

    def train_epoch(self, model, data_loader, optimizer, **kwargs):
        '''Train the model for an epoch.
    
        Args:
            model (torch.nn.Module): The neural network model.
            data_loader (torch.utils.data.DataLoader): The data loader which contains the ``train`` dataset.
            optimizer (torch.optim.Adam): The optimizer provided by PyTorch.
            decision_maker (DecisionMakerDQNA): The decision maker that chooses actions according to the model\'s output.
            model_target (torch.nn.Module): The target model in DQN.

        Returns:
            sum_loss, sum_reward (float, float): The average loss and reward of each sequence in the data loader.
        '''
        decision_maker = kwargs["decision_maker"]
        model_target = kwargs["model_target"]
        model.train()
        decision_maker.training = True
        sum_loss, sum_reward, data_amount = 0, 0, 0
        for batch_id, data_batch in enumerate(data_loader):
            # loss
            state_batch, ask_price_batch, bid_price_batch = data_batch
            state_batch = state_batch.to(self.device)
            loss, reward = self.calculate_loss(
                model, model_target, decision_maker, state_batch, ask_price_batch, bid_price_batch
            )
            # log
            sum_loss += loss.tolist()
            sum_reward += reward
            data_amount += state_batch.shape[0]
            # optimize
            optimizer.zero_grad()
            loss.backward()
            for param in model.parameters():
                param.grad.data.clamp_(-self.clamp_bound, self.clamp_bound)
            optimizer.step()
            # update
            self.update_target_network(model, model_target)
        sum_loss /= data_amount
        sum_reward /= data_amount
        return sum_loss, sum_reward

    def test_epoch(self, model, data_loader, **kwargs):
        '''Test the model for an epoch.
    
        Args:
            model (torch.nn.Module): The neural network model.
            data_loader (torch.utils.data.DataLoader): The data loader which contains the ``dev (or test)`` dataset.
            decision_maker (DecisionMakerDQNA): The decision maker that chooses actions according to the model\'s output.
            model_target (torch.nn.Module): The target model in DQN.

        Returns:
            sum_loss, sum_reward (float, float): The average loss and reward of each sequence in the data loader.
        '''
        decision_maker = kwargs["decision_maker"]
        model_target = kwargs["model_target"]
        model.eval()
        decision_maker.training = False
        with torch.no_grad():
            sum_loss, sum_reward, data_amount = 0, 0, 0
            for batch_id, data_batch in enumerate(data_loader):
                state_batch, ask_price_batch, bid_price_batch = data_batch
                state_batch = state_batch.to(self.device)
                loss, reward = self.calculate_loss(
                    model, model_target, decision_maker, state_batch, ask_price_batch, bid_price_batch
                )
                sum_loss += loss.tolist()
                sum_reward += reward
                data_amount += state_batch.shape[0]
            sum_loss /= data_amount
            sum_reward /= data_amount
        return sum_loss, sum_reward

