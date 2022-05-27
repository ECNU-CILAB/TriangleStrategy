# -*- coding: utf-8 -*-

'''A2C (Advantage Actor-Critic).
'''

from ..Environment import EnvX
import torch
import numpy as np


class DecisionMakerA2C():
    '''The decision maker of A2C.
    
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
        actor_value_data, critic_value_data = [], []
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
            (actor_value, critic_value), info = model(
                input_data[:, i:i+1, :],
                (state_data, info)
            )
            actor_value_data.append(actor_value)
            critic_value_data.append(critic_value)
        actor_value_data = torch.stack(actor_value_data, dim=1)
        critic_value_data = torch.stack(critic_value_data, dim=1)
        return actor_value_data, critic_value_data

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
                (actor_value, critic_value), info = model(
                    input_data[:, i:i+1, :],
                    (state_data, info)
                )
                if i < self.head_mask or i >= self.tail_mask:
                    action = torch.ones(seq_num).to(
                        torch.int64).to(self.device)
                else:
                    prob = torch.nn.functional.softmax(actor_value, dim=-1)
                    prob = torch.clamp(prob, min=0.001, max=0.999)
                    action = (torch.ones(seq_num) * 2).to(torch.int64).to(self.device)
                    r = torch.rand(action.shape).to(self.device)
                    action[r < prob[:, 0]+prob[:, 1]] = 1
                    action[r < prob[:, 0]] = 0
                signal_series.append(action)
            signal_series = torch.stack(signal_series, dim=1)
            return signal_series


class TrainerA2C():
    '''The trainer of A2C.
    
    Args:
        head_mask (int): ``head_mask``.
        tail_mask (int): ``tail_mask``.
        buy_cost_pct (float): The cost ratio when the agent chooses to buy.
        sell_cost_pct (float): The cost ratio when the agent chooses to sell.
        gamma (float): The discount factor in the reward value function.
        clamp_bound (float): The upper bound on the absolute value of the gradients.
        critic_alpha (float): The coefficient of critic loss.
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
        clamp_bound=0.1,
        critic_alpha=10.0,
        device=torch.device("cuda")
    ):
        self.head_mask = head_mask
        self.tail_mask = tail_mask
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.gamma = gamma
        self.clamp_bound = clamp_bound
        self.critic_loss_fn = torch.nn.MSELoss()
        self.critic_alpha = critic_alpha
        self.env = EnvX()
        self.difficulty = 1.0
        self.device = device

    def calculate_loss(self, model, decision_maker, input_data_batch, buy_price_batch, sell_price_batch):
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
        for position_signal_series, buy_price_series, sell_price_series in zip(
                action_batch.tolist(), buy_price_batch_.tolist(), sell_price_batch_.tolist()):
            reward_series = self.env.simulate_trade(
                [i-1 for i in position_signal_series],
                buy_price_series,
                sell_price_series,
                self.buy_cost_pct*self.difficulty,
                self.sell_cost_pct*self.difficulty
            )
            reward_action.append(reward_series)
        reward_action = torch.tensor(reward_action).to(self.device)
        # prob
        actor_value, critic_value = decision_maker.get_model_output(
            model, input_data_batch, action_batch)
        prob = torch.nn.functional.softmax(actor_value, dim=2)
        critic_value = critic_value.reshape(
            critic_value.shape[0], critic_value.shape[1]
        )
        critic_value = critic_value[:, self.head_mask:self.tail_mask]
        # r(i)+gamma*r(i+1)+gamma^2*r(i+2)...
        with torch.no_grad():
            reward_batch = reward_action.clone().detach()
            for i in range(reward_batch.shape[1]-2, -1, -1):
                reward_batch[:, i] += self.gamma*reward_batch[:, i+1]
        reward_batch = reward_batch[:, self.head_mask:self.tail_mask]
        # actor loss
        mask = torch.nn.functional.one_hot(
            action_batch, num_classes=3).to(self.device)
        prob_action = torch.sum(
            prob*mask, dim=2, keepdim=False)[:, self.head_mask:self.tail_mask]
        reward_baseline = critic_value.detach()
        actor_loss = -(reward_batch - reward_baseline)*torch.log(prob_action)
        actor_loss = torch.sum(actor_loss)
        # critic loss
        critic_loss = self.critic_loss_fn(critic_value, reward_batch)
        critic_loss = torch.sum(critic_loss)
        # loss and reward
        loss_value = actor_loss + self.critic_alpha*critic_loss
        reward_value = torch.sum(reward_action)
        return loss_value, reward_value

    def train_epoch(self, model, data_loader, optimizer, **kwargs):
        '''Train the model for an epoch.
    
        Args:
            model (torch.nn.Module): The neural network model.
            data_loader (torch.utils.data.DataLoader): The data loader which contains the ``train`` dataset.
            optimizer (torch.optim.Adam): The optimizer provided by PyTorch.
            decision_maker (DecisionMakerA2C): The decision maker that chooses actions according to the model\'s output.

        Returns:
            sum_loss, sum_reward (float, float): The average loss and reward of each sequence in the data loader.
        '''
        decision_maker = kwargs["decision_maker"]
        model.train()
        decision_maker.training = True
        sum_loss, sum_reward, data_amount = 0, 0, 0
        for batch_id, data_batch in enumerate(data_loader):
            # loss
            state_batch, ask_price_batch, bid_price_batch = data_batch
            state_batch = state_batch.to(self.device)
            loss, reward = self.calculate_loss(
                model, decision_maker, state_batch, ask_price_batch, bid_price_batch
            )
            # log
            sum_loss += loss.tolist()
            sum_reward += reward.tolist()
            data_amount += state_batch.shape[0]
            # optimize
            optimizer.zero_grad()
            loss.backward()
            for param in model.parameters():
                param.grad.data.clamp_(-self.clamp_bound, self.clamp_bound)
            optimizer.step()
        sum_loss /= data_amount
        sum_reward /= data_amount
        return sum_loss, sum_reward

    def test_epoch(self, model, data_loader, **kwargs):
        '''Test the model for an epoch.
    
        Args:
            model (torch.nn.Module): The neural network model.
            data_loader (torch.utils.data.DataLoader): The data loader which contains the ``dev (or test)`` dataset.
            decision_maker (DecisionMakerA2C): The decision maker that chooses actions according to the model\'s output.

        Returns:
            sum_loss, sum_reward (float, float): The average loss and reward of each sequence in the data loader.
        '''
        decision_maker = kwargs["decision_maker"]
        model.eval()
        decision_maker.training = False
        with torch.no_grad():
            sum_loss, sum_reward, data_amount = 0, 0, 0
            for batch_id, data_batch in enumerate(data_loader):
                state_batch, ask_price_batch, bid_price_batch = data_batch
                state_batch = state_batch.to(self.device)
                loss, reward = self.calculate_loss(
                    model, decision_maker, state_batch, ask_price_batch, bid_price_batch
                )
                sum_loss += loss.tolist()
                sum_reward += reward.tolist()
                data_amount += state_batch.shape[0]
            sum_loss /= data_amount
            sum_reward /= data_amount
        return sum_loss, sum_reward