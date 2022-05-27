# -*- coding: utf-8 -*-

'''DDPG (Deep Deterministic Policy Gradient).
'''

from ..Environment import EnvX
import torch
import numpy as np


class DecisionMakerDDPG():
    '''The decision maker of DDPG.
    
    Args:
        head_mask (int): ``head_mask``.
        tail_mask (int): ``tail_mask``.
        device (torch.device): The computing device.

    Attributes:
        training (bool): If ``training=True``, ``model.train()`` will be called and epsilon-greedy strategy will be enabled. If ``training=False``, ``model.eval()`` will be called and epsilon-greedy strategy will be disabled.
    '''
    def __init__(self, head_mask, tail_mask, device=torch.device("cuda")):
        self.head_mask = head_mask
        self.tail_mask = tail_mask
        self.training = True
        self.device = device

    def __call__(self, model, input_data):
        '''Get the decisions of a model.
    
        Args:
            model (torch.nn.Module): The neural network model.
            input_data (torch.tensor): A batch of sequencial environment states. The shape is ``(batch_size, sequence_length, input_size)``.

        Returns:
            signal_series (torch.tensor): A batch of sequencial actions.
        '''
        if self.training:
            model.train()
        else:
            model.eval()
        if len(input_data.shape) == 2:
            input_data = input_data.reshape(
                1, input_data.shape[0], input_data.shape[1]
            )
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(
                input_data, dtype=torch.float).to(self.device)
        with torch.no_grad():
            signal_series = model(input_data)
            signal_series = signal_series.reshape(
                signal_series.shape[0],
                signal_series.shape[1]
            )
            signal_series = torch.round(signal_series) + 1
            signal_series[:, :self.head_mask] = 1
            signal_series[:, self.tail_mask:] = 1
            return signal_series


class TrainerDDPG():
    '''The trainer of DQN.
    
    Args:
        head_mask (int): ``head_mask``.
        tail_mask (int): ``tail_mask``.
        buy_cost_pct (float): The cost ratio when the agent chooses to buy.
        sell_cost_pct (float): The cost ratio when the agent chooses to sell.
        gamma (float): The discount factor in the reward value function.
        noise_size (float): The variance of random noise on action values.
        target_tao (float): The coefficient for target model updating.
        warm_up_epoch (int): The number of epochs for warming up. At the first ``warm_up_epoch`` epochs, only critic model is updated.
        device (torch.device): The computing device.

    Attributes:
        difficulty (float): ``difficulty``.
    '''

    def __init__(
        self,
        head_mask=20,
        tail_mask=220,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        gamma=0.99,
        noise_size=0.1,
        target_tao=0.01,
        warm_up_epoch=0,
        device=torch.device("cuda")
    ):
        self.head_mask = head_mask
        self.tail_mask = tail_mask
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.gamma = gamma
        self.noise_size = noise_size
        self.target_tao = target_tao
        self.critic_loss_fn = torch.nn.MSELoss()
        self.warm_up_epoch = warm_up_epoch
        self.device = device
        self.env = EnvX(allow_continuous_action=True)
        self.difficulty = 0.0
        self.epoch_count = 0

    def update_target_network(self, model, model_target):
        state_dict = model.state_dict()
        target_dict = model_target.state_dict()
        for i in state_dict:
            target_dict[i] = target_dict[i]*(1-self.target_tao)\
                + state_dict[i]*self.target_tao
        model_target.load_state_dict(target_dict)

    def get_reward(self,env,action_batch,buy_price_batch, sell_price_batch):
        # price
        mid_price_batch = (buy_price_batch + sell_price_batch) / 2
        buy_price_batch_ = mid_price_batch + \
            (buy_price_batch - mid_price_batch) * self.difficulty
        sell_price_batch_ = mid_price_batch + \
            (sell_price_batch - mid_price_batch) * self.difficulty
        # reward
        reward_batch, sum_reward = [], 0
        for position_signal_series, buy_price_series, sell_price_series in zip(
                action_batch.tolist(), buy_price_batch_.tolist(), sell_price_batch_.tolist()):
            reward_series = env.simulate_trade(
                [i[0] for i in position_signal_series],
                buy_price_series,
                sell_price_series,
                self.buy_cost_pct*self.difficulty,
                self.sell_cost_pct*self.difficulty
            )
            sum_reward += sum(reward_series)
            reward_batch.append(reward_series)
        reward_batch = torch.tensor(reward_batch).to(self.device)
        return reward_batch, sum_reward

    def train_epoch(self, model, data_loader, optimizer, **kwargs):
        '''Train the model for an epoch.
    
        Args:
            model (A list that contains 4 torch.nn.Module): The neural network model, including actor_model, critic_model, target_actor, target_critic.
            data_loader (torch.utils.data.DataLoader): The data loader which contains the ``train`` dataset.
            optimizer (A list that contains 2 torch.optim.Adam): The optimizer provided by PyTorch, including actor_optimizer, critic_optimizer.

        Returns:
            sum_loss, sum_reward (float, float): The average loss and reward of each sequence in the data loader.
        '''
        self.epoch_count += 1
        if self.epoch_count == self.warm_up_epoch:
            print("actor start learning!")
        actor_model, critic_model, target_actor, target_critic = model
        actor_model.train()
        critic_model.train()
        target_actor.train()
        target_critic.train()
        actor_optimizer, critic_optimizer = optimizer
        sum_actor_loss, sum_critic_loss, sum_reward, data_amount = 0, 0, 0, 0
        for batch_id, data_batch in enumerate(data_loader):
            # state
            state_batch, ask_price_batch, bid_price_batch = data_batch
            state_batch = state_batch.to(self.device)
            data_amount += state_batch.shape[0]
            # action
            action_batch = actor_model(state_batch)
            action_batch = action_batch + \
                torch.randn(action_batch.shape).to(self.device) * self.noise_size
            action_batch = action_batch.clamp(-1.0, 1.0)
            action_batch[:, :self.head_mask, :] = 0
            action_batch[:, self.tail_mask:, :] = 0
            # reward
            reward_batch, reward_value = self.get_reward(
                self.env, action_batch, ask_price_batch, bid_price_batch
            )
            sum_reward += reward_value
            # critic loss
            target_action = target_actor(state_batch)
            reward_now = reward_batch[:, self.head_mask:self.tail_mask].reshape((
                reward_batch.shape[0], self.tail_mask-self.head_mask, 1
            ))
            y = reward_now + self.gamma*target_critic(
                state_batch[:, self.head_mask+1:self.tail_mask+1, :],
                target_action[:, self.head_mask+1:self.tail_mask+1, :]
            )
            critic_value = critic_model(
                state_batch[:, self.head_mask:self.tail_mask, :],
                action_batch[:, self.head_mask:self.tail_mask, :]
            )
            critic_loss = self.critic_loss_fn(critic_value, y)
            critic_loss = torch.mean(critic_loss)
            sum_critic_loss += critic_loss.tolist()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            # actor loss
            if self.epoch_count >= self.warm_up_epoch:
                actions = actor_model(state_batch)
                critic_value = critic_model(state_batch, actions)
                actor_loss = -critic_value[:, self.head_mask:self.tail_mask, :]
                actor_loss = torch.mean(actor_loss)
                sum_actor_loss += actor_loss.tolist()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
            # update
            self.update_target_network(critic_model, target_critic)
            self.update_target_network(actor_model, target_actor)
        sum_loss = (sum_actor_loss+sum_critic_loss)/data_amount
        sum_reward /= data_amount
        return sum_loss, sum_reward

    def test_epoch(self, model, data_loader, **kwargs):
        '''Test the model for an epoch.
    
        Args:
            model (A list that contains 4 torch.nn.Module): The neural network model, including actor_model, critic_model, target_actor, target_critic.
            data_loader (torch.utils.data.DataLoader): The data loader which contains the ``test`` dataset.

        Returns:
            sum_loss, sum_reward (float, float): The average loss and reward of each sequence in the data loader.
        '''
        actor_model, critic_model, target_actor, target_critic = model
        actor_model.eval()
        sum_loss, sum_reward, data_amount = 0, 0, 0
        for batch_id, data_batch in enumerate(data_loader):
            # state
            state_batch, ask_price_batch, bid_price_batch = data_batch
            state_batch = state_batch.to(self.device)
            data_amount += state_batch.shape[0]
            # action
            action_batch = actor_model(state_batch)
            action_batch = action_batch + \
                torch.randn(action_batch.shape).to(
                    self.device) * self.noise_size
            action_batch = action_batch.clamp(-1.0, 1.0)
            action_batch[:, :self.head_mask, :] = 0
            action_batch[:, self.tail_mask:, :] = 0
            # reward
            reward_batch, reward_value = self.get_reward(
                self.env, action_batch, ask_price_batch, bid_price_batch
            )
            sum_reward += reward_value
        sum_loss /= data_amount
        sum_reward /= data_amount
        return sum_loss, sum_reward
