# -*- coding: utf-8 -*-

'''PPO (Proximal Policy Optimization).
'''

from ..Environment import EnvX
import torch
import numpy as np

class DecisionMakerPPO():
    '''The decision maker of PPO.
    
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
        value_data = []
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
            value, info = model(
                input_data[:, i:i+1, :],
                (state_data, info)
            )
            value_data.append(value)
        value_data = torch.stack(value_data, dim=1)
        return value_data

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
                    prob = torch.nn.functional.softmax(q_value, dim=-1)
                    prob = torch.clamp(prob, min=0.001, max=0.999)
                    action = (torch.ones(seq_num) * 2).to(torch.int64).to(self.device)
                    r = torch.rand(action.shape).to(self.device)
                    action[r < prob[:, 0]+prob[:, 1]] = 1
                    action[r < prob[:, 0]] = 0
                signal_series.append(action)
            signal_series = torch.stack(signal_series, dim=1)
            return signal_series


class TrainerPPO():
    '''The trainer of PPO.
    
    Args:
        head_mask (int): ``head_mask``.
        tail_mask (int): ``tail_mask``.
        buy_cost_pct (float): The cost ratio when the agent chooses to buy.
        sell_cost_pct (float): The cost ratio when the agent chooses to sell.
        gamma (float): The discount factor in the reward value function.
        eps (float): The bound value in PPO.
        huber_loss_delta (float): The delta value in huber loss function.
        target_tao (float): The coefficient for target (old) model updating.
        clamp_bound (float): The upper bound on the absolute value of the gradients.
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
        eps=0.2,
        huber_loss_delta=1.0,
        target_tao=0.01,
        clamp_bound=0.01,
        device=torch.device("cuda")
    ):
        self.head_mask = head_mask
        self.tail_mask = tail_mask
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.gamma = gamma
        self.eps = eps
        self.critic_loss = torch.nn.HuberLoss(delta=huber_loss_delta)
        self.target_tao = target_tao
        self.clamp_bound = clamp_bound
        self.env = EnvX()
        self.difficulty = 0.0
        self.warm_up = False
        self.device = device

    def update_target_network(self, model, model_target):
        state_dict = model.state_dict()
        target_dict = model_target.state_dict()
        for i in state_dict:
            target_dict[i] = target_dict[i]*(1-self.target_tao)\
                + state_dict[i]*self.target_tao
        model_target.load_state_dict(target_dict)

    def calculate_loss(self, model, decision_maker, input_data_batch, buy_price_batch, sell_price_batch):
        actor_model, critic_model, actor_model_old, critic_model_old = model
        # price
        mid_price_batch = (buy_price_batch + sell_price_batch) / 2
        buy_price_batch_ = mid_price_batch + \
            (buy_price_batch - mid_price_batch) * self.difficulty
        sell_price_batch_ = mid_price_batch + \
            (sell_price_batch - mid_price_batch) * self.difficulty
        # action
        action_batch = decision_maker(actor_model, input_data_batch)
        if self.warm_up:
            action_batch = torch.randint(0, 3, action_batch.shape, dtype=torch.int64).to(self.device)
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
        sum_reward = torch.sum(reward_action)
        # actor and critic value
        actor_value = decision_maker.get_model_output(
            actor_model, input_data_batch, action_batch)
        prob = torch.nn.functional.softmax(actor_value, dim=-1)
        prob = torch.clamp(prob, min=0.001, max=0.999)
        critic_value = decision_maker.get_model_output(
            critic_model, input_data_batch, action_batch)
        critic_value = critic_value.reshape(
            critic_value.shape[0], critic_value.shape[1]
        )
        # old actor and critic value
        with torch.no_grad():
            actor_value_old = decision_maker.get_model_output(
                actor_model_old, input_data_batch, action_batch)
            prob_old = torch.nn.functional.softmax(actor_value_old, dim=-1)
            prob_old = torch.clamp(prob_old, min=0.001, max=0.999)
            critic_value_old = decision_maker.get_model_output(
                critic_model_old, input_data_batch, action_batch)
            critic_value_old = critic_value_old.reshape(
                critic_value_old.shape[0], critic_value_old.shape[1]
            )
        # advance value
        with torch.no_grad():
            critic_value_target = reward_action[:, self.head_mask:self.tail_mask] + \
                self.gamma * critic_value_old[:, self.head_mask+1:self.tail_mask+1]
            baseline = critic_value_old[:, self.head_mask:self.tail_mask]
            advance = critic_value_target - baseline
        # r_t
        mask = torch.nn.functional.one_hot(
            action_batch, num_classes=3).to(self.device)
        prob_action = torch.sum(
            prob*mask, dim=2, keepdim=False)[:, self.head_mask:self.tail_mask]
        prob_action_old = torch.sum(
            prob_old*mask, dim=2, keepdim=False)[:, self.head_mask:self.tail_mask]
        r_t = prob_action/prob_action_old
        # actor_loss
        r_t_clip = torch.clamp(r_t, 1-self.eps, 1+self.eps)
        actor_loss = torch.minimum(r_t*advance, r_t_clip*advance)
        actor_loss = -torch.sum(actor_loss)
        # critic_loss
        critic_loss = self.critic_loss(
            critic_value[:, self.head_mask:self.tail_mask],
            critic_value_target
        )
        critic_loss = torch.sum(critic_loss)
        return actor_loss, critic_loss, sum_reward

    def train_epoch(self, model, data_loader, optimizer, **kwargs):
        '''Train the model for an epoch.
    
        Args:
            model (A list that contains 4 torch.nn.Module): The neural network model, including actor_model, critic_model, actor_model_old, critic_model_old.
            data_loader (torch.utils.data.DataLoader): The data loader which contains the ``train`` dataset.
            optimizer (A list that contains 2 torch.optim.Adam): The optimizer provided by PyTorch, including actor_optimizer, critic_optimizer.
            decision_maker (DecisionMakerPPO): The decision maker that chooses actions according to the model\'s output.

        Returns:
            sum_loss, sum_reward (float, float): The average loss and reward of each sequence in the data loader.
        '''
        decision_maker = kwargs["decision_maker"]
        decision_maker.training = True
        actor_model, critic_model, actor_model_old, critic_model_old = model
        actor_model.train()
        critic_model.train()
        actor_model_old.eval()
        critic_model_old.eval()
        actor_optimizer, critic_optimizer = optimizer
        sum_loss, sum_reward, data_amount = 0, 0, 0
        for batch_id, data_batch in enumerate(data_loader):
            # loss
            state_batch, ask_price_batch, bid_price_batch = data_batch
            state_batch = state_batch.to(self.device)
            actor_loss, critic_loss, reward = self.calculate_loss(
                model, decision_maker, state_batch, ask_price_batch, bid_price_batch
            )
            # log
            sum_loss += actor_loss.tolist() + critic_loss.tolist()
            sum_reward += reward.tolist()
            data_amount += state_batch.shape[0]
            # optimize
            actor_optimizer.zero_grad()
            actor_loss.backward()
            for param in actor_model.parameters():
                param.grad.data.clamp_(-self.clamp_bound, self.clamp_bound)
            actor_optimizer.step()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            for param in critic_model.parameters():
                param.grad.data.clamp_(-self.clamp_bound, self.clamp_bound)
            critic_optimizer.step()
            # update
            self.update_target_network(actor_model, actor_model_old)
            self.update_target_network(critic_model, critic_model_old)
        sum_loss /= data_amount
        sum_reward /= data_amount
        return sum_loss, sum_reward

    def test_epoch(self, model, data_loader, **kwargs):
        '''Test the model for an epoch.
    
        Args:
            model (A list that contains 4 torch.nn.Module): The neural network model, including actor_model, critic_model, actor_model_old, critic_model_old.
            data_loader (torch.utils.data.DataLoader): The data loader which contains the ``test`` dataset.
            decision_maker (DecisionMakerPPO): The decision maker that chooses actions according to the model\'s output.

        Returns:
            sum_loss, sum_reward (float, float): The average loss and reward of each sequence in the data loader.
        '''
        decision_maker = kwargs["decision_maker"]
        decision_maker.training = False
        actor_model, critic_model, actor_model_old, critic_model_old = model
        actor_model.eval()
        critic_model.eval()
        actor_model_old.eval()
        critic_model_old.eval()
        sum_loss, sum_reward, data_amount = 0, 0, 0
        for batch_id, data_batch in enumerate(data_loader):
            # loss
            state_batch, ask_price_batch, bid_price_batch = data_batch
            state_batch = state_batch.to(self.device)
            actor_loss, critic_loss, reward = self.calculate_loss(
                model, decision_maker, state_batch, ask_price_batch, bid_price_batch
            )
            # log
            sum_loss += actor_loss.tolist() + critic_loss.tolist()
            sum_reward += reward.tolist()
            data_amount += state_batch.shape[0]
        sum_loss /= data_amount
        sum_reward /= data_amount
        return sum_loss, sum_reward
