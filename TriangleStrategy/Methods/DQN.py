# -*- coding: utf-8 -*-

'''DQN (Deep Q network).
'''

from ..Environment import EnvX
import torch
import numpy as np


class DecisionMakerDQN():
    '''The decision maker of DQN.
    
    Args:
        head_mask (int): ``head_mask``.
        tail_mask (int): ``tail_mask``.
        epsilon (float): The probability that the agent chooses actions randomly.
        device (torch.device): The computing device.

    Attributes:
        epsilon (float): The probability that the agent chooses actions randomly. We recommend setting it to 1 initially and then decreasing it to 0.01.
        training (bool): If ``training=True``, ``model.train()`` will be called and epsilon-greedy strategy will be enabled. If ``training=False``, ``model.eval()`` will be called and epsilon-greedy strategy will be disabled.
    '''
    def __init__(self, head_mask, tail_mask, epsilon=0.01, device=torch.device("cuda")):
        self.epsilon = epsilon
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
                    action = torch.argmax(q_value, dim=-1, keepdim=False).tolist()
                    if self.training:
                        for seq_id in range(seq_num):
                            if np.random.random() < self.epsilon:
                                action[seq_id] = np.random.randint(3)
                    action = torch.Tensor(action).to(torch.int64).to(self.device)
                signal_series.append(action)
            signal_series = torch.stack(signal_series, dim=1)
            return signal_series


class TrainerDQN():
    '''The trainer of DQN.
    
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
        self.env = EnvX()
        self.difficulty = 1.0
        self.device = device

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
        # q_value
        q = decision_maker.get_model_output(
            model, input_data_batch, action_batch)
        with torch.no_grad():
            q_target = decision_maker.get_model_output(
                model_target, input_data_batch, action_batch)
        # mask unchoosed actions
        mask = torch.nn.functional.one_hot(
            action_batch, num_classes=3).to(self.device)
        q_now = torch.sum(
            q*mask, dim=2, keepdim=False
        )[:, self.head_mask:self.tail_mask]
        # q_future=r_i+max_a Q(s',a)
        q_future = torch.max(
            q_target.detach(), dim=2, keepdim=False
        )[0][:, self.head_mask+1:self.tail_mask+1]
        q_future = reward_action[:, self.head_mask:self.tail_mask] \
            + q_future*self.gamma
        # loss and reward
        loss_value = self.loss_fn(q_now, q_future)
        reward_value = torch.sum(reward_action)
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
            decision_maker (DecisionMakerDQN): The decision maker that chooses actions according to the model\'s output.
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
            sum_reward += reward.tolist()
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
            decision_maker (DecisionMakerDQN): The decision maker that chooses actions according to the model\'s output.
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
                sum_reward += reward.tolist()
                data_amount += state_batch.shape[0]
            sum_loss /= data_amount
            sum_reward /= data_amount
        return sum_loss, sum_reward
