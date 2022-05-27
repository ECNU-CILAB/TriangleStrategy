# -*- coding: utf-8 -*-

'''We provide some basic RNN structural neural network models as function estimators. The backbones of these models are all LSTM. For convenience, we use ``input_size`` to denote the dimension of environment state at one time step. The parameters ``state_num`` and ``action_num`` are set 3 by default and we don\'t support other values now.
'''

import torch


class ModelGinga(torch.nn.Module):
    '''The basic model for DQN, ADQN, PG, PPO(actor), etc. This model contains a multilayer LSTM and a Fully connected layer. The position state will be encoded as a one-hot vector and concatenated to the input tensor, thus the actual input dimension of this model is ``input_size+3``. The output dimension is `3`, which corresponds to the discrete action space $\{-1, 0, 1\}$.
    
    Args:
        input_size (int): The dimension of environment state at one time step.
        state_num (int): The number of position states.
        action_num (int): The number of actions.
        hidden_size (int): The hidden_size in LSTM module.
        num_layers (int): The number of layer in LSTM module.
        dropout (int): The dropout probability of each layer except the last layer in LSTM module.

    .. note::
        There is no activation function (e.g., softmax function in PPO) on the last layer. You must implement it in ``trainer`` if you need.
    '''

    def __init__(self, input_size, state_num=3, action_num=3, hidden_size=32, num_layers=3, dropout=0.2):
        super(ModelGinga, self).__init__()
        self.LSTM_layers = torch.nn.LSTM(
            input_size=input_size+state_num,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.output_layer = torch.nn.Linear(
            hidden_size,
            action_num,
            bias=False
        )

    def forward(self, input_data, info=(None, None)):
        '''Forward function.

        Args:
            input_data (torch.tensor): The environment state at one time step. ``shape=(batch_size, 1, input_size)``
            info ((torch.tensor, torch.tensor)): The hidden state of LSTM module at the last time step. It will be zero (at the first time step) if ``info=(None, None)``.

        Returns:
            model_output, info (torch.tensor, (torch.tensor, torch.tensor)): ``model_output`` is the model\'s output tensor. The shape of ``model_output`` is ``(batch_size, 1, action_num)``. ``info=(h_n, c_n)`` is the hidden state of LSTM module.
        '''
        state_data, (h_0, c_0) = info
        if h_0 is not None and c_0 is not None:
            lstm_out, (h_n, c_n) = self.LSTM_layers(
                torch.concat((input_data, state_data), axis=-1),
                (h_0, c_0)
            )
        else:
            lstm_out, (h_n, c_n) = self.LSTM_layers(
                torch.concat((input_data, state_data), axis=-1)
            )
        model_output = self.output_layer(lstm_out)[:, -1, :]
        info = (h_n, c_n)
        return model_output, info


class ModelGingaStrium(torch.nn.Module):
    '''The basic critic model for PPO(critic), etc. This model contains a multilayer LSTM and a Fully connected layer. The position state will be encoded as a one-hot vector and concatenated to the input tensor, thus the actual input dimension of this model is ``input_size+3``. The output dimension is `1`.

    Args:
        input_size (int): The dimension of environment state at one time step.
        state_num (int): The number of position states.
        hidden_size (int): The hidden_size in LSTM module.
        num_layers (int): The number of layer in LSTM module.
        dropout (int): The dropout probability of each layer except the last layer in LSTM module.
    '''

    def __init__(self, input_size, state_num=3, hidden_size=32, num_layers=3, dropout=0.2):
        super(ModelGingaStrium, self).__init__()
        self.LSTM_layers = torch.nn.LSTM(
            input_size=input_size+state_num,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.output_layer = torch.nn.Linear(
            hidden_size,
            1,
            bias=False
        )

    def forward(self, input_data, info=(None, None)):
        '''Forward function.

        Args:
            input_data (torch.tensor): The environment state at one time step. ``shape=(batch_size, 1, input_size)``
            info ((torch.tensor, torch.tensor)): The hidden state of LSTM module at the last time step. It will be zero (at the first time step) if ``info=(None, None)``.

        Returns:
            model_output, info (torch.tensor, (torch.tensor, torch.tensor)): ``model_output`` is the model\'s output tensor. The shape of ``model_output`` is ``(batch_size, 1, action_num)``. ``info=(h_n, c_n)`` is the hidden state of LSTM module.
        '''
        state_data, (h_0, c_0) = info
        if h_0 is not None and c_0 is not None:
            lstm_out, (h_n, c_n) = self.LSTM_layers(
                torch.concat((input_data, state_data), axis=-1),
                (h_0, c_0)
            )
        else:
            lstm_out, (h_n, c_n) = self.LSTM_layers(
                torch.concat((input_data, state_data), axis=-1)
            )
        model_output = self.output_layer(lstm_out)[:, -1, :]
        info = (h_n, c_n)
        return model_output, info


class ModelVictory(torch.nn.Module):
    '''A basic model designed for A2C methods. This model contains a multilayer LSTM and a Fully connected layer. The position state will be encoded as a one-hot vector and concatenated to the input tensor, thus the actual input dimension of this model is ``input_size+3``. The output includes two parts, i.e., actor value and critic value. The dimensions are 3 and 1 respectively.
    
    Args:
        input_size (int): The dimension of environment state at one time step.
        state_num (int): The number of position states.
        action_num (int): The number of actions.
        hidden_size (int): The hidden_size in LSTM module.
        num_layers (int): The number of layer in LSTM module.
        dropout (int): The dropout probability of each layer except the last layer in LSTM module.

    .. note::
        The LSTM module is shared by both the actor and the critic. But in our experiments, this architecture doesn\'t work well. Another similar model, ``ModelGingaVictory``, is recommended to replace this model.
    '''

    def __init__(self, input_size, state_num=3, action_num=3, hidden_size=32, num_layers=3, dropout=0.2):
        super(ModelVictory, self).__init__()
        self.LSTM_layers = torch.nn.LSTM(
            input_size=input_size+state_num,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.actor_layer = torch.nn.Linear(
            hidden_size,
            action_num,
            bias=False
        )
        self.critic_layer = torch.nn.Linear(
            hidden_size,
            1,
            bias=False
        )

    def forward(self, input_data, info=(None, None)):
        '''Forward function.

        Args:
            input_data (torch.tensor): The environment state at one time step. ``shape=(batch_size, 1, input_size)``
            info ((torch.tensor, torch.tensor)): The hidden state of LSTM module at the last time step. It will be zero (at the first time step) if ``info=(None, None)``.

        Returns:
            model_output, info ((torch.tensor, torch.tensor), (torch.tensor, torch.tensor)): ``model_output=(actor_value, critic_value)`` is the model\'s output tensor. The shape of ``actor_value`` is ``(batch_size, 1, action_num)``, and the shape of ``critic_value`` is ``(batch_size, 1, 1)``. ``info=(h_n, c_n)`` is the hidden state of LSTM module.
        '''
        state_data, (h_0, c_0) = info
        if h_0 is not None and c_0 is not None:
            lstm_out, (h_n, c_n) = self.LSTM_layers(
                torch.concat((input_data, state_data), axis=-1),
                (h_0, c_0)
            )
        else:
            lstm_out, (h_n, c_n) = self.LSTM_layers(
                torch.concat((input_data, state_data), axis=-1)
            )
        model_output = (
            self.actor_layer(lstm_out)[:, -1, :],
            self.critic_layer(lstm_out)[:, -1, :]
        )
        info = (h_n, c_n)
        return model_output, info


class ModelGingaVictory(torch.nn.Module):
    '''A basic model designed for A2C methods. This model contains a multilayer LSTM and a Fully connected layer. The position state will be encoded as a one-hot vector and concatenated to the input tensor, thus the actual input dimension of this model is ``input_size+3``. The output includes two parts, i.e., actor value and critic value. The dimensions are 3 and 1 respectively.
    
    Args:
        input_size (int): The dimension of environment state at one time step.
        state_num (int): The number of position states.
        action_num (int): The number of actions.
        hidden_size (int): The hidden_size in LSTM module.
        num_layers (int): The number of layer in LSTM module.
        dropout (int): The dropout probability of each layer except the last layer in LSTM module.

    .. note::
        The LSTM module is **not** shared by both the actor and the critic, which is the main difference between ``ModelGingaVictory`` and ``ModelVictory``
    '''

    def __init__(self, input_size, state_num=3, action_num=3, hidden_size=32, num_layers=3, dropout=0.2):
        super(ModelGingaVictory, self).__init__()
        self.LSTM_layers_actor = torch.nn.LSTM(
            input_size=input_size+state_num,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.actor_layer = torch.nn.Linear(
            hidden_size,
            action_num,
            bias=False
        )
        self.LSTM_layers_critic = torch.nn.LSTM(
            input_size=input_size+state_num,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.critic_layer = torch.nn.Linear(
            hidden_size,
            1,
            bias=False
        )

    def forward(self, input_data, info=(None, None)):
        '''Forward function.

        Args:
            input_data (torch.tensor): The environment state at one time step. ``shape=(batch_size, 1, input_size)``
            info ((torch.tensor, torch.tensor)): The hidden state of LSTM module at the last time step. It will be zero (at the first time step) if ``info=(None, None)``.

        Returns:
            model_output, info ((torch.tensor, torch.tensor), (torch.tensor, torch.tensor)): ``model_output=(actor_value, critic_value)`` is the model\'s output tensor. The shape of ``actor_value`` is ``(batch_size, 1, action_num)``, and the shape of ``critic_value`` is ``(batch_size, 1, 1)``. ``info=(h_n, c_n)`` is the hidden state of LSTM module.
        '''
        state_data, (hc_actor, hc_critic) = info
        if hc_actor is not None and hc_critic is not None:
            lstm_out_actor, hc_actor = self.LSTM_layers_actor(
                torch.concat((input_data, state_data), axis=-1),
                hc_actor
            )
            lstm_out_critic, hc_critic = self.LSTM_layers_critic(
                torch.concat((input_data, state_data), axis=-1),
                hc_critic
            )
        else:
            lstm_out_actor, hc_actor = self.LSTM_layers_actor(
                torch.concat((input_data, state_data), axis=-1),
            )
            lstm_out_critic, hc_critic = self.LSTM_layers_critic(
                torch.concat((input_data, state_data), axis=-1),
            )
        model_output = (
            self.actor_layer(lstm_out_actor)[:, -1, :],
            self.critic_layer(lstm_out_critic)[:, -1, :]
        )
        info = (hc_actor, hc_critic)
        return model_output, info


class ModelOrb(torch.nn.Module):
    '''A basic model for OASS. This model contains a multilayer LSTM and a Fully connected layer. This model only takes environment states as input. The output at each time step is a 9-dimensional vector, i.e., a 3x3 matrix.
    
    Args:
        input_size (int): The dimension of environment state at one time step.
        hidden_size (int): The hidden_size in LSTM module.
        num_layers (int): The number of layer in LSTM module.
        dropout (int): The dropout probability of each layer except the last layer in LSTM module.
    '''

    def __init__(self, input_size, hidden_size=32, num_layers=3, dropout=0.2):
        super(ModelOrb, self).__init__()
        self.analyze_A_1 = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.actor_layer = torch.nn.Linear(hidden_size, 9, bias=False)

    def forward(self, x):
        '''Forward function.

        Args:
            x (torch.tensor): A batch of sequencial environment states. The shape is ``(batch_size, sequence_length, input_size)``.

        Returns:
            action_prob (torch.tensor): The model\'s output. The shape is ``(batch_size, sequence_length, 9)``.
        
        .. note::
            Before you use the model\'s output, you must reshape the 9-dimensional vector to a 3x3 matrix. ``Softmax`` function is supposed to apply to each row while training. Directly using ``Argmax`` function to select action while evaluating is ok.
        '''
        x = self.analyze_A_1(x)[0]
        action_prob = self.actor_layer(x)
        return action_prob


class ModelGeed(torch.nn.Module):
    '''A basic **actor** model for DDPG. This model contains a multilayer LSTM and a Fully connected layer. Considering that DDPG is a method designed for continuous action space, this model directly outputs a continuous action value in the range (-1, 1).
    
    Args:
        input_size (int): The dimension of environment state at one time step.
        hidden_size (int): The hidden_size in LSTM module.
        num_layers (int): The number of layer in LSTM module.
        dropout (int): The dropout probability of each layer except the last layer in LSTM module.
    '''

    def __init__(self, input_size, hidden_size=32, num_layers=3, dropout=0.2):
        super(ModelGeed, self).__init__()
        self.analyze_A_1 = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.actor_layer = torch.nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        '''Forward function.

        Args:
            x (torch.tensor): A batch of sequencial environment states. The shape is ``(batch_size, sequence_length, input_size)``.

        Returns:
            action (torch.tensor): The model\'s output. The shape is ``(batch_size, sequence_length, 1)``.
        '''
        x = self.analyze_A_1(x)[0]
        x = torch.tanh(self.actor_layer(x))
        return x


class ModelBeliel(torch.nn.Module):
    '''A basic **critic** model for DDPG. This model contains a multilayer LSTM and a Fully connected layer. Similar to GAN, this critic model also takes the actor model\'s output as input. The actual input dimension of this model is ``input_size+1``.
    
    Args:
        input_size (int): The dimension of environment state at one time step.
        hidden_size (int): The hidden_size in LSTM module.
        num_layers (int): The number of layer in LSTM module.
        dropout (int): The dropout probability of each layer except the last layer in LSTM module.
    '''

    def __init__(self, input_size, hidden_size=32, num_layers=3, dropout=0.2):
        super(ModelBeliel, self).__init__()
        self.analyze_A_1 = torch.nn.LSTM(
            input_size=input_size+1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.critic_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, action):
        '''Forward function.

        Args:
            x (torch.tensor): A batch of sequencial environment states. The shape is ``(batch_size, sequence_length, input_size)``.
            action (torch.tensor): The actor model\'s output. The shape is ``(batch_size, sequence_length, 1)``

        Returns:
            critic_value (torch.tensor): The model\'s output. The shape is ``(batch_size, sequence_length, 1)``.
        '''
        x = torch.cat((x, action), dim=-1)
        x = self.analyze_A_1(x)[0]
        x = self.critic_layer(x)
        return x
