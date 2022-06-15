.. TriangleStrategy documentation master file, created by
   sphinx-quickstart on Tue May 24 10:47:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TriangleStrategy's documentation!
============================================

.. toctree::
   :maxdepth: 4
   :caption: API Docs

   TriangleStrategy

.. * :ref:`modindex`

Project Description
===================

TriangleStrategy is a high-efficiency reinforcement learning based algorithmic trading library. We provide examples of mainstream general-purpose reinforcement learning algorithms, as well as several reinforcement learning algorithms designed specifically for algorithmic trading. We currently support the following methods:

* General reinforcement learning methods

    `PG <https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf>`_

    `A2C <https://openai.com/blog/baselines-acktr-a2c/>`_

    `DQN <https://www.nature.com/articles/nature14236>`_

    `DDPG <https://arxiv.org/pdf/1509.02971.pdf>`_

    `PPO <https://arxiv.org/pdf/1707.06347.pdf>`_
* Specifical reinforcement learning methods designed for algorithmic trading

    `DQN-A <https://arxiv.org/pdf/1807.02787>`_

    OASS (Our proposed method, coming sooooooooooon...)

Note that OASS is based on another individual package ``oass``. Please refer to https://pypi.org/project/oass/.

This project is open source and is jointly supported by **East China Normal University** and **Seek Data Group, Emoney Inc.**.

1. Project Architecture
-----------------------

TriangleStrategy implements algorithmic trading strategies based on reinforcement learning primarily through the following modules:

* Environment
* ModelArchitecture
* Methods (DecisionMaker, Trainer)
* TrainFunctions

1.1 Environment: reward calculater
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, we only provide a generic environment that is used to receive actions and calculate rewards to guide model training.

The environment supports two action spaces:

* Discrete action space {-1, 0, 1}
* Continuous action space [-1, 1]

The action value is the signal to adjust the position. For example, when action 1 is selected, the agent will hold 1 unit of the financial product but not buy 1 unit of the financial product. This design prevents the agent from overbuying a financial product, which can lead to significant risk. Negative action values are allowed in both the discrete and continuous action spaces. When action -1 is selected, the agent will adjust to a state of holding -1 units of the financial product. In other words, the agent sells 1 unit of the financial product in the underlying position and earns a gain from the fall in price if it can buy it back in the future at a lower price. The negative part of the action space allows the model to better perceive price declines.

The calculation of the reward value provides both the amount of change in assets and the amount of change in money. The difference is that the latter does not include the value of the financial product and is negative when buying and positive when selling. If the amount of change in money is used, then obviously agent only needs to sell to get a high reward. to avoid this, when using the discrete action space, the first and last action in the sequence must be 0. For the continuous action space, we do not make a similar restriction for now, but we still recommend to implement it manually when using it.

1.2 ModelArchitecture: basic neural network models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model structures in reinforcement learning algorithms are independent of the algorithm, so theoretically any neural network model structure can be used as long as the input and output layers are in corresponding formats, and we provide several basic model structures. The main part of these model structures are all multilayer LSTMs, which are identical except for the input and output layers.

1.3 Methods: reinforcement learning methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each method is implemented using ``DecisionMaker`` and ``Trainer``.

``DecisionMaker`` is used to generate action values based on the output values of the model. There are two parameters in each ``DecisionMaker``: ``head_mask``, ``tail_mask``. Considering the structure of the LSTM model, there is not enough historical information for the model to make stable decisions in the first few time steps of the sequence, so ``DecisionMaker`` forces the action value of ``i<head_mask`` to 0. On the other hand, in the last few time steps of the sequence, there is not enough future information for the environment to On the other hand, in the last few time steps of the sequence, there is not enough future information for the environment to generate a reward, so ``DecisionMaker`` forces the action value of ``i>=tail_mask`` to 0. In addition, ``DecisionMaker`` can be used both for training and evaluation, and can be switched by controlling the ``training`` parameter in it.

``Trainer`` will call ``model``, ``DecisionMaker``, ``optimizer``, and ``data_loader`` for training. If you want to design a new reinforcement learning method, this part will be the core part. The design of ``Trainer`` is very free, and the few methods of ``Trainer`` in this library do not even need to depend on Environment. This can significantly improve the training efficiency.

1.4 TrainFunctions: Functions used to assist in training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In our experiments, if trained directly, many methods would guide the model to produce 0 decisions, since doing so would clearly hedge the risks in financial markets. To avoid this, we borrow the idea of curriculum learning. ``TrainFunctions`` provides functions for curriculum learning, which controls the ``difficulty`` parameter in ``Trainer``, and thus the transaction cost when the model trades. The model is first trained in low transaction cost, and then the transaction cost is gradually increased to reach the real transaction cost.

2. Code examples
----------------

Take DQN as an example

.. code-block:: python

    import torch
    import TriangleStrategy as ts


    # device
    device = torch.device("cuda", 0)
    # data loader
    data_loader_train = ts.get_DataLoader(
        "normalized_train_data.npz",
        batch_size=512
    )
    data_loader_dev = ts.get_DataLoader(
        "normalized_dev_data.npz",
        batch_size=512
    )
    data_loader_test = ts.get_DataLoader(
        "normalized_test_data.npz",
        batch_size=512
    )
    # model
    model = ts.ModelGinga(input_size=7).to(device)
    model_target = ts.ModelGinga(input_size=7).to(device)
    model_target.load_state_dict(model.state_dict())
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # trainer (Note that the environment is included in the trainer.)
    trainer = ts.TrainerDQN(
        head_mask=20,
        tail_mask=120,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001
    )
    # At the begining of training process, use a decision maker that chooses actions randomly.
    ts.train_model_until_converge(
        model, data_loader_train, data_loader_dev, data_loader_test, optimizer, trainer,
        decision_maker=ts.DecisionMakerDQN(epsilon=1.0, head_mask=20, tail_mask=120),
        model_target=model_target,
        epoch_max=100,
        epoch_window=30,
        epoch_window_win=16
    )
    # Then increase the difficulty linearly.
    ts.heuristic_train_model(
        model, data_loader_train, data_loader_dev, data_loader_test, optimizer, trainer,
        decision_maker=ts.DecisionMakerDQN(epsilon=0.01, head_mask=20, tail_mask=120),
        model_target=model_target,
        steps=10,
        epoch_max=100,
        epoch_window=30,
        epoch_window_win=16
    )

3. Datasets
-----------

The dataset is stored in ``.npz`` format. If you want to build the dataset yourself using another data source, use the ``numpy.savez_compressed`` function to get the ``.npz`` file. The data file nust contain the following three parts:

* ``input_data``: Environment states at each time step. This part is the representation of the market condition and is inputed to the model. The shape is ``(sequence_num, sequence_length, input_size)``.
* ``buy_price``: The transaction price when the agent chooses to buy at the corresponding time step. The shape is ``(sequence_num, sequence_length)``.
* ``sell_price``: The transaction price when the agent chooses to sell at the corresponding time step. The shape is ``(sequence_num, sequence_length)``.

Usually the ``buy_price`` is higher than the ``sell_price``, but you can use the same values.

In addition, we provide two datasets collected from China stock market.

* **TriangleStrategy-minute** is high-frequency limited order book (LOB) data, including the minute-level price and volume information. The ask price and bid price are used as the transaction price. There are 240 time steps in each sequence, which corresponds to 240 minutes in a trading day.
* **TriangleStrategy-day** is low-frequency data, including day-level market information. The close price every day is used as the transaction price. Considering that the data amount of low-frequency trading is much smaller than that of high-frequency trading, we collect low-frequency trading data over a larger time span. There are 140 time steps in each sequence, which corresponds to 140 continuous trading days.

To gain access to the dataset, please refer to http://www.seek-data.com/research.html.
