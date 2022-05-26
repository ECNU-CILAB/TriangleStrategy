# -*- coding: utf-8 -*-

'''We provide an environment that is slightly different from most general-purpose learning frameworks. To improve performance, this environment computes rewards for the entire sequence, rather than by frequent interactions.
'''

class EnvX():
    '''Simulated profit environment

    Attributes:
        allow_continuous_action (bool): If ``allow_continuous_action=True``, any action values within the range [-1,1] will be allowed. If ``allow_continuous_action=False``, action values can only be -1, 0 or 1, additionally, the first and the last action value must be 0.
        use_price_diff (bool): If ``use_price_diff=True``, the reward values represent the amount of change in the asset. If ``use_price_diff=False``, the reward values represent the amount of change in the money (it doesn\'t include the value of financial products).
    '''
    def __init__(self, allow_continuous_action=False, use_price_diff=True):
        '''Initialization
        '''
        self.allow_continuous_action = allow_continuous_action
        self.use_price_diff = use_price_diff
    
    def simulate_trade(
        self,
        position_signal_series,
        buy_price_series,
        sell_price_series,
        buy_cost_pct,
        sell_cost_pct
    ):
        '''Simulate the trading process. In this function, we construct buy-sell behaviours according to ``position_signal_series``. the agent will buy financial products if ``position_signal_series[i]>position_signal_series[i-1]`` and sell financial products otherwise.
        
        Args:
            position_signal_series (List or np.array): A series represents the position action.
            buy_price_series (List or np.array): A series represents the transaction price when the agent chooses to buy at the corresponding time step.
            sell_price_series (List or np.array): A series represents the transaction price when the agent chooses to sell at the corresponding time step.
            buy_cost_pct (float): The cost ratio when the agent chooses to buy.
            sell_cost_pct (float): The cost ratio when the agent chooses to sell.
        
        Returns:
            reward_series (List): The normalized reward value series.
        '''
        money = 0
        asset_list = [money]
        last_position_signal = 0
        for position_signal, buy_price, sell_price in zip(
            position_signal_series, buy_price_series, sell_price_series
        ):
            if self.allow_continuous_action:
                if position_signal<-1.001 or position_signal>1.001:
                    raise ValueError("position_signal must be in [-1, 1].")
            else:
                if position_signal != -1 and position_signal != 0 and position_signal != 1:
                    raise ValueError("position_signal must be -1,0,1.")
            position_delta = position_signal-last_position_signal
            if position_delta > 0:
                money = money-position_delta*buy_price
                money = money-position_delta*buy_price*buy_cost_pct
            else:
                position_delta = -position_delta
                money = money+position_delta*sell_price
                money = money-position_delta*sell_price*sell_cost_pct
            last_position_signal = position_signal
            asset_mow = money
            if self.use_price_diff:
                asset_mow += last_position_signal*(buy_price+sell_price)/2
            asset_list.append(asset_mow)
        initial_price = (buy_price_series[0]+sell_price_series[0])/2
        asset_list = [i/initial_price for i in asset_list]
        reward_series = [i-j for i, j in zip(asset_list[1:], asset_list[:-1])]
        return reward_series
