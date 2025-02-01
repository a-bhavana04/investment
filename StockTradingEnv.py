import gym
from gym import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.balance = 100000
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_steps = len(self.data) - 1
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = 100000
        self.shares_held = 0
        self.net_worth = self.balance
        return self._get_observation()

    def _get_observation(self):
        obs = self.data[['Price', 'Change %', 'Volume', 'High', 'Low', 'Volatility', 'Price_SMA10']].iloc[self.current_step].values
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.data['Price'].iloc[self.current_step]
        if action == 1 and self.balance >= current_price:
            self.shares_held += 1
            self.balance -= current_price
        elif action == 2 and self.shares_held > 0:
            self.shares_held -= 1
            self.balance += current_price
        self.net_worth = self.balance + (self.shares_held * current_price)
        reward = (self.net_worth - 100000) / 100000
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self._get_observation(), reward, done, {}
