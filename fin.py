import pygsheets
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from StockTradingEnv import StockTradingEnv 

client = pygsheets.authorize(service_file='stock-449613-7413d6080b00.json')
sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/12SzviXwEFOnGIc2j7E-oyQfiwO-PeNlBXW7L3V42dGE/edit')
worksheet = sheet[0]

data = worksheet.get_as_df()
data = data.dropna(subset=['Price', 'Change %'])
numeric_columns = ['Price', 'Change %', 'Volume', 'High', 'Low', 'Open']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
data = data.replace([np.inf, -np.inf], 0).fillna(0)
data['Price_SMA10'] = data['Price'].rolling(window=10).mean().fillna(data['Price'])
data['Returns'] = data['Price'].pct_change().fillna(0)
data['Volatility'] = data['Returns'].rolling(window=5).std().fillna(0)
data[numeric_columns + ['Price_SMA10']] = (
    data[numeric_columns + ['Price_SMA10']] - data[numeric_columns + ['Price_SMA10']].mean()
) / data[numeric_columns + ['Price_SMA10']].std()

env = StockTradingEnv(data)

eval_callback = EvalCallback(
    env,
    best_model_save_path='./best_model',
    log_path='./logs',
    eval_freq=100,
    n_eval_episodes=5,
    callback_on_new_best=StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=2),
)

model = PPO('MlpPolicy', env, learning_rate=0.0001, gamma=0.98, verbose=1, n_steps=32)
model.learn(total_timesteps=10000, callback=eval_callback)
model.save("stock_trading_ppo_model")
