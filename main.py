import pygsheets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from fastapi import FastAPI, Response
from stable_baselines3 import PPO
from StockTradingEnv import StockTradingEnv

app = FastAPI()
model = PPO.load("stock_trading_ppo_model")
client = pygsheets.authorize(service_file='stock-449613-7413d6080b00.json')
sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/12SzviXwEFOnGIc2j7E-oyQfiwO-PeNlBXW7L3V42dGE/edit')

@app.get("/")
def root():
    return {"message": "Stock trading prediction API is running"}

@app.get("/predict")
def predict_stocks():
    worksheet = sheet[0]
    data = worksheet.get_as_df()
    data = data.dropna(subset=['Price', 'Change %'], how='any')
    numeric_columns = ['Price', 'Change %', 'Volume', 'High', 'Low', 'Open']
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan)
    data.fillna(method='ffill', inplace=True)
    data.fillna(0, inplace=True)
    data['Price_SMA10'] = data['Price'].rolling(window=10).mean().fillna(data['Price'])
    data['Returns'] = data['Price'].pct_change().fillna(0)
    data['Volatility'] = data['Returns'].rolling(window=5).std().fillna(0)
    data[numeric_columns + ['Price_SMA10']] = (
        data[numeric_columns + ['Price_SMA10']] - data[numeric_columns + ['Price_SMA10']].mean()
    ) / data[numeric_columns + ['Price_SMA10']].std()
    env = StockTradingEnv(data)
    obs = env.reset()
    suggestions = []
    risk_factors = []
    for _ in range(len(data)):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        current_price = data['Price'].iloc[env.current_step]
        risk = data['Volatility'].iloc[env.current_step]
        suggestion = "Buy" if action == 1 else "Sell" if action == 2 else "Hold"
        suggestions.append({
            "step": env.current_step,
            "stock": data['SYMBOL'].iloc[env.current_step],
            "suggestion": suggestion,
            "price": current_price,
            "risk": risk
        })
        risk_factors.append({"stock": data['SYMBOL'].iloc[env.current_step], "risk": risk})
        if done:
            break
    return {"suggestions": suggestions, "risk_factors": risk_factors}

@app.get("/graph/{stock_symbol}")
def generate_graph(stock_symbol: str):
    worksheet = sheet[0]
    data = worksheet.get_as_df()
    stock_data = data[data['SYMBOL'] == stock_symbol]
    if stock_data.empty:
        return {"error": "Stock symbol not found"}
    numeric_columns = ['Price', 'Change %', 'Volume', 'High', 'Low', 'Open']
    stock_data[numeric_columns] = stock_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    stock_data.fillna(method='ffill', inplace=True)
    stock_data.fillna(0, inplace=True)
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['Price'], label='Close', color='brown')
    for i in range(len(stock_data)):
        action, _ = model.predict(stock_data.iloc[i].values)
        if action == 1:
            plt.scatter(i, stock_data['Price'].iloc[i], marker='^', color='green', label='Buy signal')
        elif action == 2:
            plt.scatter(i, stock_data['Price'].iloc[i], marker='v', color='red', label='Sell signal')
    plt.title(f"Buy/Sell Signals for {stock_symbol}")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")
