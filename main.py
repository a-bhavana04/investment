import pygsheets
import pandas as pd
import numpy as np
from flask import Flask, jsonify
from stable_baselines3 import PPO
from StockTradingEnv import StockTradingEnv

# Initialize Flask app
app = Flask(__name__)

# Load model and authenticate Google Sheets
model = PPO.load("stock_trading_ppo_model")
client = pygsheets.authorize(service_file='stock-449613-7413d6080b00.json')
sheet = client.open_by_url('https://docs.google.com/spreadsheets/d/12SzviXwEFOnGIc2j7E-oyQfiwO-PeNlBXW7L3V42dGE/edit')

@app.route("/", methods=["GET"])
def root():
    """Root endpoint to check if the API is running."""
    return jsonify({"message": "Stock trading prediction API is running"}), 200

@app.route("/predict", methods=["GET"])
def predict_stocks():
    """Predict stock trading actions."""
    try:
        worksheet = sheet[0]
        data = worksheet.get_as_df()

        # Preprocess data
        data = data.dropna(subset=['Price', 'Change %'], how='any')
        numeric_columns = ['Price', 'Change %', 'Volume', 'High', 'Low', 'Open']
        data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
        data = data.replace([np.inf, -np.inf], np.nan)
        data.fillna(method='ffill', inplace=True)
        data.fillna(0, inplace=True)

        # Feature engineering
        data['Price_SMA10'] = data['Price'].rolling(window=10).mean().fillna(data['Price'])
        data['Returns'] = data['Price'].pct_change().fillna(0)
        data['Volatility'] = data['Returns'].rolling(window=5).std().fillna(0)

        # Normalize numeric columns
        data[numeric_columns + ['Price_SMA10']] = (
            data[numeric_columns + ['Price_SMA10']] - data[numeric_columns + ['Price_SMA10']].mean()
        ) / data[numeric_columns + ['Price_SMA10']].std()

        # Initialize the stock trading environment
        env = StockTradingEnv(data)
        obs = env.reset()

        suggestions = []
        risk_factors = []

        # Generate predictions for each step
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

        return jsonify({"suggestions": suggestions, "risk_factors": risk_factors}), 200

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
