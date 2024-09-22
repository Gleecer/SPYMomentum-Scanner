import yfinance as yf
import pandas as pd
import numpy as np
import praw 
from talib import RSI, BOP, MACD, ADX
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from datetime import datetime


# Reddit API credentials (replace with your own)
reddit = praw.Reddit(
    client_id='B_3OP6wtYHBpgcxthSTyVQ',
    client_secret='BJC_8XZL1UHQp1Y2GTFDGiPcBPv1Qw',
    user_agent='Trying something new/0.1 by Quanteroooooni'
)

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table['Symbol'].tolist()

def get_stock_data(ticker, period="1d"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    data["Stock Splits"] = data["Stock Splits"].fillna(0)
    return data

def calculate_rsi(data, window=14):
    close_data = data['Close'].copy()
    return RSI(close_data, timeperiod=window)

def calculate_bop(data):
    return BOP(data['Open'], data['High'], data['Low'], data['Close'])

def analyze_volume(data, short_window=5, long_window=20):
    short_vol_ma = data['Volume'].rolling(window=short_window).mean()
    long_vol_ma = data['Volume'].rolling(window=long_window).mean()
    return short_vol_ma / long_vol_ma

def analyze_price_trend(data, window=20):
    ma = data['Close'].rolling(window=window).mean()
    return (data['Close'] > ma).astype(int)

def calculate_macd(data):
    macd, signal, _ = MACD(data['Close'])
    return macd - signal

def calculate_adx(data, period=14):
    return ADX(data['High'], data['Low'], data['Close'], timeperiod=period)

def analyze_buy_volume(data, window=5):
    data['Buy_Volume'] = np.where(data['Close'] > data['Open'], data['Volume'], 0)
    return data['Buy_Volume'].rolling(window=window).mean() / data['Volume'].rolling(window=window).mean()

def get_reddit_sentiment(ticker):
    # Implement Reddit sentiment analysis here
    # This is a placeholder function
    return np.random.uniform(-1, 1)

def calculate_weekly_metrics(data):
    weekly_data = data.resample('W').agg({
        'Open': 'first', 
        'High': 'max', 
        'Low': 'min', 
        'Close': 'last',
        'Volume': 'sum'
    })
    
    weekly_rsi = calculate_rsi(weekly_data)
    weekly_bop = calculate_bop(weekly_data)
    weekly_volume_ratio = analyze_volume(weekly_data)
    weekly_price_trend = analyze_price_trend(weekly_data)
    weekly_macd = calculate_macd(weekly_data)
    weekly_adx = calculate_adx(weekly_data)
    weekly_buy_volume_ratio = analyze_buy_volume(weekly_data)
    
    return {
        'RSI': weekly_rsi,
        'BOP': weekly_bop,
        'Volume Ratio': weekly_volume_ratio,
        'Price Trend': weekly_price_trend,
        'MACD': weekly_macd,
        'ADX': weekly_adx,
        'Buy Volume Ratio': weekly_buy_volume_ratio
    }

def grade_opportunity(ticker):
    data = get_stock_data(ticker, period="1mo")  # Increased period to 1 month
    
    rsi = calculate_rsi(data)
    bop = calculate_bop(data)
    volume_ratio = analyze_volume(data)
    price_trend = analyze_price_trend(data)
    sentiment = get_reddit_sentiment(ticker)
    macd = calculate_macd(data)
    adx = calculate_adx(data)
    buy_volume_ratio = analyze_buy_volume(data)
    
    weekly_metrics = calculate_weekly_metrics(data)
    
    current_price = data['Close'].iloc[-1]
    
    df = pd.DataFrame({
        'Metric': ['Price', 'RSI', 'BOP', 'Volume Ratio', 'Price Trend', 'MACD', 'ADX', 'Buy Volume Ratio'],
        'Current': [current_price, rsi.iloc[-1], bop.iloc[-1], volume_ratio.iloc[-1], 
                    price_trend.iloc[-1], macd.iloc[-1], adx.iloc[-1], buy_volume_ratio.iloc[-1]],
        'Weekly High': [data['High'].tail(5).max()] + [metric.tail(5).max() for metric in weekly_metrics.values()],
        'Weekly Low': [data['Low'].tail(5).min()] + [metric.tail(5).min() for metric in weekly_metrics.values()]
    })
    
    for col in ['Current', 'Weekly High', 'Weekly Low']:
        df[col] = df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (float, np.float64)) else x)
    
    grade = (
        (float(df.loc[df['Metric'] == 'RSI', 'Current'].iloc[0]) / 100) * 0.15 +
        (float(df.loc[df['Metric'] == 'BOP', 'Current'].iloc[0]) + 1) / 2 * 0.1 +
        float(df.loc[df['Metric'] == 'Volume Ratio', 'Current'].iloc[0]) * 0.15 +
        float(df.loc[df['Metric'] == 'Price Trend', 'Current'].iloc[0]) * 0.05 +
        (sentiment + 1) / 2 * 0.1 +
        (float(df.loc[df['Metric'] == 'MACD', 'Current'].iloc[0]) > 0) * 0.05 +
        (float(df.loc[df['Metric'] == 'ADX', 'Current'].iloc[0]) > 25) * 0.15 +
        float(df.loc[df['Metric'] == 'Buy Volume Ratio', 'Current'].iloc[0]) * 0.25
    ) * 5
    
    return round(grade, 2)

def estimate_timeframe(data):
    # Implement your timeframe estimation logic here
    # This is a placeholder function
    return "1-3 months"

def get_market_cap(ticker):
    stock = yf.Ticker(ticker)
    return stock.info.get('marketCap', 0)

def calculate_size_weight(market_cap):
    if market_cap < 2e9:  # Small Cap
        return 0.8
    elif market_cap < 10e9:  # Mid Cap
        return 1.0
    else:  # Large Cap
        return 1.2

def analyze_stock(ticker):
    try:
        grade = grade_opportunity(ticker)
        timeframe = estimate_timeframe(get_stock_data(ticker))
        market_cap = get_market_cap(ticker)
        size_weight = calculate_size_weight(market_cap)
        weighted_grade = grade * size_weight
        return (ticker, grade, weighted_grade, timeframe, market_cap)
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        return None

def main():
    print("Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    
    print(f"Analyzing {len(tickers)} stocks...")
    results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(analyze_stock, ticker): ticker for ticker in tickers}
        for future in tqdm(as_completed(future_to_ticker), total=len(tickers)):
            result = future.result()
            if result:
                results.append(result)
    
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)  # Sort by weighted grade
    
    top_tickers = sorted_results[:3]  # Get the top 3 tickers
    
    # Print the results
    print(f"\nTop 3 Opportunities (Size-Weighted) as of {time.strftime('%Y-%m-%d %H:%M:%S')}:")
    print("=" * 80)
    print(f"{'Ticker':<10}{'Raw Grade':<15}{'Weighted Grade':<20}{'Timeframe':<15}{'Market Cap':<20}")
    print("-" * 80)
    for ticker, grade, weighted_grade, timeframe, market_cap in top_tickers:
        print(f"{ticker:<10}{grade:<15.2f}{weighted_grade:<20.4f}{timeframe:<15}{market_cap:,.0f}")
    print("=" * 80)

    # Save top 3 results to CSV
    csv_file_path = 'topmom_tickers.csv'
    
    if not os.path.isfile(csv_file_path):
        # If the file does not exist, create it and write the header
        with open(csv_file_path, 'w') as f:
            f.write('Ticker,Raw Grade,Weighted Grade,Timeframe,Market Cap\n')
    

    with open(csv_file_path, 'a') as f:
        for ticker, grade, weighted_grade, timeframe, market_cap in top_tickers:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get the current date and time
            f.write(f'{ticker},{grade},{weighted_grade},{timeframe},{timestamp}\n')
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")