import pandas as pd
import yfinance as yf
import numpy as np

# Load CAGR and Volatility data from Excel files
cagr_df = pd.read_excel("CAGRs.xlsx")
volatility_df = pd.read_excel("Volatilities.xlsx")

# Merge the data based on the 'Symbol' column
nifty_data = pd.merge(cagr_df, volatility_df, on='Symbol')

# Get user input for expected return and maximum volatility
user_expected_return = float(input("Enter your expected return (as a decimal, e.g., 0.1 for 10%): "))
user_max_volatility = float(input("Enter the maximum volatility you can afford (as a decimal, e.g., 0.2 for 20%): "))

# Create a function to run the Monte Carlo Simulation
def monte_carlo_simulation(symbol, expected_return, max_volatility, years=1, simulations=10000):
    # Fetch historical data from Yahoo Finance
    stock_data = yf.download(symbol, period=f"{years}y")

    # Calculate daily returns
    daily_returns = stock_data['Adj Close'].pct_change().dropna()

    # Calculate mean daily return and volatility
    mean_daily_return = daily_returns.mean()
    daily_volatility = daily_returns.std()

    # Initialize variables to track acceptable simulations
    acceptable_simulations = []

    # Run Monte Carlo simulations
    for _ in range(simulations):
        # Generate a random set of daily returns based on mean and volatility
        random_returns = np.random.normal(mean_daily_return, daily_volatility, 252)

        # Calculate the annual return and volatility
        annual_return = (1 + random_returns).prod() - 1
        annual_volatility = random_returns.std() * np.sqrt(252)

        # Check if the simulation meets the user's criteria
        if annual_return >= expected_return and annual_volatility <= max_volatility:
            acceptable_simulations.append((annual_return, annual_volatility))

    return acceptable_simulations

# Run Monte Carlo Simulation for each stock
acceptable_stocks = {}
for _, row in nifty_data.iterrows():
    symbol = row['Symbol']
    cagr = row['1Y CAGR']
    volatility = row['Volatility']

    simulations = monte_carlo_simulation(symbol, cagr, volatility, years=1)

    if simulations:
        acceptable_stocks[symbol] = simulations

# Sort stocks by the number of acceptable simulations
sorted_stocks = sorted(acceptable_stocks.items(), key=lambda x: len(x[1]), reverse=True)

# Get the top 5 stocks with the most acceptable simulations
top_5_stocks = sorted_stocks[:5]

print("Top 5 Suggested Stocks:")
for stock, simulations in top_5_stocks:
    print(f"Symbol: {stock}")
    print(f"Number of Acceptable Simulations: {len(simulations)}")
    print("Average Return and Volatility:")
    avg_return = np.mean([sim[0] for sim in simulations])
    avg_volatility = np.mean([sim[1] for sim in simulations])
    print(f"Expected Return: {avg_return:.4f}")
    print(f"Volatility: {avg_volatility:.4f}")
    print()

# Save the combined CAGR and volatility data to a text file
combined_data = [{'Symbol': stock, '1Y CAGR': nifty_data[nifty_data['Symbol'] == stock]['1Y CAGR'].values[0], 'Volatility': nifty_data[nifty_data['Symbol'] == stock]['Volatility'].values[0]} for stock, _ in top_5_stocks]
with open("Combined_CAGR_Volatility.txt", "w") as file:
    for item in combined_data:
        file.write(f"Symbol: {item['Symbol']}\n")
        file.write(f"1Y CAGR: {item['1Y CAGR']:.4f}\n")
        file.write(f"Volatility: {item['Volatility']:.4f}\n\n")

print("Combined CAGR and Volatility data saved to Combined_CAGR_Volatility.txt")
