import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from yfinance_dataFetch import StockDataFetcher

def fetch_and_prepare_data(file_path, start_date, end_date, symbols=None):
    """Fetches and prepares stock data."""
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, index_col="Date", keep_default_na=True)
    else:
        fetcher = StockDataFetcher(start_date=start_date, end_date=end_date, symbols=symbols)
        data = fetcher.fetch_data()
        data.dropna(axis=1, how='all', inplace=True)  # Drop columns with all missing values
        data.dropna(inplace=True)  # Drop rows with any missing values
        data.index.name = 'Date'
        fetcher.save_to_csv(data, file_path, index=True)
    return data

def plot_explained_variance(pca, num_components):
    """Plots explained variance and cumulative explained variance."""
    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, num_components + 1), explained_variance, alpha=0.6, label='Explained Variance')
    plt.step(range(1, num_components + 1), cumulative_explained_variance, where='mid', label='Cumulative Explained Variance', color='red')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Variance')
    plt.title('Explained Variance by Principal Components')
    plt.xticks(range(1, num_components + 1))
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.show()

def perform_pca_and_regression(ret1, retFut1, num_factors, top_n, lookback):
    """Performs PCA and regression to determine positions."""
    positions = np.zeros(ret1.shape)
    retPred1 = np.full((1, ret1.shape[1]), np.nan)

    for t in range(lookback, len(ret1)):
        print(f"on day {t}")
        trainset = list(range(t - lookback + 1, t + 1))
        R = ret1.iloc[trainset].copy()
        hasData = np.where(np.all(np.isfinite(R), axis=0))[0]

        if len(hasData) > 2 * top_n:
            R_clean = R.iloc[:, hasData]
            pca = PCA(n_components=num_factors)
            factors = pca.fit_transform(R_clean)

            if t == lookback:
                plot_explained_variance(pca, num_factors)

            for s in range(len(hasData)):
                model = LinearRegression()
                model.fit(factors[:-1, :], retFut1.iloc[trainset, hasData[s]].values[:-1])
                retPred1[0, hasData[s]] = model.predict(factors[-1, :].reshape(1, -1))[0]

            isGoodData = np.where(np.isfinite(retPred1))[1]
            sorted_indices = np.argsort(retPred1[0, isGoodData])
            positions[t, isGoodData[sorted_indices[:top_n]]] = -1
            positions[t, isGoodData[sorted_indices[-top_n:]]] = 1

    return positions

def calculate_max_dd(cumret):
    """Calculates the maximum drawdown and duration."""
    high_water_mark = np.maximum.accumulate(cumret)
    drawdowns = (1 + cumret) / (1 + high_water_mark) - 1
    max_dd = np.min(drawdowns)
    end_idx = np.argmin(drawdowns)
    start_idx = np.argmax(cumret[:end_idx] == high_water_mark[:end_idx])
    return max_dd, end_idx - start_idx

def main():
    """Main function to execute the analysis."""
    cl500 = fetch_and_prepare_data('sp500_closing_prices_last_5_years.csv', "2019-01-01", "2024-08-01", ["SPX500"])
    cl500.index = pd.to_datetime(cl500.index.astype(str), format='%d/%m/%Y')
    cl500 = cl500[cl500.index.year >= 2019]

    ret1 = cl500.pct_change(fill_method=None)
    retFut1 = ret1.shift(-1)

    lookback = 252
    topN = 10
    numFactors = 5

    positions = perform_pca_and_regression(ret1, retFut1, numFactors, topN, lookback)

    shifted_positions = np.vstack((np.full((1, positions.shape[1]), np.nan), positions[:-1]))
    daily_ret = np.nansum(shifted_positions * ret1, axis=1) / np.nansum(np.abs(shifted_positions), axis=1)
    daily_ret[~np.isfinite(daily_ret)] = 0

    cumret = np.cumprod(1 + daily_ret) - 1
    testset_start = np.min(np.where(np.any(positions != 0, axis=1)))
    testset = range(testset_start, len(cl500))

    dates = pd.to_datetime(cl500.index[testset].astype(str), format='%Y-%m-%d')
    plt.plot(dates, cumret[testset])
    plt.title('Statistical factor prediction: Out-of-sample')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.show()

    cagr = np.prod(1 + daily_ret[testset]) ** (252 / len(daily_ret[testset])) - 1
    sharpe_ratio = np.sqrt(252) * np.mean(daily_ret[testset]) / np.std(daily_ret[testset])

    print(f'CAGR={cagr:.6f} Sharpe={sharpe_ratio:.6f}')

    max_dd, max_ddd = calculate_max_dd(cumret)
    calmar_ratio = -cagr / max_dd

    print(f'maxDD={max_dd:.6f} maxDDD={max_ddd} Calmar ratio={calmar_ratio:.6f}')

if __name__ == "__main__":
    main()