import numpy as np
import datetime
import matplotlib.pyplot as plt
import math

def date_parser(x):
    return datetime.datetime.strptime(x,'%Y-%m-%d')

def get_technical_indicators(dataset, target_col):
    # create 7 and 21 daya moving average
    dataset['ma7'] = dataset[target_col].rolling(window=7).mean()
    dataset['ma21'] = dataset[target_col].rolling(window=21).mean()

    # create MACD: Provides exponential weighted functions.
    dataset['26ema'] = dataset[target_col].ewm(span=26).mean()
    dataset['12ema'] = dataset[target_col].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])

    # create Bollinger Bands
    dataset['20sd'] = dataset[target_col].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + dataset['20sd'] * 2
    dataset['lower_band'] = dataset['ma21'] - dataset['20sd'] * 2

    # create expoential moving average
    dataset['ema'] = dataset[target_col].ewm(com=0.5).mean()

    # create momentum
    dataset['momentum'] = dataset[target_col] - 1
    dataset['log_momentum'] = np.log(dataset[target_col] - 1)
    return dataset

def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0-last_days

    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ =list(dataset.index)

    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'],label='MA 7', color='g',linestyle='--')
    plt.plot(dataset['Close'],label='Closing Price', color='b')
    plt.plot(dataset['ma21'],label='MA 21', color='r',linestyle='--')
    plt.plot(dataset['upper_band'],label='Upper Band', color='c')
    plt.plot(dataset['lower_band'],label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'],label='MACD', linestyle='-.')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(dataset['log_momentum'],label='Momentum', color='b',linestyle='-')

    plt.legend()
    plt.show()

def get_feature_importance_data(data_income):
    data = data_income.copy()
    y = data['Close']
    X = data.iloc[:, 1:]

    train_samples = int(X.shape[0] * 0.65)

    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]

    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]

    return (X_train, y_train), (X_test, y_test)

def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))

def relu(x):
    return max(x, 0)

def lrelu(x):
    return max(0.01*x, x)
