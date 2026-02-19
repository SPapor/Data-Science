import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.seasonal import seasonal_decompose

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

sns.set_style("whitegrid")
plt.rcParams.update({"figure.figsize": (14, 6)})

try:
    df = pd.read_csv("usd_uah.csv")
except FileNotFoundError:
    exit("Файл не знайдено.")

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date').sort_index()
df.columns = ['value']

def remove_outliers(series, window=30, sigma=3):
    roll_mean = series.rolling(window=window, center=True).mean()
    roll_std = series.rolling(window=window, center=True).std()
    outliers = (series > roll_mean + sigma * roll_std) | (series < roll_mean - sigma * roll_std)
    clean = series.copy()
    clean[outliers] = np.nan
    return clean.interpolate(method='time').fillna(series)

df['value_clean'] = remove_outliers(df['value'])

plt.figure()
plt.plot(df.index, df['value'], label='Оригінальні дані', alpha=0.5, color='gray')
plt.plot(df.index, df['value_clean'], label='Очищені дані', color='blue')
plt.title('Очищення часового ряду від аномалій')
plt.legend()
plt.show()

try:
    decomp = seasonal_decompose(df['value_clean'].dropna(), model='additive', period=30)
    fig = decomp.plot()
    fig.set_size_inches(14, 8)
    plt.show()
except Exception as e:
    pass

train_size = int(len(df) * 0.85)
train, test = df['value_clean'].iloc[:train_size], df['value_clean'].iloc[train_size:]

model_arima = auto_arima(train, seasonal=False, trace=False, error_action='ignore')
arima_forecast = pd.Series(model_arima.predict(n_periods=len(test)), index=test.index)

X_train = np.arange(len(train)).reshape(-1, 1)
X_test = np.arange(len(train), len(df)).reshape(-1, 1)
poly = PolynomialFeatures(degree=3)
model_poly = LinearRegression().fit(poly.fit_transform(X_train), train.values)
poly_forecast = pd.Series(model_poly.predict(poly.transform(X_test)), index=test.index)

lstm_forecast = None
if TF_AVAILABLE:
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    SEQ_LEN = 30

    def create_sequences(data, seq_len):
        xs, ys = [], []
        for i in range(len(data) - seq_len):
            xs.append(data[i:(i + seq_len)])
            ys.append(data[i + seq_len])
        return np.array(xs), np.array(ys)

    X_train_lstm, y_train_lstm = create_sequences(train_scaled, SEQ_LEN)

    model_lstm = Sequential([
        LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
        Dropout(0.2),
        LSTM(50),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=0)

    inputs = df['value_clean'].values[len(df) - len(test) - SEQ_LEN:]
    inputs = scaler.transform(inputs.reshape(-1, 1))
    X_test_lstm = []
    for i in range(SEQ_LEN, len(inputs)):
        X_test_lstm.append(inputs[i - SEQ_LEN:i, 0])
    X_test_lstm = np.array(X_test_lstm).reshape(-1, SEQ_LEN, 1)

    pred_lstm = model_lstm.predict(X_test_lstm, verbose=0)
    lstm_forecast = pd.Series(scaler.inverse_transform(pred_lstm).flatten(), index=test.index)

def show_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name:20} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

print("\nРезультати:")
show_metrics(test, arima_forecast, "ARIMA")
show_metrics(test, poly_forecast, "Polynomial Reg")
if lstm_forecast is not None:
    show_metrics(test, lstm_forecast, "LSTM")


plt.figure()
plt.plot(train.index, train, label='Навчальні дані', alpha=0.4)
plt.plot(test.index, test, label='Реальні дані', color='black', linewidth=2)
plt.plot(test.index, arima_forecast, label='ARIMA', linestyle='--')
plt.plot(test.index, poly_forecast, label='PolyReg', linestyle='-.')
if lstm_forecast is not None:
    plt.plot(test.index, lstm_forecast, label='LSTM', color='red')
plt.title('Порівняння прогнозів')
plt.legend()
plt.show()