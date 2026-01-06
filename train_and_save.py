import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
import joblib
import os

# 1. Coleta de Dados
symbol = 'AAPL'
start_date = '2018-01-01'
end_date = '2024-07-20'
sequence_length = 60

# Criar um DataFrame dummy para simular os dados de fechamento
dates = pd.date_range(start=start_date, end=end_date)
close_prices = np.random.rand(len(dates)) * 100 + 50 # Preços entre 50 e 150
df = pd.DataFrame({"Close": close_prices}, index=dates)
df.index.name = "Date"

# 2. Pré-processamento dos Dados
data = df.filter(['Close'])
dataset = data.values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

training_data_len = int(np.ceil(len(scaled_data) * .8))

train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(sequence_length, len(train_data)):
    x_train.append(train_data[i-sequence_length:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Dados de teste (apenas para simular o processo, não serão usados para treinamento aqui)
test_data = scaled_data[training_data_len - sequence_length:, :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i-sequence_length:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 3. Construção e Treinamento do Modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo (usando menos épocas para agilizar a simulação)
epochs = 5 # Reduzido para simulação
batch_size = 32

print("Iniciando treinamento do modelo...")
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
print("Treinamento concluído.")

# 4. Salvamento e Exportação do Modelo e Scaler
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, 'lstm_model.h5')
model.save(model_path)
print(f"Modelo salvo em: {model_path}")

scaler_path = os.path.join(model_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler salvo em: {scaler_path}")

# Opcional: Avaliação básica para verificar se o modelo funciona
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE (simulado): {rmse}")

