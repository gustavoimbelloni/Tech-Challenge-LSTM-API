import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df: pd.DataFrame, target_column: str = 'Close', sequence_length: int = 60):
    """
    Pré-processa os dados para o modelo LSTM.

    Args:
        df (pd.DataFrame): DataFrame com os dados históricos.
        target_column (str): Coluna alvo para previsão (padrão: 'Close').
        sequence_length (int): Comprimento da sequência de entrada para o LSTM.

    Returns:
        tuple: X_train, y_train, X_test, y_test, scaler
    """
    data = df.filter([target_column])
    dataset = data.values

    # Escalonar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Dividir os dados em treino e teste (80% treino, 20% teste)
    training_data_len = int(np.ceil(len(scaled_data) * .8))

    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []

    for i in range(sequence_length, len(train_data)):
        x_train.append(train_data[i-sequence_length:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len - sequence_length:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(sequence_length, len(test_data)):
        x_test.append(test_data[i-sequence_length:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, scaler

if __name__ == '__main__':
    # Exemplo de uso
    # Criar um DataFrame de exemplo
    data = {
        'Close': np.random.rand(100) * 100
    }
    df_example = pd.DataFrame(data)

    x_train, y_train, x_test, y_test, scaler = preprocess_data(df_example)
    print(f"X_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

