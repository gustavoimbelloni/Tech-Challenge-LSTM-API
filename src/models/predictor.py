import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

class StockPredictor:
    def __init__(self, model_path: str, scaler: MinMaxScaler, sequence_length: int = 60):
        self.model = load_model(model_path)
        self.scaler = scaler
        self.sequence_length = sequence_length

    def predict(self, historical_data: pd.DataFrame) -> float:
        """
        Faz a previsão do próximo preço de fechamento com base nos dados históricos fornecidos.

        Args:
            historical_data (pd.DataFrame): DataFrame com os dados históricos recentes (coluna 'Close').

        Returns:
            float: O preço de fechamento previsto.
        """
        # Certifica-se de que os dados de entrada têm o comprimento correto
        if len(historical_data) < self.sequence_length:
            raise ValueError(f"Historical data must have at least {self.sequence_length} entries.")

        # Pega os últimos 'sequence_length' valores da coluna 'Close'
        last_sequence = historical_data["Close"].values[-self.sequence_length:].reshape(-1, 1)

        # Escala os dados
        scaled_last_sequence = self.scaler.transform(last_sequence)

        # Reshape para o formato esperado pelo LSTM (1, sequence_length, 1)
        x_input = scaled_last_sequence.reshape(1, self.sequence_length, 1)

        # Faz a previsão
        predicted_scaled_price = self.model.predict(x_input)

        # Inverte a escala para obter o preço real
        predicted_price = self.scaler.inverse_transform(predicted_scaled_price)

        return predicted_price[0][0]

if __name__ == '__main__':
    # Exemplo de uso (requer um modelo salvo e um scaler)
    # Este bloco é apenas para demonstração e não funcionará sem um modelo e scaler pré-treinados.
    print("Para usar o StockPredictor, você precisa de um modelo e um scaler treinados e salvos.")
    print("Por favor, treine o modelo usando o notebook 'treinamento_modelo.ipynb' primeiro.")

    # Exemplo de como seria o uso se tivéssemos um modelo e scaler
    # from tensorflow.keras.models import load_model
    # import joblib # Para salvar/carregar o scaler

    # # Supondo que você salvou seu modelo e scaler
    # model_path = 'lstm_model.h5'
    # scaler_path = 'scaler.pkl'

    # # Criar dados históricos de exemplo
    # sample_data = pd.DataFrame({
    #     'Close': np.random.rand(60) * 100
    # })

    # # Carregar o scaler e o modelo (isso falharia sem os arquivos)
    # try:
    #     loaded_scaler = joblib.load(scaler_path)
    #     predictor = StockPredictor(model_path, loaded_scaler)
    #     predicted_value = predictor.predict(sample_data)
    #     print(f"Preço previsto: {predicted_value}")
    # except FileNotFoundError:
    #     print("Modelo ou scaler não encontrados. Por favor, treine e salve-os primeiro.")

