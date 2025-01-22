import torch
from torch.utils.data import Dataset

class LSTMDataset(Dataset):
    def __init__(self, dataframe, sequence_length=10):
        """
        LSTMDataset 생성자.
        :param dataframe: 입력 데이터프레임 (Open, High, Low, Close, Volume, RSI, EMA 포함)
        :param sequence_length: LSTM 모델에 사용할 시계열 길이
        """
        self.data = dataframe
        self.sequence_length = sequence_length

        # 정규화
        self.scaled_data = self.data.copy()
        self.scaled_data[['Open', 'High', 'Low', 'Close', 'RSI', 'EMAF']] = \
            self.scaled_data[['Open', 'High', 'Low', 'Close', 'RSI', 'EMAF']].apply(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
    
    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        """
        특정 인덱스에 해당하는 데이터 반환.
        """
        # 시계열 데이터 슬라이스
        features = self.scaled_data[['Open', 'High', 'Low', 'Close', 'RSI', 'EMAF']].iloc[
                   idx:idx + self.sequence_length].values
        target = self.data['Target'].iloc[idx + self.sequence_length]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

