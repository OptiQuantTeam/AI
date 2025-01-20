import torch
import torch.nn as nn

class BitcoinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        """
        Bitcoin 예측 LSTM 모델
        :param input_size: 입력 특성 수 (Open, High, Low, Close, Volume, RSI, MACD, EMA 등)
        :param hidden_size: LSTM 은닉 상태 크기
        :param num_layers: LSTM 계층 수
        :param num_classes: 출력 클래스 수 (매수, 매도, 대기)
        """
        super(BitcoinLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 초기 은닉 상태와 셀 상태 정의
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))
        
        # 마지막 시퀀스의 출력만 사용
        out = self.fc(out[:, -1, :])
        return out