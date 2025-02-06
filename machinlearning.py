import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from sklearn.model_selection import train_test_split
#마찬가지로 sklearn 사용을 위해 라이브러리 설치(코드:pip install torch torchmetrics scikit-learn)
# 데이터 로드(preprocessing.py 즉 전처리된 데이터 save 해 둔것을 가져온다)


X = np.load("X.npy")  # 입력 데이터
y = np.load("y.npy")  # 출력 데이터
scaler = np.load("scaler.npy", allow_pickle=True).item()  # 스케일러 로드

# 데이터 확인(머신러닝 진행 전에 우선적으로 preprocessing.py(전처리 돌려서 x,y save 해둬야함))
#print(f"입력 데이터 형태: {X.shape}")
#print(f"출력 데이터 형태: {y.shape}")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        
        # LSTM 레이어
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        
        # Fully Connected 레이어
        self.linear = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# 하이퍼파라미터 설정
input_size = 5  # Open, High, Low, Close, Volume (5개 특성)
hidden_layer_size = 64  # LSTM hidden layer 크기
output_size = 1  # 예측할 값은 'Close' 1개

# 모델 초기화
model = LSTMModel(input_size, hidden_layer_size, output_size)

# 모델 구조 확인
#print(model)

# 데이터셋을 PyTorch Tensor로 변환
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
# 데이터 확인
#print(f"X_tensor shape: {X_tensor.shape}")
#print(f"y_tensor shape: {y_tensor.shape}")

# 훈련 데이터와 검증 데이터 분할 (80% 훈련, 20% 검증)
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, shuffle=False)
print(f"훈련 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")


criterion = nn.MSELoss() # 손실 함수 (Mean Squared Error)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 최적화 기법 (Adam)
# 하이퍼파라미터
num_epochs = 20
batch_size = 64

# 학습 시작
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  # 기울기 초기화
    
    # 배치 처리
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        y_pred = model(X_batch) # 모델 예측
        loss = criterion(y_pred, y_batch) # 손실 계산
        loss.backward() # 역전파
        optimizer.step()  # 가중치 업데이트
    
    # 검증
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val)
        val_loss = criterion(y_val_pred, y_val)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}") # 학습 경과 출력

# 예측
model.eval()
with torch.no_grad():
    predictions = model(X_val)
# 예측값을 실제값과 비교
print(f"예측값: {predictions[:5]}")
print(f"실제값: {y_val[:5]}")

# 예측값 복원 (정규화 해제)
predictions_rescaled = scaler.inverse_transform(predictions.detach().numpy())

# 실제값 복원 (정규화 해제)
y_val_rescaled = scaler.inverse_transform(y_val.detach().numpy().reshape(-1, 1))

print(f"복원된 예측값: {predictions_rescaled[:5]}")
print(f"복원된 실제값: {y_val_rescaled[:5]}")
