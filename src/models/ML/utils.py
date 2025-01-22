import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score
import pandas as pd
# 모델 불러오기(존재하면 추가 학습, 없다면 새로 학습)
# 데이터 불러오기
# 학습하기
# 모델 저장하기

def train(model, train_loader, val_loader, epochs, learning_rate, device):
    """
    LSTM 모델 학습 함수
    :param model: LSTM 모델
    :param train_loader: 학습 데이터 로더
    :param val_loader: 검증 데이터 로더
    :param epochs: 학습 에폭 수
    :param learning_rate: 학습률
    :param device: 학습에 사용할 디바이스 (CPU/GPU)
    """
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 모델 학습
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_targets = []
        train_predictions = []

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            # 순전파
            outputs = model(features)
            loss = criterion(outputs, targets)
            train_loss += loss.item()

            # 역전파 및 가중치 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 예측 저장
            _, predicted = torch.max(outputs, 1)
            train_targets.extend(targets.cpu().numpy())
            train_predictions.extend(predicted.cpu().numpy())
        
        # 학습 정확도
        train_accuracy = accuracy_score(train_targets, train_predictions)

        # 검증
        model.eval()
        val_loss = 0
        val_targets = []
        val_predictions = []
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_targets.extend(targets.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())
        
        val_accuracy = accuracy_score(val_targets, val_predictions)

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.4f}")

    print("Training Complete") 


def test(model, pred_loader, device):
    """
    실시간 예측 함수
    :param model: 학습된 LSTM 모델
    :param input_data: 새로운 입력 데이터 (시퀀스 형태)
    :param device: 사용할 디바이스 (CPU/GPU)
    :return: 매수(0), 매도(1), 대기(2) 중 하나
    
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
        print(input_tensor.shape)
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()
    """
    model.eval()

    pred_targets = []
    pred_predictions = []
    with torch.no_grad():
        for features, targets in pred_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)

            _, predicted = torch.max(outputs, 1)
            pred_predictions.extend(predicted.cpu().numpy())
            pred_targets.extend(targets.cpu().numpy())
    pred_accuracy = accuracy_score(pred_targets, pred_predictions)
    print(f"Predict Acc: {pred_accuracy:.4f}")
    

# 입력 데이터에 대한 JSON 형태로 출력을 반환

def predict(model, input_data, device):
    """
    실시간 예측 함수
    :param model: 학습된 LSTM 모델
    :param input_data: 새로운 입력 데이터 (시퀀스 형태)
    :param device: 사용할 디바이스 (CPU/GPU)
    :return: 매수(0), 매도(1), 대기(2) 중 하나
    
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
        print(input_tensor.shape)
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()
    """
    model.eval()
 
    with torch.no_grad():
        input_tensor = torch.tensor(input_data.to_numpy(), dtype=torch.float32).unsqueeze(0).to(device)
        outputs = model(input_tensor)

        _, predicted = torch.max(outputs, 1)
    
    
    print("실시간 예측 결과:", ["매수", "매도", "대기"][predicted.item()])
    #print("실제 결과 : ", new_data['Target'])
    return predicted.item()