import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score

# 입력 데이터에 대한 JSON 형태로 출력을 반환

def predict(model, pred_loader, device):
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
    #print("실시간 예측 결과:", ["매수", "매도", "대기"][predicted.item()])
    #print("실제 결과 : ", new_data['Target'])