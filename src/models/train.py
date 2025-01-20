import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score

def train(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """
    LSTM 모델 학습 함수
    :param model: LSTM 모델
    :param train_loader: 학습 데이터 로더
    :param val_loader: 검증 데이터 로더
    :param num_epochs: 학습 에폭 수
    :param learning_rate: 학습률
    :param device: 학습에 사용할 디바이스 (CPU/GPU)
    """
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 모델 학습
    for epoch in range(num_epochs):
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

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.4f}")

    print("Training Complete")