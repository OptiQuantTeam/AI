import torch.nn as nn
import torch

class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

model = MyNeuralNetwork()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    # 예시: 입력 데이터와 정답 레이블
    inputs = torch.randn(32, 10)  # 배치 크기 32, 입력 크기 10
    labels = torch.randint(0, 5, (32,))  # 정답 레이블

    optimizer.zero_grad()  # 기울기 초기화
    outputs = model(inputs)  # 모델의 예측 값 계산
    loss = loss_function(outputs, labels)
    