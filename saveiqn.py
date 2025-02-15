import torch
'''
# IQN 모델 저장 (신경망 가중치만 저장)
torch.save(iqn_model.state_dict(), "iqn_model.pth")

# IQNAgent 저장 (모델 가중치 포함하여 전체 상태 저장)
agent_state = {
    "model_state": iqn_model.state_dict(),  # IQN 모델 가중치
    "optimizer_state": optimizer.state_dict(),  # 옵티마이저 상태 (Adam 등)
    "epsilon": agent.epsilon,  # 탐색률 (필요하다면)
    "hyperparameters": agent.hyperparameters,  # 하이퍼파라미터 (있다면)
    "replay_buffer": agent.replay_buffer,  # 경험 리플레이 버퍼 (선택적)
}
torch.save(agent_state, "iqn_agent.pth")


# IQN 모델 로드
iqn_model.load_state_dict(torch.load("iqn_model.pth"))
iqn_model.eval()  # 평가 모드 전환 (필요 시)

# IQNAgent 로드
checkpoint = torch.load("iqn_agent.pth")
iqn_model.load_state_dict(checkpoint["model_state"])  # 모델 가중치 로드
optimizer.load_state_dict(checkpoint["optimizer_state"])  # 옵티마이저 상태 로드
agent.epsilon = checkpoint["epsilon"]  # 탐색률 로드
agent.hyperparameters = checkpoint["hyperparameters"]  # 하이퍼파라미터 로드 (있다면)
agent.replay_buffer = checkpoint["replay_buffer"]  # 리플레이 버퍼 로드 (선택적)
'''