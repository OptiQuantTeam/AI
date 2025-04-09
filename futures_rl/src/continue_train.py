import torch
from pathlib import Path
from train import train
from ppo import PPO

def load_checkpoint(checkpoint_path):
    if torch.cuda.is_available():
        return torch.load(checkpoint_path, map_location=torch.device('cuda'))
    else:
        return torch.load(checkpoint_path, map_location=torch.device('cpu'))

def continue_training(env, model_path, num_episodes=100, checkpoint_term=100):
    try:
        # 모델 로드
        checkpoint = torch.load(model_path)
        model_name = checkpoint['model_name']
        state_dim = checkpoint['state_dim']
        action_dim = checkpoint['action_dim']
        
        # 옵티마이저에서 학습률 가져오기
        optimizer_state = checkpoint['optimizer_state_dict']
        lr_actor = optimizer_state['param_groups'][0]['lr']  # actor의 학습률
        lr_critic = optimizer_state['param_groups'][3]['lr']  # critic의 학습률
        
        gamma = checkpoint['gamma']
        epsilon = checkpoint['epsilon']
        epochs = checkpoint['epochs']
        
        # PPO 에이전트 초기화
        ppo_agent = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            model_name=model_name,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            epsilon=epsilon,
            epochs=epochs
        )
        
        # 모델 상태 로드
        ppo_agent.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        ppo_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 학습 재개
        train(env, ppo_agent, num_episodes=num_episodes, checkpoint_term=checkpoint_term)
    except Exception as e:
        print(f"\n에러 발생: {str(e)}")
        raise e

if __name__ == "__main__":
    # 최신 모델 찾기
    checkpoint_dir = Path('futures_rl/models')
    model_files = list(checkpoint_dir.glob('final_model*.pth'))
    
    if not model_files:
        print("최종 모델 파일을 찾을 수 없습니다.")
        exit(1)
    
    # 가장 최근 모델 선택
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"발견된 최신 모델: {latest_model}")
    
    # 사용자 확인
    user_input = input("이 모델로 추가 학습을 진행하시겠습니까? (y/n): ")
    
    if user_input.lower() == 'y':
        try:
            episodes = int(input("추가로 학습할 에피소드 수를 입력하세요 (기본값: 1000): "))
        except ValueError:
            episodes = 1000
            print(f"기본값 {episodes}로 설정됩니다.")
        
        # 추가 학습 시작
        continue_training(latest_model, episodes)
    else:
        print("학습을 취소합니다.") 