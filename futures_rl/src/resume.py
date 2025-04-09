import torch
import datetime
from pathlib import Path
from train import train
from ppo import PPO
import json

def load_checkpoint(path):
    if torch.cuda.is_available():
        return torch.load(path, map_location=torch.device('cuda'))
    else:
        return torch.load(path, map_location=torch.device('cpu'))

def resume(env, checkpoint_path):
    try:
        # 체크포인트 로드
        print(f"체크포인트를 로드합니다: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
        
        # 이전 학습 정보 확인
        previous_episodes = checkpoint.get('current_episode', 0)
        previous_returns = checkpoint.get('all_episode_returns', [])
        
        print(f"이전 학습 정보:")
        print(f"- 완료된 에피소드: {previous_episodes}")
        print(f"- 최고 스탭: {max(previous_returns) if previous_returns else 'N/A'}")
        print(f"- 마지막 스탭: {previous_returns[-1] if previous_returns else 'N/A'}")
        
        # 추가 학습 에피소드 수 입력
        try:
            additional_episodes = int(input("추가로 학습할 에피소드 수를 입력하세요 (기본값: 100): ") or "100")
            checkpoint_term = int(input("체크포인트 저장 간격을 입력하세요 (기본값: 100): ") or "100")
        except ValueError:
            print("올바른 숫자를 입력하지 않아 기본값을 사용합니다.")
            additional_episodes = 100
            checkpoint_term = 100
        
        # 이전 학습 상태 확인
        model_name = checkpoint.get('model_name', 'ppo')
        state_dim = checkpoint.get('state_dim', env.observation_space.shape[0])
        action_dim = checkpoint.get('action_dim', env.action_space.n)  # 이산적인 액션 공간 사용
        gamma = checkpoint.get('gamma', 0.99)
        epsilon = checkpoint.get('epsilon', 0.2)
        epochs = checkpoint.get('epochs', 10)
        
        # 옵티마이저에서 학습률 가져오기
        optimizer_state = checkpoint['optimizer_state_dict']
        lr_actor = optimizer_state['param_groups'][0]['lr']  # actor의 학습률
        lr_critic = optimizer_state['param_groups'][3]['lr']  # critic의 학습률
        
        print(f"이전 학습률 설정:")
        print(f"- Actor 학습률: {lr_actor}")
        print(f"- Critic 학습률: {lr_critic}")
        
        # 학습 파라미터 조정 여부 확인
        adjust = input("학습률을 조정하시겠습니까? (y/n): ").lower() == 'y'
        if adjust:
            try:
                new_lr_actor = float(input("새로운 Actor 학습률 (현재: 3e-4): ") or "3e-4")
                new_lr_critic = float(input("새로운 Critic 학습률 (현재: 1e-3): ") or "1e-3")
                
                # 옵티마이저 학습률 조정
                for param_group in optimizer_state['param_groups']:
                    if 'critic' in str(param_group['params']):
                        param_group['lr'] = new_lr_critic
                    else:
                        param_group['lr'] = new_lr_actor
                
                print(f"학습률이 조정되었습니다: Actor={new_lr_actor}, Critic={new_lr_critic}")
            except ValueError:
                print("올바른 숫자를 입력하지 않아 기본 학습률을 유지합니다.")
        
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
        
        # 모델 가중치 로드
        ppo_agent.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        ppo_agent.optimizer.load_state_dict(optimizer_state)
        
        # 이전 학습 히스토리 로드
        previous_rewards = checkpoint.get('rewards_history', [])
        previous_returns = checkpoint.get('returns_history', [])
        previous_episode_returns = checkpoint.get('all_episode_returns', [])
        
        # 추가 학습 시작
        print(f"\n추가 학습을 시작합니다 (목표: {additional_episodes} 에피소드)")
        new_rewards, new_returns, new_episode_returns = train(
            env, 
            ppo_agent, 
            num_episodes=additional_episodes,
            checkpoint_term=checkpoint_term
        )
        
        # 학습 히스토리 통합
        combined_rewards = previous_rewards + new_rewards
        combined_returns = previous_returns + new_returns
        combined_episode_returns = previous_episode_returns + new_episode_returns
        
        # 결과 시각화
        #plot_results(combined_rewards, combined_returns, combined_episode_returns)
        
        # 성능 비교 출력
        print("\n학습 결과 비교:")
        print(f"이전 최고 수익률: {max(previous_episode_returns):.2f}%")
        print(f"새로운 최고 수익률: {max(new_episode_returns):.2f}%")
        print(f"전체 최고 수익률: {max(combined_episode_returns):.2f}%")
        
        # 메타데이터 저장
        metadata = {
            'original_model': str(checkpoint_path),
            'additional_episodes': additional_episodes,
            'total_episodes': len(combined_episode_returns),
            'previous_best_return': max(previous_episode_returns) if previous_episode_returns else float('-inf'),
            'new_best_return': max(new_episode_returns) if new_episode_returns else float('-inf'),
            'overall_best_return': max(combined_episode_returns),
            'training_improvement': max(new_episode_returns) - max(previous_episode_returns) if previous_episode_returns and new_episode_returns else 0
        }
        
        metadata_path = f'futures_rl/models/{model_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
    except Exception as e:
        print(f"\n에러 발생: {str(e)}")
        raise e

if __name__ == "__main__":

    # 가장 최근의 체크포인트 찾기
    checkpoint_dir = Path('futures_rl/checkpoints')
    checkpoints = list(checkpoint_dir.glob('*.pth'))
    
    if not checkpoints:
        print("사용 가능한 체크포인트를 찾을 수 없습니다.")
        exit(1)
    
    # 가장 최근 체크포인트 선택
    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
    print(f"가장 최근 체크포인트: {latest_checkpoint}")
    
    # 사용자 확인
    user_input = input("이 체크포인트로부터 학습을 재개하시겠습니까? (y/n): ")
    
    if user_input.lower() == 'y':
        # 학습 재개
        resume(latest_checkpoint, None)
    else:
        print("학습을 재개하지 않습니다.") 