import torch
import json
import datetime
from pathlib import Path
from environment import FuturesEnv
from ppo import PPO
from train import train_ppo, plot_results

def load_checkpoint(checkpoint_path):
    """체크포인트 파일 로드"""
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

def resume_training(checkpoint_path, num_episodes=2000):
    try:
        checkpoint_dir = Path('futures_rl/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 체크포인트 로드
        print(f"체크포인트를 로드합니다: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
        
        # 이전 학습 상태 확인
        current_episode = checkpoint.get('current_episode', 0)
        total_episodes = checkpoint.get('total_episodes', num_episodes)
        remaining_episodes = total_episodes - current_episode
        
        print(f"이전 학습 진행 상황: {current_episode}/{total_episodes} 에피소드")
        print(f"남은 에피소드 수: {remaining_episodes}")
        
        # 환경 생성
        env = FuturesEnv(path='/workspace/data/preprocess/BTCUSDT/BTCUSDT-1h.csv')
        
        # PPO 에이전트 재생성
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        ppo_agent = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            epsilon=0.2,
            epochs=10
        )
        
        # 모델 가중치 로드
        ppo_agent.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        ppo_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 이전 학습 히스토리 로드
        previous_rewards = checkpoint.get('rewards_history', [])
        previous_returns = checkpoint.get('returns_history', [])
        previous_episode_returns = checkpoint.get('all_episode_returns', [])
        
        print(f"이전 학습 에피소드 수: {len(previous_episode_returns)}")
        print(f"이전 최고 수익률: {max(previous_episode_returns) if previous_episode_returns else 'N/A'}")
        
        # 사용자에게 추가 에피소드 수 확인
        user_input = input(f"기존 목표({total_episodes})까지 계속 진행하시겠습니까? (y/n): ")
        if user_input.lower() != 'y':
            try:
                new_total = int(input("새로운 총 에피소드 수를 입력하세요: "))
                if new_total > current_episode:
                    remaining_episodes = new_total - current_episode
                    total_episodes = new_total
                else:
                    print("현재 에피소드보다 큰 수를 입력해야 합니다.")
                    return
            except ValueError:
                print("올바른 숫자를 입력하세요.")
                return
        
        # 학습 재개
        start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        print(f"\n{start_time}에 학습을 재개합니다. (에피소드 {current_episode + 1}부터 시작)")
        
        # 새로운 학습 실행
        new_rewards, new_returns, new_episode_returns = train_ppo(
            env, 
            ppo_agent, 
            num_episodes=remaining_episodes
        )
        
        # 이전 히스토리와 새로운 히스토리 합치기
        combined_rewards = previous_rewards + new_rewards
        combined_returns = previous_returns + new_returns
        combined_episode_returns = previous_episode_returns + new_episode_returns
        
        # 결과 시각화
        #plot_results(combined_rewards, combined_returns, combined_episode_returns)
        
        # 새로운 체크포인트 저장
        new_checkpoint_path = checkpoint_dir / f'resumed_checkpoint_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
        torch.save({
            'actor_critic_state_dict': ppo_agent.actor_critic.state_dict(),
            'optimizer_state_dict': ppo_agent.optimizer.state_dict(),
            'rewards_history': combined_rewards,
            'returns_history': combined_returns,
            'all_episode_returns': combined_episode_returns,
            'current_episode': current_episode + len(new_rewards),
            'total_episodes': total_episodes,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        }, new_checkpoint_path)
        
        # 메타데이터 업데이트
        metadata = {
            'original_checkpoint': str(checkpoint_path),
            'resume_start_time': start_time,
            'end_time': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'total_episodes': len(combined_episode_returns),
            'previous_episodes': len(previous_episode_returns),
            'new_episodes': len(new_episode_returns),
            'best_return': max(combined_episode_returns),
            'final_return': combined_episode_returns[-1]
        }
        
        with open(checkpoint_dir / f'resume_metadata_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"\n학습이 완료되었습니다. 새로운 체크포인트 저장됨: {new_checkpoint_path}")
        
    except Exception as e:
        print(f"\n에러 발생: {str(e)}")
        raise e
    
    finally:
        # 현재 상태 긴급 저장
        try:
            emergency_path = checkpoint_dir / f'emergency_resume_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
            torch.save({
                'actor_critic_state_dict': ppo_agent.actor_critic.state_dict(),
                'optimizer_state_dict': ppo_agent.optimizer.state_dict(),
                'rewards_history': combined_rewards if 'combined_rewards' in locals() else [],
                'returns_history': combined_returns if 'combined_returns' in locals() else [],
                'all_episode_returns': combined_episode_returns if 'combined_episode_returns' in locals() else [],
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            }, emergency_path)
            print(f"\n긴급 체크포인트가 저장되었습니다: {emergency_path}")
            
        except Exception as save_error:
            print(f"\n긴급 저장 중 에러 발생: {str(save_error)}")

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
        # 추가 에피소드 수 입력
        try:
            num_episodes = int(input("추가로 학습할 에피소드 수를 입력하세요 (기본값: 2000): "))
        except ValueError:
            num_episodes = 2000
            print(f"기본값 {num_episodes}로 설정됩니다.")
        
        # 학습 재개
        resume_training(latest_checkpoint, num_episodes)
    else:
        print("학습을 재개하지 않습니다.") 