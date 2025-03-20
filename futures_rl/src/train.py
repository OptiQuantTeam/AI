from environment import FuturesEnv
from ppo import PPO
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from preprocess import preprocess_data
import datetime
import json
from pathlib import Path


def train_ppo(env, ppo_agent, num_episodes=1000):
    episode_rewards = []
    all_episode_returns = []
    episode_start_positions = []
    is_normal_exit = False
    
    checkpoint_dir = Path('futures_rl/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    try:
        for episode in range(num_episodes):
            print(f"\n에피소드 {episode + 1} 시작")
            state = env.reset()
            episode_reward = 0
            was_liquidated = False
            
            episode_start_positions.append({
                'step': env.current_step,
                'timestamp': str(env.data.index[env.current_step]),
                'price': env.data.iloc[env.current_step]['Close']
            })
            
            update_count = 0
            
            while True:
                env.num += 1
                action, value, log_prob = ppo_agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                if info['liquidated']:
                    was_liquidated = True
                
                ppo_agent.store_transition((state, action, reward, next_state, log_prob, value, done))
                
                episode_reward += reward
                state = next_state
                
                if len(ppo_agent.memory) >= 512:
                    update_count += 1
                    ppo_agent.update()
                
                if done:
                    env.render()
                    break
            
            print(f"  현재 step: {env.num}, 에피소드 보상: {episode_reward:.2f}, 업데이트 횟수: {update_count}")
            print(f"  청산 여부: {'청산됨' if was_liquidated else '정상 종료'}")
            
            episode_rewards.append(episode_reward)
            final_return = env.returns_history[-1]
            all_episode_returns.append(final_return)
            
            # 주기적으로 체크포인트 저장
            if (episode + 1) % 100 == 0:
                checkpoint_path = checkpoint_dir / f'checkpoint_ep_{episode + 1}.pth'
                torch.save({
                    'actor_critic_state_dict': ppo_agent.actor_critic.state_dict(),
                    'optimizer_state_dict': ppo_agent.optimizer.state_dict(),
                    'rewards_history': episode_rewards,
                    'returns_history': env.returns_history,
                    'all_episode_returns': all_episode_returns,
                    'episode_start_positions': episode_start_positions,
                    'current_episode': episode + 1,
                    'total_episodes': num_episodes
                }, checkpoint_path)
                print(f"\n체크포인트 저장됨: {checkpoint_path}")
        
        is_normal_exit = True  # 모든 에피소드가 정상적으로 완료됨
        
    except KeyboardInterrupt:
        print("\n학습이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n에러 발생: {str(e)}")
        raise e
    finally:
        try:
            # 저장할 파일 경로 결정
            if is_normal_exit:
                save_path = f'futures_rl/models/final_model_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
                save_prefix = "최종"
            else:
                save_path = checkpoint_dir / f'emergency_checkpoint_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
                save_prefix = "긴급"
            
            # 체크포인트 데이터 준비
            checkpoint_data = {
                'actor_critic_state_dict': ppo_agent.actor_critic.state_dict(),
                'optimizer_state_dict': ppo_agent.optimizer.state_dict(),
                'rewards_history': episode_rewards if 'episode_rewards' in locals() else [],
                'returns_history': env.returns_history if hasattr(env, 'returns_history') else [],
                'all_episode_returns': all_episode_returns if 'all_episode_returns' in locals() else [],
                'episode_start_positions': episode_start_positions if 'episode_start_positions' in locals() else [],
                'current_episode': episode + 1 if 'episode' in locals() else 0,
                'total_episodes': num_episodes,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            }
            
            # 체크포인트 저장
            torch.save(checkpoint_data, save_path)
            
            # 메타데이터 저장
            if 'episode_start_positions' in locals() and episode_start_positions:
                start_steps = [pos['step'] for pos in episode_start_positions]
                metadata = {
                    'start_time': start_time,
                    'end_time': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                    'total_episodes': len(all_episode_returns) if 'all_episode_returns' in locals() else 0,
                    'best_return': max(all_episode_returns) if 'all_episode_returns' in locals() and all_episode_returns else float('-inf'),
                    'final_return': all_episode_returns[-1] if 'all_episode_returns' in locals() and all_episode_returns else float('-inf'),
                    'start_positions_stats': {
                        'min_step': min(start_steps),
                        'max_step': max(start_steps),
                        'mean_step': sum(start_steps) / len(start_steps)
                    }
                }
                
                with open(checkpoint_dir / 'training_metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=4)
            
            print(f"\n{save_prefix} 체크포인트가 저장되었습니다: {save_path}")
            
        except Exception as save_error:
            print(f"\n체크포인트 저장 중 에러 발생: {str(save_error)}")
    
    return episode_rewards, env.returns_history, all_episode_returns

def plot_results(rewards, returns_history, all_episode_returns):
    plt.figure(figsize=(15, 10))
    
    # 보상 그래프
    plt.subplot(2, 1, 1)
    plt.plot(rewards)
    plt.title('reward Graph')
    plt.xlabel('episode')
    plt.ylabel('reward')
    
    # 수익률 그래프
    plt.subplot(2, 1, 2)
    plt.plot(returns_history, label='step per episode')
    plt.title('profit rate Graph')
    plt.xlabel('episode')
    plt.ylabel('profit rate (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

if __name__ == "__main__":
    # 환경 생성
    env = FuturesEnv(path='/workspace/data/preprocess/BTCUSDT/BTCUSDT-1h.csv')
    
    # PPO 에이전트 생성
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
    
    # 학습 실행
    rewards_history, returns_history, all_episode_returns = train_ppo(env, ppo_agent, num_episodes=2000)