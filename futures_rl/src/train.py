import env
from ppo import PPO
import datetime
import json
from graph import plot_cumulative_result

def train(env, ppo_agent, **kwargs):
    episode_rewards = []
    all_episode_returns = []
    episode_start_positions = []
    episode_results = [0]

    checkpoint_term = kwargs['checkpoint_term'] if 'checkpoint_term' in kwargs else 100
    num_episodes = kwargs['num_episodes'] if 'num_episodes' in kwargs else 100
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    try:
        episode = 0
        while episode < num_episodes:
            print(f'\n에피소드 {episode + 1} 시작')
            state = env.reset()
            episode_reward = 0
            
            episode_start_positions.append({
                'step': env.current_step,
                'timestamp': str(env.data.index[env.current_step]),
                'price': env.data.iloc[env.current_step]['Close']
            })
            
            update_count = 0
            step_count = 0

            while True:
                env.num += 1
                step_count += 1
                action, value, log_prob = ppo_agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                ppo_agent.store_transition((state, action, reward, next_state, log_prob, value, done))
                
                episode_reward += reward
                state = next_state
                
                if len(ppo_agent.memory) >= 256:
                    update_count += 1
                    ppo_agent.update()
                
                if done:
                    if info['clear']:
                        result = 1
                        print(f"에피소드 {episode + 1}: 성공 (목표 달성)")
                    elif info['liquidated']:
                        result = 0
                        print(f"에피소드 {episode + 1}: 실패 (자본 소진)")
                    else:
                        result = 0
                        print(f"에피소드 {episode + 1}: 실패")
                    episode_results.append(result)
                    break
            
            episode_rewards.append(episode_reward)
            final_return = env.returns_history[-1] if env.returns_history else 0
            all_episode_returns.append(final_return)
            
            if (episode + 1) % checkpoint_term == 0:
                checkpoint_data = {
                    'model_name': ppo_agent.model_name,
                    'state_dim': ppo_agent.state_dim,
                    'action_dim': ppo_agent.action_dim,
                    'checkpoint_term': checkpoint_term,
                    'gamma': ppo_agent.gamma,
                    'epsilon': ppo_agent.epsilon,
                    'epochs': ppo_agent.epochs,
                    'actor_critic_state_dict': ppo_agent.actor_critic.state_dict(),
                    'optimizer_state_dict': ppo_agent.optimizer.state_dict(),
                    'rewards_history': episode_rewards,
                    'returns_history': env.returns_history,
                    'all_episode_returns': all_episode_returns,
                    'episode_start_positions': episode_start_positions,
                    'current_episode': episode + 1
                }
                ppo_agent.checkpoint(checkpoint_data, f'futures_rl/checkpoints/{ppo_agent.model_name}_{datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")}_ep_{episode + 1}.pth')
                print(f"\n체크포인트 저장됨: {episode + 1}")
                
            episode += 1

    except KeyboardInterrupt:
        print("\n학습이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n에러 발생: {str(e)}")
        raise e
    finally:
        try:
            # 체크포인트 데이터 준비
            checkpoint_data = {
                'model_name': ppo_agent.model_name,
                'state_dim': ppo_agent.state_dim,
                'action_dim': ppo_agent.action_dim,
                'checkpoint_term': checkpoint_term,
                'gamma': ppo_agent.gamma,
                'epsilon': ppo_agent.epsilon,
                'epochs': ppo_agent.epochs,
                'actor_critic_state_dict': ppo_agent.actor_critic.state_dict(),
                'optimizer_state_dict': ppo_agent.optimizer.state_dict(),
                'rewards_history': episode_rewards if 'episode_rewards' in locals() else [],
                'returns_history': env.returns_history if hasattr(env, 'returns_history') else [],
                'all_episode_returns': all_episode_returns if 'all_episode_returns' in locals() else [],
                'episode_start_positions': episode_start_positions if 'episode_start_positions' in locals() else [],
                'current_episode': episode + 1 if 'episode' in locals() else 0,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            }
            
            # 메타데이터 저장
            metadata = {
                'start_time': start_time,
                'end_time': datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                'total_episodes': len(all_episode_returns) if 'all_episode_returns' in locals() else 0,
                'best_return': max(all_episode_returns) if 'all_episode_returns' in locals() and all_episode_returns else float('-inf'),
                'final_return': all_episode_returns[-1] if 'all_episode_returns' in locals() and all_episode_returns else float('-inf')
            }

            if 'episode_start_positions' in locals() and episode_start_positions:
                start_steps = [pos['step'] for pos in episode_start_positions]
                metadata['start_positions_stats'] = {
                    'min_step': min(start_steps),
                    'max_step': max(start_steps),
                    'mean_step': sum(start_steps) / len(start_steps) if start_steps else 0
                }

            time = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")
            ppo_agent.save_model(checkpoint_data, f'futures_rl/models/{ppo_agent.model_name}_{time}.pth')
            with open(f'futures_rl/json/{ppo_agent.model_name}_metadata_{time}.json', 'w') as f:
                json.dump(metadata, f, indent=4)
            plot_cumulative_result(episode_results, f'futures_rl/results/{ppo_agent.model_name}_result_{time}.png')

            print(f"\n체크포인트가 저장되었습니다: {time}")
            
        except Exception as save_error:
            print(f"\n체크포인트 저장 중 에러 발생: {str(save_error)}")
    
    return episode_rewards, env.returns_history, all_episode_returns



if __name__ == "__main__":
    # 환경 생성
    env = env.FuturesEnv2(path='/workspace/data/preprocess/BTCUSDT/BTCUSDT-1h.csv')
    
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
    rewards_history, returns_history, all_episode_returns = train(env, ppo_agent, num_episodes=2000)