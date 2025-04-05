import env
import algo
import datetime
import json
from graph import plot_cumulative_result
import numpy as np
from Logger import Logger


def train(env, ppo_agent, num_episodes=1000, model_info=None, logger=None, **kwargs):
    if logger is None:
        logger = Logger(ppo_agent.model_name, f'logs/{ppo_agent.model_name}.log')
    env.logger = logger
    
    logger.render_training_start(time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))

    # 이전 학습 데이터 로드 (있는 경우)
    episode_rewards = model_info.get('rewards_history', [])
    all_episode_returns = model_info.get('all_episode_returns', [])
    episode_start_positions = model_info.get('episode_start_positions', [])
    episode_results = model_info.get('episode_results', [0])
    total_steps = model_info.get('total_steps', 0)
    env.num = total_steps  # 전체 스텝 수 복원
    
    # 체크포인트 주기 설정
    checkpoint_term = model_info.get('checkpoint_term', 100)
    start_episode = model_info.get('current_episode', 0)
    
    # 이전 학습 시간 로드 (없으면 현재 시간 사용)
    start_time = model_info.get('start_time', 
                          (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))
    
    is_normal_exit = False
    episode = start_episode  # try 블록 밖에서 초기화
    
    try:
        for episode in range(start_episode, num_episodes):
            logger.render_episode_start(episode + 1)

            state = env.reset()
            episode_reward = 0
            
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
                
                
                
                ppo_agent.store_transition((state, action, reward, next_state, log_prob, value, done))
                
                episode_reward += reward
                state = next_state
                
                if len(ppo_agent.memory) >= 256:
                    update_count += 1
                    ppo_agent.update(success_rate=sum(episode_results) / len(episode_results))
                
                if done:
                    if info['clear']:
                        result = 1
                    else:
                        result = 0
                    episode_results.append(result)
                    break
            
            logger.render(f"  반복한 step: {env.num}, 에피소드 보상: {episode_reward:.2f}, 업데이트 횟수: {update_count}")
            env.render()
            logger.render_episode_end()
            
            episode_rewards.append(episode_reward)
            final_return = env.returns_history[-1]
            all_episode_returns.append(final_return)
            
            # 주기적으로 체크포인트 저장
            if (episode + 1) % checkpoint_term == 0:
                ppo_agent.checkpoint({
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
                    'episode_results': episode_results,
                    'all_episode_returns': all_episode_returns,
                    'episode_start_positions': episode_start_positions,
                    'current_episode': episode + 1,
                    'total_episodes': num_episodes,
                    'learning_params': {
                        'lr_actor': ppo_agent.optimizer.param_groups[0]['lr'],
                        'lr_critic': ppo_agent.optimizer.param_groups[-1]['lr'],
                        'batch_size': 256,
                        'device': str(ppo_agent.device)
                    },
                    'training_stats': {
                        'total_episodes': len(all_episode_returns),
                        'completed_episodes': sum(episode_results),
                        'win_rate': sum(episode_results) / len(episode_results) if episode_results else 0,
                        'total_steps': env.num,
                        'update_counts': update_count if 'update_count' in locals() else 0
                    },
                    'environment_info': {
                        'data_path': env.path if hasattr(env, 'path') else None,
                        'total_data_length': len(env.data) if hasattr(env, 'data') else 0,
                        'training_period': {
                            'start': str(env.data.index[0]) if hasattr(env, 'data') else None,
                            'end': str(env.data.index[-1]) if hasattr(env, 'data') else None
                        }
                    },
                    'cumulative_stats': {
                        'total_steps': env.num,
                        'total_episodes': len(all_episode_returns),
                        'start_time': start_time,
                        'previous_checkpoints': model_info.get('previous_checkpoints', []),
                        'previous_episodes': model_info.get('start_episode', 0),
                        'current_session_episodes': episode + 1 - model_info.get('start_episode', 0),
                        'training_sessions': model_info.get('training_sessions', 0) + 1
                    }
                }, f'checkpoints/{ppo_agent.model_name}_{(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%Y%m%d_%H:%M:%S")}_ep_{episode + 1}.pth')
                logger.render(f" <체크포인트 저장됨: {episode + 1}>")
        
        is_normal_exit = True

        result = {
            'total_episodes': len(all_episode_returns),
            'completed_episodes': sum(episode_results),
            'win_rate': sum(episode_results) / len(episode_results) if episode_results else 0
        }
        logger.render_training_result(result=result)
    except KeyboardInterrupt:
        logger.error("\n학습이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"\n에러 발생: {str(e)}")
        raise e
    finally:
        try:
            if is_normal_exit:

                logger.render_training_end(time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))
            else:
                logger.render_training_stop(time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))
            
            time = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%Y%m%d_%H:%M:%S")

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
                'rewards_history': episode_rewards,
                'returns_history': env.returns_history,
                'episode_results': episode_results,
                'all_episode_returns': all_episode_returns,
                'episode_start_positions': episode_start_positions,
                'current_episode': episode + 1,
                'total_episodes': num_episodes,
                'learning_params': {
                    'lr_actor': ppo_agent.optimizer.param_groups[0]['lr'],
                    'lr_critic': ppo_agent.optimizer.param_groups[-1]['lr'],
                    'batch_size': 256,
                    'device': str(ppo_agent.device)
                },
                'training_stats': {
                    'total_episodes': len(all_episode_returns),
                    'completed_episodes': sum(episode_results),
                    'win_rate': sum(episode_results) / len(episode_results) if episode_results else 0,
                    'total_steps': env.num,
                    'update_counts': update_count if 'update_count' in locals() else 0
                },
                'environment_info': {
                    'data_path': env.path if hasattr(env, 'path') else None,
                    'total_data_length': len(env.data) if hasattr(env, 'data') else 0,
                    'training_period': {
                        'start': str(env.data.index[0]) if hasattr(env, 'data') else None,
                        'end': str(env.data.index[-1]) if hasattr(env, 'data') else None
                    }
                },
                'cumulative_stats': {
                    'total_steps': env.num,
                    'total_episodes': len(all_episode_returns),
                    'start_time': start_time,
                    'previous_checkpoints': model_info.get('previous_checkpoints', []),
                    'previous_episodes': model_info.get('start_episode', 0),
                    'current_session_episodes': episode + 1 - model_info.get('start_episode', 0),
                    'training_sessions': model_info.get('training_sessions', 0) + 1
                },

                'session_info': {
                    'session_type': 'new',
                    'session_time': time,
                    'log_file': f'logs/{ppo_agent.model_name}.log'
                }
            }
            
            # 메타데이터 저장
            if 'episode_start_positions' in locals() and episode_start_positions:
                start_steps = [pos['step'] for pos in episode_start_positions]
                
                # 수익률 통계 계산
                returns_array = np.array(all_episode_returns) if 'all_episode_returns' in locals() and all_episode_returns else np.array([])
                returns_std = float(np.std(returns_array)) if len(returns_array) > 0 else 0
                
                # 승률 계산
                win_rate = sum(episode_results) / len(episode_results) if episode_results else 0
                
                metadata = {
                    # 기본 정보
                    'model_name': ppo_agent.model_name,
                    'initial_start_time': start_time,  # 최초 학습 시작 시간
                    'training_start_time': start_time,  # 이번 학습 시작 시간
                    'end_time': time,
                    
                    # 학습 파라미터
                    'learning_params': {
                        'state_dim': ppo_agent.state_dim,
                        'action_dim': ppo_agent.action_dim,
                        'lr_actor': ppo_agent.optimizer.param_groups[0]['lr'],
                        'lr_critic': ppo_agent.optimizer.param_groups[-1]['lr'],
                        'gamma': ppo_agent.gamma,
                        'epsilon': ppo_agent.epsilon,
                        'epochs': ppo_agent.epochs,
                        'batch_size': 256,  # 메모리 크기
                        'device': str(ppo_agent.device)
                    },
                    
                    # 학습 결과 통계
                    'training_stats': {
                        'total_episodes': len(all_episode_returns) if 'all_episode_returns' in locals() else 0,
                        'completed_episodes': sum(episode_results) if episode_results else 0,
                        'win_rate': win_rate,
                        'total_steps': env.num if hasattr(env, 'num') else 0,
                        'update_counts': update_count if 'update_count' in locals() else 0
                    },
                    
                    # 수익률 통계
                    'returns_stats': {
                        'best_return': float(max(returns_array)) if len(returns_array) > 0 else float('-inf'),
                        'worst_return': float(min(returns_array)) if len(returns_array) > 0 else float('-inf'),
                        'final_return': float(returns_array[-1]) if len(returns_array) > 0 else float('-inf'),
                        'mean_return': float(np.mean(returns_array)) if len(returns_array) > 0 else 0,
                        'return_std': returns_std,
                        'sharpe_ratio': float(np.mean(returns_array) / returns_std) if returns_std != 0 and len(returns_array) > 0 else 0
                    },
                    
                    # 보상 통계
                    'reward_stats': {
                        'total_reward': sum(episode_rewards) if 'episode_rewards' in locals() else 0,
                        'mean_reward': np.mean(episode_rewards) if 'episode_rewards' in locals() else 0,
                        'max_reward': max(episode_rewards) if 'episode_rewards' in locals() else float('-inf'),
                        'min_reward': min(episode_rewards) if 'episode_rewards' in locals() else float('inf'),
                        'reward_std': float(np.std(episode_rewards)) if 'episode_rewards' in locals() else 0
                    },
                    
                    # 시작 위치 통계
                    'start_positions_stats': {
                        'min_step': min(start_steps),
                        'max_step': max(start_steps),
                        'mean_step': sum(start_steps) / len(start_steps),
                        'total_positions': len(start_steps)
                    },
                    
                    # 환경 정보
                    'environment_info': {
                        'data_path': env.path if hasattr(env, 'path') else None,
                        'total_data_length': len(env.data) if hasattr(env, 'data') else 0,
                        'training_period': {
                            'start': str(env.data.index[0]) if hasattr(env, 'data') else None,
                            'end': str(env.data.index[-1]) if hasattr(env, 'data') else None
                        }
                    },
                    
                    # 누적 학습 통계
                    'cumulative_stats': {
                        'total_training_episodes': len(all_episode_returns),
                        'previous_episodes': start_episode,  # 이전까지 학습한 에피소드
                        'current_session_episodes': episode + 1 - start_episode,  # 이번 세션에서 학습한 에피소드
                        'total_episodes_all_sessions': start_episode + (episode + 1 - start_episode),  # 총 학습한 에피소드
                        'total_steps': env.num,
                        'previous_steps': total_steps,  # 이전까지의 스텝 수
                        'current_session_steps': env.num - total_steps,  # 이번 세션의 스텝 수
                        'total_completed_episodes': sum(episode_results),
                        'overall_win_rate': win_rate,
                        'training_sessions': model_info.get('training_sessions', 0) + 1  # 학습 세션 횟수
                    },
                    
                    # 현재 세션 통계
                    'current_session': {
                        'session_number': model_info.get('training_sessions', 0) + 1,  # 현재 세션 번호
                        'start_episode': start_episode,
                        'current_episode': episode + 1,
                        'episodes_this_session': episode + 1 - start_episode,
                        'steps_this_session': env.num - total_steps,
                        'session_start_time': start_time,
                        'session_end_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S')
                    },
                    
                    # 학습 히스토리 추가
                    'training_history': {
                        'previous_sessions': model_info.get('training_history', []),
                        'current_session': {
                            'session_number': model_info.get('training_sessions', 0) + 1,
                            'start_episode': start_episode,
                            'end_episode': episode + 1,
                            'episodes_trained': episode + 1 - start_episode,
                            'start_time': start_time,
                            'end_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'),
                            'total_steps': env.num - total_steps,
                            'win_rate': win_rate
                        }
                    },

                    'session_info': {
                        'session_type': 'new',
                        'session_time': time,
                        'log_file': f'logs/{ppo_agent.model_name}.log'
                    }
                }

            
            if is_normal_exit:
                ppo_agent.save_model(checkpoint_data, f'models/{ppo_agent.model_name}_{time}.pth')
                with open(f'json/{ppo_agent.model_name}_metadata_{time}.json', 'w') as f:
                    json.dump(metadata, f, indent=4)
                plot_cumulative_result(episode_results, f'results/{ppo_agent.model_name}_result_{time}.png')
                
            else:
                ppo_agent.save_model(checkpoint_data, f'checkpoints/{ppo_agent.model_name}_{time}.pth')
                with open(f'json/{ppo_agent.model_name}_checkpoint_{time}.json', 'w') as f:
                    json.dump(metadata, f, indent=4)

                
            
            logger.render(f" <체크포인트가 저장되었습니다: {time}>")
            
        except Exception as save_error:
            logger.error(f" <<체크포인트 저장 중 에러 발생: {str(save_error)}>>")
    
    
    return episode_rewards, env.returns_history, all_episode_returns, {
        'total_steps': env.num,
        'start_time': start_time,
        'episode_start_positions': episode_start_positions,
        'episode_results': episode_results
    }



if __name__ == "__main__":
    # 새로운 학습 시작
    env = env.FuturesEnv2(path='data/preprocess/BTCUSDT/BTCUSDT-1h.csv')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    ppo_agent = algo.PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        epsilon=0.2,
        epochs=10
    )
    
    # 학습 실행
    results = train(env, ppo_agent, num_episodes=2000)
    
    # 또는 학습 재개
