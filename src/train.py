import env
import algo
import datetime
import json
from graph import  plot_episode_metrics, plot_learning_progress
import numpy as np
from Logger import Logger
import os


def train(env, ppo_agent, num_episodes=1000, model_info=None, logger=None, **kwargs):

    env.logger = logger if logger is not None else Logger(ppo_agent.model_name, f'logs/{ppo_agent.model_name}.log')
    
    logger.render_training_start(time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))

    # 학습 진행 상황
    if 'training_state' in model_info:
        training_state = model_info['training_state']
        start_episode = training_state.get('current_episode', 0)
        end_episode = training_state.get('total_episodes', num_episodes)
        checkpoint_term = training_state.get('checkpoint_term', 100)
        total_steps = training_state.get('total_steps', 0)
    # 성능 지표
    if 'training_results' in model_info:
        training_results = model_info['training_results']
        episode_rewards = training_results.get('rewards_history', [])
        all_episode_returns = training_results.get('all_episode_returns', [])
        episode_start_positions = training_results.get('episode_start_positions', [])
        episode_results = training_results.get('episode_results', [0])
        profit_rate_history = training_results.get('profit_rate_history', [])

    all_balance_history = []
    all_sharpe_ratios = []

    env.num = total_steps  # 전체 스텝 수 복원

    
    # 이전 학습 시간 로드 (없으면 현재 시간 사용)
    start_time = model_info.get('start_time', (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))
    
    is_normal_exit = False
    episode = start_episode  # try 블록 밖에서 초기화
    
    try:
        for episode in range(start_episode, end_episode):
            logger.render_episode_start(episode + 1)
            balance_history = []
            actions = []
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
                
                actions.append(action[0])
                
                ppo_agent.store_transition((state, action, reward, next_state, log_prob, value, done))
                
                episode_reward += reward
                state = next_state
                
                if len(ppo_agent.memory) >= 256:
                    #update_count += ppo_agent.update(success_rate=sum(episode_results) / len(episode_results))
                    update_count += ppo_agent.update()
                balance_history.append(info['balance'])
                if done:
                    profit_rate_history.append(info['profit_rate'])
                    episode_results.append(1 if env.balance_profit_rate_history[-1] > 0.05 else 0)
                    update_count += ppo_agent.update(success_rate=sum(episode_results) / len(episode_results)) if info['liquidated'] else 0
                    break
            
            logger.render(f"  반복한 step: {env.num}, 에피소드 보상: {episode_reward:.2f}, 업데이트 횟수: {update_count}")
            env.render()
            logger.render_episode_end()
            
            episode_rewards.append(episode_reward)
            final_return = env.last_step
            all_episode_returns.append(final_return)
            all_balance_history.append(env.balance)
            all_sharpe_ratios.append(np.sqrt(7) * np.mean(env.balance_profit_rate_history) / np.std(env.balance_profit_rate_history))

            # 학습 진행 상황 평가 및 시각화 (50 에피소드마다)
            if (episode + 1) % 100 == 0:
                # 학습 진행 상황 평가 및 시각화
                os.makedirs(f'results/{ppo_agent.model_name}/learning', exist_ok=True)
                metrics = plot_episode_metrics(
                    balance_history=env.balance_history,
                    profit_history=env.profit_history,
                    price_history=env.price_history,
                    actions=actions,
                    balance_profit_rate_history=env.balance_profit_rate_history,
                    path=f'results/{ppo_agent.model_name}/learning/{ppo_agent.model_name}_learning_{episode + 1}.png'
                )
                
                # 학습 지표 로깅
                logger.render(f"  학습 지표 - 샤프 비율: {metrics['sharpe_ratio']:.2f}, 승률: {metrics['win_rate']:.2%}")

            # 주기적으로 체크포인트 저장
            if (episode + 1) % checkpoint_term == 0:
                checkpoint_data = {
                    # 모델 기본 정보
                    'model_name': ppo_agent.model_name,
                    'state_dim': ppo_agent.state_dim,
                    'action_dim': ppo_agent.action_dim,
                    
                    # 모델 가중치 및 옵티마이저 상태
                    'actor_critic_state_dict': ppo_agent.actor_critic.state_dict(),
                    'optimizer_state_dict': ppo_agent.optimizer.state_dict(),
                    
                    # 학습 파라미터
                    'learning_params': {
                        'gamma': ppo_agent.gamma,
                        'epsilon': ppo_agent.epsilon,
                        'epochs': ppo_agent.epochs,
                        'lr_actor': ppo_agent.optimizer.param_groups[0]['lr'],
                        'lr_critic': ppo_agent.optimizer.param_groups[-1]['lr'],
                        'batch_size': 256,
                        'device': str(ppo_agent.device)
                    },
                    
                    # 학습 진행 상태
                    'training_state': {
                        'current_episode': episode + 1,
                        'total_episodes': num_episodes,
                        'last_step': env.last_step,
                        'checkpoint_term': checkpoint_term,
                        'total_steps': env.num,
                        'update_counts': update_count if 'update_count' in locals() else 0
                    },
                    
                    # 학습 결과
                    'training_results': {
                        'rewards_history': episode_rewards,
                        'episode_results': episode_results,
                        'all_episode_returns': all_episode_returns,
                        'episode_start_positions': episode_start_positions,
                        'total_episodes': len(all_episode_returns),
                        'completed_episodes': sum(episode_results),
                        'win_rate': sum(episode_results) / len(episode_results) * 100 if episode_results else 0,
                        'profit_rate_history': profit_rate_history if 'profit_rate_history' in locals() else []
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
                    
                    # 세션 정보
                    'session_info': {
                        'session_type': 'checkpoint',
                        'session_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'),
                        'start_time': start_time,
                        'log_file': f'logs/{ppo_agent.model_name}.log',
                        'previous_checkpoints': model_info.get('previous_checkpoints', []),
                        'previous_episodes': model_info.get('start_episode', 0),
                        'current_session_episodes': episode + 1 - model_info.get('start_episode', 0),
                        'training_sessions': model_info.get('training_sessions', 0) + 1
                    }
                }
                
                ppo_agent.checkpoint(checkpoint_data, f'checkpoints/{ppo_agent.model_name}_{(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%Y%m%d_%H:%M:%S")}_ep_{episode + 1}.pth')
                
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
                # 모델 기본 정보
                'model_name': ppo_agent.model_name,
                'state_dim': ppo_agent.state_dim,
                'action_dim': ppo_agent.action_dim,
                
                # 모델 가중치 및 옵티마이저 상태
                'actor_critic_state_dict': ppo_agent.actor_critic.state_dict(),
                'optimizer_state_dict': ppo_agent.optimizer.state_dict(),
                
                # 학습 파라미터
                'learning_params': {
                    'gamma': ppo_agent.gamma,
                    'epsilon': ppo_agent.epsilon,
                    'epochs': ppo_agent.epochs,
                    'lr_actor': ppo_agent.optimizer.param_groups[0]['lr'],
                    'lr_critic': ppo_agent.optimizer.param_groups[-1]['lr'],
                    'batch_size': 256,
                    'device': str(ppo_agent.device)
                },
                
                # 학습 진행 상태
                'training_state': {
                    'current_episode': episode + 1,
                    'total_episodes': num_episodes,
                    'last_step': env.last_step,
                    'checkpoint_term': checkpoint_term,
                    'total_steps': env.num,
                    'update_counts': update_count if 'update_count' in locals() else 0
                },
                
                # 학습 결과
                'training_results': {
                    'rewards_history': episode_rewards,
                    'episode_results': episode_results,
                    'all_episode_returns': all_episode_returns,
                    'episode_start_positions': episode_start_positions,
                    'total_episodes': len(all_episode_returns),
                    'completed_episodes': sum(episode_results),
                    'win_rate': sum(episode_results) / len(episode_results) * 100 if episode_results else 0,
                    'profit_rate_history': profit_rate_history if 'profit_rate_history' in locals() else []
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
                win_rate = sum(episode_results) / len(episode_results) * 100 if episode_results else 0
                
                # 수익률 데이터
                profit_rates = np.array(profit_rate_history) if 'profit_rate_history' in locals() and profit_rate_history else np.array([])
                
                metadata = {
                    # 모델 기본 정보
                    'model_name': ppo_agent.model_name,
                    'initial_start_time': start_time,  # 최초 학습 시작 시간
                    'training_start_time': start_time,  # 이번 학습 시작 시간
                    'end_time': time,
                    
                    # 학습 파라미터
                    'learning_params': {
                        'state_dim': ppo_agent.state_dim,
                        'action_dim': ppo_agent.action_dim,
                        'gamma': ppo_agent.gamma,
                        'epsilon': ppo_agent.epsilon,
                        'epochs': ppo_agent.epochs,
                        'lr_actor': ppo_agent.optimizer.param_groups[0]['lr'],
                        'lr_critic': ppo_agent.optimizer.param_groups[-1]['lr'],
                        'batch_size': 256,  # 메모리 크기
                        'device': str(ppo_agent.device)
                    },
                    
                    # 학습 진행 상태
                    'training_state': {
                        'current_episode': episode + 1,
                        'total_episodes': num_episodes,
                        'last_step': env.last_step,
                        'checkpoint_term': checkpoint_term,
                        'total_steps': env.num if hasattr(env, 'num') else 0,
                        'update_counts': update_count if 'update_count' in locals() else 0
                    },
                    
                    # 학습 결과
                    'training_results': {
                        'total_episodes': len(all_episode_returns) if 'all_episode_returns' in locals() else 0,
                        'completed_episodes': sum(episode_results) if episode_results else 0,
                        'win_rate': win_rate
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
                    
                    # 수익률 데이터 통계
                    'profit_rate_stats': {
                        'best_profit_rate': float(max(profit_rates)) if len(profit_rates) > 0 else float('-inf'),
                        'worst_profit_rate': float(min(profit_rates)) if len(profit_rates) > 0 else float('-inf'),
                        'final_profit_rate': float(profit_rates[-1]) if len(profit_rates) > 0 else float('-inf'),
                        'mean_profit_rate': float(np.mean(profit_rates)) if len(profit_rates) > 0 else 0,
                        'profit_rate_std': float(np.std(profit_rates)) if len(profit_rates) > 0 else 0,
                        'positive_rate': float(np.sum(profit_rates > 0) / len(profit_rates)) if len(profit_rates) > 0 else 0
                    },
                    
                    # 보상 통계
                    'reward_stats': {
                        'total_reward': sum(episode_rewards) if 'episode_rewards' in locals() else 0,
                        'mean_reward': np.mean(episode_rewards) if 'episode_rewards' in locals() else 0,
                        'max_reward': max(episode_rewards) if 'episode_rewards' in locals() else float('-inf'),
                        'min_reward': min(episode_rewards) if 'episode_rewards' in locals() else float('inf'),
                        'reward_std': float(np.std(episode_rewards)) if 'episode_rewards' in locals() else 0
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
                    
                    # 세션 정보
                    'session_info': {
                        'session_type': 'completed' if is_normal_exit else 'interrupted',
                        'session_time': time,
                        'start_time': start_time,
                        'log_file': f'logs/{ppo_agent.model_name}.log',
                        'previous_episodes': start_episode,
                        'current_session_episodes': episode + 1 - start_episode,
                        'total_episodes_all_sessions': start_episode + (episode + 1 - start_episode),
                        'previous_steps': total_steps,
                        'current_session_steps': env.num - total_steps if hasattr(env, 'num') else 0,
                        'training_sessions': model_info.get('training_sessions', 0) + 1
                    },
                    
                    # 학습 히스토리
                    'training_history': {
                        'previous_sessions': model_info.get('training_history', {}).get('previous_sessions', []),
                        'current_session': {
                            'session_number': model_info.get('training_sessions', 0) + 1,
                            'start_episode': start_episode,
                            'end_episode': episode + 1,
                            'episodes_trained': episode + 1 - start_episode,
                            'start_time': start_time,
                            'end_time': time,
                            'total_steps': env.num - total_steps if hasattr(env, 'num') else 0,
                            'win_rate': win_rate,
                            'mean_profit_rate': float(np.mean(profit_rates)) if len(profit_rates) > 0 else 0
                        }
                    }
                }
            
            if is_normal_exit:
                os.makedirs('models', exist_ok=True)
                os.makedirs('json', exist_ok=True)
                os.makedirs(f'results/{ppo_agent.model_name}', exist_ok=True)
                
                # 체크포인트 데이터에 최종 상태 표시
                checkpoint_data['session_info']['session_type'] = 'completed'
                
                ppo_agent.save_model(checkpoint_data, f'models/{ppo_agent.model_name}_{time}.pth')
                with open(f'json/{ppo_agent.model_name}_metadata_{time}.json', 'w') as f:
                    json.dump(metadata, f, indent=4)
                
                # 최종 학습 진행 상황 평가 및 시각화
                metrics = plot_learning_progress(
                    all_balance_history=all_balance_history,
                    profit_rate_history=profit_rate_history,
                    all_sharpe_ratios=all_sharpe_ratios,
                    episode_rewards=episode_rewards,
                    episode_results=episode_results,
                    path=f'results/{ppo_agent.model_name}/{ppo_agent.model_name}_result_{time}.png'
                )

            else:
                os.makedirs('checkpoints', exist_ok=True)
                os.makedirs('json', exist_ok=True)
                
                # 체크포인트 데이터에 중단 상태 표시
                checkpoint_data['session_info']['session_type'] = 'interrupted'
                
                ppo_agent.save_model(checkpoint_data, f'checkpoints/{ppo_agent.model_name}_{time}.pth')
                with open(f'json/{ppo_agent.model_name}_checkpoint_{time}.json', 'w') as f:
                    json.dump(metadata, f, indent=4)

            logger.render(f" <체크포인트가 저장되었습니다: {time}>")
            
        except Exception as save_error:
            logger.error(f" <<체크포인트 저장 중 에러 발생: {str(save_error)}>>")
    
    return episode_rewards, env.last_step, all_episode_returns, {
        'total_steps': env.num,
        'start_time': start_time,
        'episode_start_positions': episode_start_positions,
        'episode_results': episode_results,
        'learning_metrics': metrics if 'metrics' in locals() else None
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
