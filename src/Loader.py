import torch
from pathlib import Path
import algorithm as algo
import env
from Logger import Logger, LogLevel
import datetime
import numpy as np
import os
import json
from graph import plot_episode_metrics, plot_learning_progress
import glob

class Loader():
    def __init__(self, env_path, env_path_test, further=None, auto=False):
        self.env = env.FuturesEnv3(path=env_path)
        self.test_env = env.FuturesEnv_test(path=env_path_test)

        if auto:
            self.agent, self.model_info, self.learning_info = self._load_model(further, auto)
            self.logger = Logger(self.agent.model_name, f'logs/{self.agent.model_name}.log', console_level=LogLevel.CRITICAL, file_level=LogLevel.CRITICAL)
            self.env.logger = self.logger
            self.test_env.logger = self.logger
        else:
            self.agent, self.model_info, self.learning_info = self._set_model() if further is None else self._load_model(further, auto)
            console_level, file_level = self._set_log_level()
            self.logger = Logger(self.agent.model_name, f'logs/{self.agent.model_name}.log', console_level=console_level, file_level=file_level)
            self.env.logger = self.logger
            self.test_env.logger = self.logger
            if self.agent is None:
                self.logger.error("모델을 로드할 수 없습니다.")
                exit(1)
            # 모델 정보 출력
            self.logger.render_model_info(self.model_info)
            input('학습을 시작합니다 [Enter]')

    def _set_log_level(self):
        # 로그 레벨 매핑 딕셔너리
        LEVEL_MAP = {
            0: LogLevel.DEBUG,
            1: LogLevel.INFO,
            2: LogLevel.WARNING,
            3: LogLevel.ERROR,
            4: LogLevel.CRITICAL
        }
        
        def get_log_level(prompt, default=1):
            try:
                level = int(input(prompt) or str(default))
                return LEVEL_MAP.get(level, LogLevel.ERROR)
            except ValueError:
                return LogLevel.ERROR
        
        console_level = get_log_level('콘솔 로그 레벨을 선택해주세요. [0:DEBUG, 1:INFO, 2:WARNING, 3:ERROR, 4:CRITICAL] [default: 2]: ')
        file_level = get_log_level('파일 로그 레벨을 선택해주세요. [0:DEBUG, 1:INFO, 2:WARNING, 3:ERROR, 4:CRITICAL] [default: 2]: ')
        
        return console_level, file_level

    def __select_model(self, model_path, auto):
        models_dir = Path(model_path) 
        model_files = list(models_dir.glob('*.pth'))
        
        if not model_files:
            self.logger.error("사용 가능한 체크포인트를 찾을 수 없습니다.")
            exit(1)
        
        # 수정 시간 기준으로 정렬된 체크포인트 리스트 생성
        sorted_models = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)
        current_index = 0
        
        if auto:
            current_model = sorted_models[0]
            current_model_learning_info = self.__find_learning_info_file(model_path, current_model)
            return current_model, current_model_learning_info
        else:
            while True:
                current_model = sorted_models[current_index]
                print(f"\n현재 선택된 모델: {current_model}")
                user_input = input("이 모델로 학습을 진행하시겠습니까? (y/n): ")
                
                if user_input.lower() == 'y':
                    current_model_learning_info = self.__find_learning_info_file(model_path, current_model)
                    print(f"학습 정보 파일: {current_model_learning_info}")
                    return current_model, current_model_learning_info

                elif user_input.lower() == 'n':
                    current_index = (current_index + 1) % len(sorted_models)
                    if current_index == 0:
                        print("\n모든 모델을 확인했습니다. 처음부터 다시 시작합니다.")
                else:
                    print("학습을 취소합니다.")
                    exit(1)

    def __find_learning_info_file(self, model_path, current_model):
        """
        learning_info 폴더에서 해당 모델의 학습 정보 파일을 찾습니다.
        """
        # 모델 파일 경로에서 파일 이름 추출
        if isinstance(current_model, Path):
            model_name = current_model.stem  # 확장자 제외한 파일 이름
        else:
            model_name = os.path.splitext(os.path.basename(current_model))[0]
        # learning_info 폴더 경로 구성
        learning_info_dir = f'{model_path}/learning_info'
        if not os.path.exists(learning_info_dir):
            return None
            
        # 동일한 이름의 JSON 파일 찾기
        json_file = os.path.join(learning_info_dir, f"{model_name}.json")
        if os.path.exists(json_file):
            return json_file
            
        return None
    
    def _load_model(self, further, auto):
        model_path, learning_info_path = self.__select_model('models' if further else 'checkpoints', auto)
        if model_path is None:
            return None, None
        
        model_info = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        with open(learning_info_path, 'r') as f:
            learning_info = json.load(f)

        # 모델 기본 정보
        model_name = model_info.get('model_name', 'ppo')
        state_dim = model_info.get('state_dim', self.env.observation_space.shape[0])
        action_dim = model_info.get('action_dim', self.env.action_space.n)
        
        # 학습 파라미터 (새 구조)
        learning_params = learning_info.get('learning_params', {})
        gamma = learning_params.get('gamma', 0.99)
        epsilon = learning_params.get('epsilon', 0.2)
        batch_size = learning_params.get('batch_size', 32)
        alpha = learning_params.get('alpha', 0.5)
        epochs = learning_params.get('epochs', 20)
        
        # 학습 진행 상태 (새 구조)
        training_state = learning_info.get('training_state', {})
        current_episode = training_state.get('current_episode', 0)
        self.checkpoint_term = training_state.get('checkpoint_term', 100)
        self.num_episodes = training_state.get('total_episodes', 2000)
        remaining_episodes = self.num_episodes - current_episode
        
        # 옵티마이저에서 학습률 가져오기
        optimizer_state = model_info['optimizer_state_dict']
        lr_actor = optimizer_state['param_groups'][0]['lr']  # actor의 학습률
        lr_critic = optimizer_state['param_groups'][-1]['lr']  # critic의 학습률
        
        # 학습 결과
        training_results = learning_info.get('training_results', {})
        rewards_history = training_results.get('rewards_history', [])
        episode_results = training_results.get('episode_results', [])
        total_episodes = training_results.get('total_episodes', 0)
        completed_episodes = training_results.get('completed_episodes', 0)
        win_rate = training_results.get('win_rate', 0.0)
        episode_win_rate = training_results.get('episode_win_rate', [])
        profit_rate_history = training_results.get('profit_rate_history', [])
        all_balance_history = training_results.get('all_balance_history', [])
        step_num_history = training_results.get('step_num_history', [])

        if further:
            if auto:
                self.num_episodes = current_episode + 1000
            else:
                self.num_episodes = current_episode + int(input("추가로 학습할 에피소드 수를 입력하세요 (기본값: 1000): ") or "1000")
            
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

        else:
            print(f"이전 학습 진행 상황: {current_episode}/{self.num_episodes} 에피소드")
            print(f"남은 에피소드 수: {remaining_episodes}")
            # 사용자에게 추가 에피소드 수 확인
            user_input = input(f"기존 목표({self.num_episodes})까지 계속 진행하시겠습니까? (y/n): ")
            if user_input.lower() != 'y':
                try:
                    new_total = int(input("새로운 총 에피소드 수를 입력하세요: "))
                    if new_total > current_episode:
                        self.num_episodes = new_total
                    else:
                        print("현재 에피소드보다 큰 수를 입력해야 합니다.")
                        return
                except ValueError:
                    print("올바른 숫자를 입력하세요.")
                    return
                
        # PPO 에이전트 재생성
        ppo_agent = algo.PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            model_name=model_name,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            epsilon=epsilon,
            epochs=epochs,
            batch_size=batch_size,
            alpha=alpha
        )

        # 모델 가중치 로드
        ppo_agent.actor_critic.load_state_dict(model_info['actor_critic_state_dict'])
        
        # 옵티마이저 상태 로드 (새로운 학습률 적용)
        optimizer_state = model_info['optimizer_state_dict']
        optimizer_state['param_groups'][0]['lr'] = lr_actor  # actor 학습률
        optimizer_state['param_groups'][-1]['lr'] = lr_critic  # critic 학습률
        ppo_agent.optimizer.load_state_dict(optimizer_state)


        # 모델 정보 구성
        model_info = {
            'model_name': model_info.get('model_name', 'ppo'),
            'state_dim': state_dim,
            'action_dim': action_dim,
            'learning_params': {
                'gamma': gamma,
                'epsilon': epsilon,
                'epochs': epochs,
                'lr_actor': lr_actor,
                'lr_critic': lr_critic,
                'batch_size': batch_size,
                'alpha': alpha,
                'device': str(ppo_agent.device)
            }
        }

        learning_info = { 
            # 학습 진행 상태
            'training_state': {
                'current_episode': current_episode,
                'total_episodes': self.num_episodes,
                'last_step': 0,
                'checkpoint_term': self.checkpoint_term
            },
            # 학습 결과
            'training_results': {
                'rewards_history': rewards_history,
                'episode_results': episode_results,
                'total_episodes': total_episodes,
                'completed_episodes': completed_episodes,
                'win_rate': win_rate,
                'episode_win_rate': episode_win_rate,
                'profit_rate_history': profit_rate_history,
                'all_balance_history': all_balance_history,
                'step_num_history': step_num_history
            },
            # 환경 정보
            'environment_info': {
                'data_path': self.env.path if hasattr(self.env, 'path') else None,
                'total_data_length': len(self.env.data) if hasattr(self.env, 'data') else 0,
                'training_period': {
                    'start': str(self.env.data.index[0]) if hasattr(self.env, 'data') and hasattr(self.env.data, 'index') else None,
                    'end': str(self.env.data.index[-1]) if hasattr(self.env, 'data') and hasattr(self.env.data, 'index') else None
                }
            },
            # 세션 정보
            'session_info': {
                'session_type': 'new',
                'session_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'),
                'start_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'),
                'log_file': f'logs/{model_name}.log',
                'previous_episodes': 0,
                'current_session_episodes': 0,
                'total_episodes_all_sessions': 0,
            }
        }

        return ppo_agent, model_info, learning_info
        
    def _set_model(self):
        model_name = input('\n모델 저장 이름을 입력해주세요. [default: ppo]: ') or "ppo"
        self.num_episodes = int(input('학습할 에피소드 수를 입력해주세요. [default: 1000]: ') or "1000")
        self.checkpoint_term = int(input('체크포인트 저장 주기를 입력해주세요. [default: 100]: ') or "100")
        lr_actor = float(input('lr_actor [default: 3e-4]: ') or "3e-4")            
        lr_critic = float(input('lr_critic [default: 1e-3]: ') or "1e-3")
        gamma = float(input('gamma [default: 0.99]: ') or "0.99")
        epsilon = float(input('epsilon [default: 0.2]: ') or "0.2")
        batch_size = int(input('batch_size [default: 32]: ') or "32")
        alpha = float(input('alpha [default: 0.5]: ') or "0.5")
        epochs = int(input('epochs [default: 20]: ') or "20")

        ppo_agent = algo.PPO(
                state_dim=self.env.observation_space.shape[0],
                action_dim=self.env.action_space.n,
                model_name=model_name,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                batch_size=batch_size,
                alpha=alpha,
                epsilon=epsilon,
                epochs=epochs)

        # 새 모델의 초기 정보 구성 - 새로운 데이터 구조 적용
        model_info = {
            # 모델 기본 정보
            'model_name': model_name,
            'state_dim': self.env.observation_space.shape[0],
            'action_dim': self.env.action_space.n,
            
            # 모델 가중치는 초기 상태로 유지
            
            # 학습 파라미터
            'learning_params': {
                'gamma': gamma,
                'epsilon': epsilon,
                'epochs': epochs,
                'lr_actor': lr_actor,
                'lr_critic': lr_critic,
                'batch_size': batch_size,
                'alpha': alpha,
                'device': str(ppo_agent.device)
            },
        }

        learning_info = { 
            # 학습 진행 상태
            'training_state': {
                'current_episode': 0,
                'total_episodes': self.num_episodes,
                'last_step': 0,
                'checkpoint_term': self.checkpoint_term
            },
            
            # 학습 결과 - 초기 상태
            'training_results': {
                'rewards_history': [],
                'episode_results': [],
                'total_episodes': 0,
                'completed_episodes': 0,
                'win_rate': 0.0,
                'episode_win_rate': [],
                'profit_rate_history': [],
                'all_balance_history': [],
                'step_num_history': []
            },
            
            # 환경 정보
            'environment_info': {
                'data_path': self.env.path if hasattr(self.env, 'path') else None,
                'total_data_length': len(self.env.data) if hasattr(self.env, 'data') else 0,
                'training_period': {
                    'start': str(self.env.data.index[0]) if hasattr(self.env, 'data') and hasattr(self.env.data, 'index') else None,
                    'end': str(self.env.data.index[-1]) if hasattr(self.env, 'data') and hasattr(self.env.data, 'index') else None
                }
            },
            
            # 세션 정보 - 초기 상태
            'session_info': {
                'session_type': 'new',
                'session_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'),
                'start_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'),
                'log_file': f'logs/{model_name}.log',
                'previous_episodes': 0,
                'current_session_episodes': 0,
                'total_episodes_all_sessions': 0,
                'previous_steps': 0,
                'current_session_steps': 0,
                'training_sessions': 1
            }
        }

        return ppo_agent, model_info, learning_info

    def train(self, **kwargs):
        # Agent 학습 모드 전환
        self.agent.test_mode = False
        self.agent.alpha = 1
        self.logger.setTrainLevel()

        self.logger.render_training_start(time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))
        
        # 학습 진행 상황
        if 'training_state' in self.learning_info:
            training_state = self.learning_info['training_state']
            start_episode = training_state.get('current_episode', 0)
            end_episode = training_state.get('total_episodes', self.num_episodes)
            checkpoint_term = training_state.get('checkpoint_term', 100)
        
        # 성능 지표
        if 'training_results' in self.learning_info:
            training_results = self.learning_info['training_results']
            episode_rewards = training_results.get('rewards_history', [])
            episode_results = training_results.get('episode_results', [])
            win_rate = training_results.get('win_rate', 0.0)
            episode_win_rate = training_results.get('episode_win_rate', [])
            profit_rate_history = training_results.get('profit_rate_history', [])
            all_balance_history = training_results.get('all_balance_history', [])
            step_num_history = training_results.get('step_num_history', [])

        # 이전 학습 시간 로드 (없으면 현재 시간 사용)
        start_time = self.learning_info.get('start_time', (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))
        
        is_normal_exit = False
        episode = start_episode  # try 블록 밖에서 초기화
        
        try:
            for episode in range(start_episode, end_episode):
                self.logger.render_episode_start(episode + 1)
                balance_history = []
                actions = []
                episode_reward = 0                
                update_count = 0
                long_count = 0
                short_count = 0
                neutral_count = 0
                state = self.env.reset()
                
                while True:
                    self.env.num += 1
                    action, value, log_prob = self.agent.select_action(state)
                    
                    if action == 1:
                        long_count += 1
                    elif action == -1:
                        short_count += 1
                    else:
                        neutral_count += 1

                    next_state, reward, done, info = self.env.step(action)
                    
                    actions.append(info['position'])
                    
                    self.agent.store_transition((state, action, reward, next_state, log_prob, value, done))
                    
                    episode_reward += reward
                    state = next_state
                    
                    if len(self.agent.memory) >= self.agent.batch_size*4:
                        update_count += self.agent.update()

                    balance_history.append(info['balance'])

                    if done:
                        profit_rate_history.append(info['profit_rate'])
                        episode_results.append(1 if self.env.balance > self.env.initial_balance * 1.01 else 0)
                        #update_count += 0 if info['liquidated'] else self.agent.update()
                        break

                total_count = long_count + short_count + neutral_count
                self.logger.basic(f"  반복한 step: {self.env.num}, 에피소드 보상: {episode_reward:.2f}, 업데이트 횟수: {update_count}")
                self.logger.basic(f"  롱 포지션: {long_count/total_count*100:.2f}%, 숏 포지션: {short_count/total_count*100:.2f}%, 중립 포지션: {neutral_count/total_count*100:.2f}%")
                self.env.render()
                self.logger.render_episode_end(sum(episode_results) / len(episode_results))
                
                episode_rewards.append(episode_reward)
                all_balance_history.append(self.env.balance)
                step_num_history.append(self.env.num)
                
                # 에피소드 내에서 승률 계산
                profit_count = sum(1 for p in self.env.profit_history if p > 0)
                loss_count = sum(1 for p in self.env.profit_history if p < 0)
                total_trades = profit_count + loss_count
                win_rate = profit_count / total_trades if total_trades > 0 else 0
                episode_win_rate.append(win_rate)
                
                # 학습 진행 상황 평가 및 시각화 (50 에피소드마다)
                if (episode + 1) % 50 == 0:
                    # 학습 진행 상황 평가 및 시각화
                    os.makedirs(f'results/{self.agent.model_name}/learning', exist_ok=True)
                    os.makedirs(f'results/{self.agent.model_name}/performance', exist_ok=True)

                    metrics = plot_episode_metrics(
                        balance_history=self.env.balance_history,
                        profit_history=self.env.profit_history,
                        profit_rate_history=self.env.profit_rate_history,
                        price_history=self.env.price_history,
                        actions=actions,
                        balance_profit_rate_history=self.env.balance_profit_rate_history,
                        path=f'results/{self.agent.model_name}/learning/{self.agent.model_name}_learning_{episode + 1}.png'
                    )

                    self.agent.plot_performance(f'results/{self.agent.model_name}/performance/{self.agent.model_name}_performance_{episode + 1}.png')

                
                if (episode + 1) % 500 == 0:
                    metrics = plot_learning_progress(
                        episode_win_rate=episode_win_rate,
                        profit_rate_history=profit_rate_history,
                        episode_rewards=episode_rewards,
                        episode_results=episode_results,
                        path=f'results/{self.agent.model_name}/{self.agent.model_name}_result_{episode + 1}.png'
                    )

                # 주기적으로 체크포인트 저장
                if (episode + 1) % checkpoint_term == 0:

                    learning_info = {
                        # 학습 진행 상태
                        'training_state': {
                            'current_episode': episode + 1,
                            'total_episodes': self.num_episodes,
                            'last_step': self.env.last_step,
                            'checkpoint_term': checkpoint_term
                        },
                        
                        # 학습 결과
                        'training_results': {
                            'rewards_history': episode_rewards,
                            'episode_results': episode_results,
                            'completed_episodes': sum(episode_results),
                            'win_rate': sum(episode_results) / len(episode_results) * 100 if episode_results else 0,
                            'episode_win_rate': episode_win_rate,
                            'profit_rate_history': profit_rate_history if 'profit_rate_history' in locals() else [],
                            'all_balance_history': all_balance_history if 'all_balance_history' in locals() else [],
                            'step_num_history': step_num_history if 'step_num_history' in locals() else []
                        },
                        
                        # 환경 정보
                        'environment_info': {
                            'data_path': self.env.path if hasattr(self.env, 'path') else None,
                            'total_data_length': len(self.env.data) if hasattr(self.env, 'data') else 0,
                            'training_period': {
                                'start': str(self.env.data.index[0]) if hasattr(self.env, 'data') else None,
                                'end': str(self.env.data.index[-1]) if hasattr(self.env, 'data') else None
                            }
                        },
                        
                        # 세션 정보
                        'session_info': {
                            'session_type': 'checkpoint',
                            'session_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'),
                            'start_time': start_time,
                            'log_file': f'logs/{self.agent.model_name}.log',
                            'previous_checkpoints': self.model_info.get('previous_checkpoints', []),
                            'previous_episodes': self.model_info.get('start_episode', 0),
                            'current_session_episodes': episode + 1 - self.model_info.get('start_episode', 0),
                            'training_sessions': self.model_info.get('training_sessions', 0) + 1
                        }
                    }
                    
                    time = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%Y%m%d_%H-%M-%S")
                    
                    os.makedirs('checkpoints', exist_ok=True)
                    os.makedirs(f'checkpoints/learning_info', exist_ok=True)

                    self.agent.save_model(f'checkpoints/{self.agent.model_name}_{time}_ep_{episode + 1}.pth')
                    self.agent.save_learning_state(learning_info, f'checkpoints/learning_info/{self.agent.model_name}_{time}_ep_{episode + 1}.json')
                    
                    self.logger.render(f" <체크포인트 저장됨: {episode + 1}>")
            
            is_normal_exit = True

            result = {
                'total_episodes': self.num_episodes,
                'completed_episodes': sum(episode_results),
                'win_rate': sum(episode_results) / len(episode_results) * 100 if episode_results else 0
            }

            self.logger.render_training_result(result=result)

        except KeyboardInterrupt:
            self.logger.error("\n학습이 사용자에 의해 중단되었습니다.")
        except Exception as e:
            self.logger.error(f"\n에러 발생: {str(e)}")
            raise e
        finally:
            try:
                if is_normal_exit:
                    self.logger.render_training_end(time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))
                else:
                    self.logger.render_training_stop(time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))
                
                time = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%Y%m%d_%H-%M-%S")
                if is_normal_exit:
                    episode += 1

                # 체크포인트 데이터 준비
                learning_info = {
                    # 학습 진행 상태
                    'training_state': {
                        'current_episode': episode,
                        'total_episodes': self.num_episodes,
                        'last_step': self.env.last_step,
                        'checkpoint_term': checkpoint_term
                    },
                    
                    # 학습 결과
                    'training_results': {
                        'rewards_history': episode_rewards[:episode],
                        'episode_results': episode_results[:episode],
                        'completed_episodes': sum(episode_results[:episode]),
                        'win_rate': sum(episode_results[:episode]) / len(episode_results[:episode]) * 100 if episode_results else 0,
                        'episode_win_rate': episode_win_rate[:episode],
                        'profit_rate_history': profit_rate_history[:episode],
                        'all_balance_history': all_balance_history[:episode],
                        'step_num_history': step_num_history[:episode]
                    },
                    
                    # 환경 정보
                    'environment_info': {
                        'data_path': self.env.path if hasattr(self.env, 'path') else None,
                        'total_data_length': len(self.env.data) if hasattr(self.env, 'data') else 0,
                        'training_period': {
                            'start': str(self.env.data.index[0]) if hasattr(self.env, 'data') else None,
                            'end': str(self.env.data.index[-1]) if hasattr(self.env, 'data') else None
                        }
                    },
                    
                    # 세션 정보
                    'session_info': {
                        'session_type': 'checkpoint',
                        'session_time': (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'),
                        'start_time': start_time,
                        'log_file': f'logs/{self.agent.model_name}.log',
                        'previous_checkpoints': self.model_info.get('previous_checkpoints', []),
                        'previous_episodes': self.model_info.get('start_episode', 0),
                        'current_session_episodes': episode + 1 - self.model_info.get('start_episode', 0),
                        'training_sessions': self.model_info.get('training_sessions', 0) + 1
                    }
                }
                    

                # 승률 계산
                win_rate = sum(episode_results) / len(episode_results) * 100 if episode_results else 0
                

                if is_normal_exit:
                    os.makedirs('models', exist_ok=True)
                    os.makedirs(f'models/learning_info', exist_ok=True)
                    os.makedirs(f'json/{self.agent.model_name}', exist_ok=True)
                    os.makedirs(f'results/{self.agent.model_name}', exist_ok=True)
                    
                    self.agent.save_model(f'models/{self.agent.model_name}_{time}.pth')
                    self.agent.save_learning_state(learning_info, f'models/learning_info/{self.agent.model_name}_{time}.json')

                    
                    # 최종 학습 진행 상황 평가 및 시각화
                    metrics = plot_learning_progress(
                        episode_win_rate=episode_win_rate,
                        profit_rate_history=profit_rate_history,
                        episode_rewards=episode_rewards,
                        episode_results=episode_results,
                        path=f'results/{self.agent.model_name}/{self.agent.model_name}_result_{time}.png'
                    )

                else:
                    os.makedirs('checkpoints', exist_ok=True)
                    os.makedirs(f'checkpoints/learning_info', exist_ok=True)
                    os.makedirs('json', exist_ok=True)
                    
                    # 체크포인트 데이터에 중단 상태 표시
                    learning_info['session_info']['session_type'] = 'interrupted'
                    
                    self.agent.save_model(f'checkpoints/{self.agent.model_name}_{time}.pth')
                    self.agent.save_learning_state(learning_info, f'checkpoints/learning_info/{self.agent.model_name}_{time}.json')


                self.logger.render(f" <체크포인트가 저장되었습니다: {time}>")
                
            except Exception as save_error:
                self.logger.error(f" <<체크포인트 저장 중 에러 발생: {str(save_error)}>>")
        
        return episode_rewards, {
            'total_steps': self.env.num,
            'start_time': start_time,
            'episode_results': episode_results,
            'learning_metrics': metrics if 'metrics' in locals() else None
        }
    
    def test(self):
        # Agent 테스트 모드 전환
        self.agent.test_mode = True
        self.agent.alpha = 0.5  # 1에서 0.5로 변경하여 지표와 액션 확률을 동일한 비중으로 사용
        self.logger.setTestLevel()
        
        try:

            self.logger.render_test_start(time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S'))
            balance_history = []
            actions = []
            episode_reward = 0                
            
            state = self.test_env.reset()

            while True:
                self.test_env.num += 1
                action, value, log_prob = self.agent.select_action(state)
                next_state, reward, done, info = self.test_env.step(action)
                
                actions.append(info['position'])
                
                episode_reward += reward
                state = next_state
                balance_history.append(info['balance'])

                if done:
                    break

            self.test_env.render()
            

            result = {
                'environment_data': {
                    'episode_reward': episode_reward,
                    'final_balance': self.test_env.balance,
                    'initial_balance': self.test_env.initial_balance,
                    'total_steps': self.test_env.num,     
                },
                'performance_metrics': {
                    'total_profit': self.test_env.balance - self.test_env.initial_balance,
                    'profit_rate': (self.test_env.balance - self.test_env.initial_balance) / self.test_env.initial_balance * 100,
                    'profitable_trades': sum(1 for i in range(1, len(balance_history)) if balance_history[i] > balance_history[i-1]),
                },
                'trading_statistics': {
                    'long_positions': sum(1 for action in actions if action == 1),
                    'short_positions': sum(1 for action in actions if action == -1),
                    'neutral_positions': sum(1 for action in actions if action == 0),
                    'consecutive_wins': self._calculate_consecutive_wins(balance_history),
                    'consecutive_losses': self._calculate_consecutive_losses(balance_history)
                }
            }

            self.logger.render_test_result(result)
            time=(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d_%H-%M-%S')
            self.logger.render_test_end(time)

            os.makedirs(f'results/{self.agent.model_name}', exist_ok=True)
            os.makedirs(f'results/{self.agent.model_name}/test', exist_ok=True)

            metrics = plot_episode_metrics(
                    balance_history=self.test_env.balance_history,
                    profit_history=self.test_env.profit_history,
                    profit_rate_history=self.test_env.profit_rate_history,
                    price_history=self.test_env.price_history,
                    actions=actions,
                    balance_profit_rate_history=self.test_env.balance_profit_rate_history,
                    path=f'results/{self.agent.model_name}/test/{self.agent.model_name}_result.png'
                )
            

        except KeyboardInterrupt:
            self.logger.error("\n학습이 사용자에 의해 중단되었습니다.")
        except Exception as e:
            self.logger.error(f"\n에러 발생: {str(e)}")
            raise e
        finally:
            try:
                metadata = {
                    # 모델 기본 정보
                    'model_name': self.agent.model_name,
                    'end_time': time,
                    
                    # 학습 파라미터
                    'learning_params': {
                        'state_dim': self.agent.state_dim,
                        'action_dim': self.agent.action_dim,
                        'gamma': self.agent.gamma,
                        'epsilon': self.agent.epsilon,
                        'epochs': self.agent.epochs,
                        'lr_actor': self.agent.optimizer.param_groups[0]['lr'],
                        'lr_critic': self.agent.optimizer.param_groups[-1]['lr'],
                        'batch_size': self.agent.batch_size,  # 메모리 크기
                        'alpha': self.agent.alpha,
                        'device': str(self.agent.device)
                    },
                    
                    # 성능 지표
                    'performance_metrics': {
                    'total_profit': self.test_env.balance - self.test_env.initial_balance,
                    'profit_rate': (self.test_env.balance - self.test_env.initial_balance) / self.test_env.initial_balance * 100,
                    'profitable_trades': sum(1 for i in range(1, len(balance_history)) if balance_history[i] > balance_history[i-1]),
                    'average_profit_per_trade': (self.test_env.balance - self.test_env.initial_balance) / len(actions) if actions else 0
                    },

                    # 테스트 거래 통계 
                    'trading_statistics': {
                        'long_positions': sum(1 for action in actions if action == 1),
                        'short_positions': sum(1 for action in actions if action == -1),
                        'neutral_positions': sum(1 for action in actions if action == 0),
                        'consecutive_wins': self._calculate_consecutive_wins(balance_history),
                        'consecutive_losses': self._calculate_consecutive_losses(balance_history)
                    },
    
                    # 환경 정보
                    'environment_info': {
                        'data_path': self.test_env.path if hasattr(self.test_env, 'path') else None,
                        'total_data_length': len(self.test_env.data) if hasattr(self.test_env, 'data') else 0,
                        'training_period': {
                            'start': str(self.test_env.data.index[0]) if hasattr(self.test_env, 'data') else None,
                            'end': str(self.test_env.data.index[-1]) if hasattr(self.test_env, 'data') else None
                        }
                    },
                }
                

                os.makedirs('json', exist_ok=True)
                os.makedirs(f'json/{self.agent.model_name}', exist_ok=True)
                # NumPy 타입을 Python 기본 타입으로 변환
                def convert_to_serializable(obj):
                    if isinstance(obj, (np.ndarray, np.number)):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        else:
                            return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_serializable(item) for item in obj]
                    return obj

                metadata = convert_to_serializable(metadata)
                
                metadata_path = f'json/{self.agent.model_name}/{self.agent.model_name}_metadata_{time}.json'
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=4)

                self.logger.render(f" <체크포인트가 저장되었습니다: {time}>")
                
            except Exception as save_error:
                self.logger.error(f" <<체크포인트 저장 중 에러 발생: {str(save_error)}>>")
        
        return True
    
    def _calculate_consecutive_wins(self, balance_history):
        if len(balance_history) < 2:
            return 0
            
        max_consecutive = 0
        current_consecutive = 0
        
        for i in range(1, len(balance_history)):
            if balance_history[i] > balance_history[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
        
    def _calculate_consecutive_losses(self, balance_history):
        if len(balance_history) < 2:
            return 0
            
        max_consecutive = 0
        current_consecutive = 0
        
        for i in range(1, len(balance_history)):
            if balance_history[i] < balance_history[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive