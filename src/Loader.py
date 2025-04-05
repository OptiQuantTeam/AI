import torch
from pathlib import Path
import algo
import env
from Logger import Logger, LogLevel

class Loader():
    def __init__(self, further=None):
        self.env = env.FuturesEnv2(path='workspace/data/preprocess/BTCUSDT/BTCUSDT-5m.csv')
        self.agent, self.model_info = self._set_model() if further is None else self._load_model(further)
        self.logger = Logger(self.agent.model_name, f'workspace/logs/{self.agent.model_name}.log', log_level=LogLevel.ERROR)

        if self.agent is None:
            self.logger.error("모델을 로드할 수 없습니다.")
            exit(1)
        # 모델 정보 출력
        self.logger.render_model_info(self.model_info)
        input('학습을 시작합니다 [Enter]')

    def __select_model(self, model_path):
        models_dir = Path(model_path) 
        model_files = list(models_dir.glob('*.pth'))
        
        if not model_files:
            self.logger.error("사용 가능한 체크포인트를 찾을 수 없습니다.")
            exit(1)
        
        # 수정 시간 기준으로 정렬된 체크포인트 리스트 생성
        sorted_models = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)
        current_index = 0
        
        while True:
            current_model = sorted_models[current_index]
            print(f"\n현재 선택된 모델: {current_model}")
            user_input = input("이 모델로 학습을 진행하시겠습니까? (y/n): ")
            
            if user_input.lower() == 'y':
                return current_model

            elif user_input.lower() == 'n':
                current_index = (current_index + 1) % len(sorted_models)
                if current_index == 0:
                    print("\n모든 모델을 확인했습니다. 처음부터 다시 시작합니다.")
            else:
                print("학습을 취소합니다.")
                return None


    def _load_model(self, further):
        model_path = self.__select_model('workspace/models' if further else 'workspace/checkpoints')
        if model_path is None:
            return None, None
        
        model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        
        
        # 이전 학습 상태 확인
        model_name = model.get('model_name', 'ppo')
        state_dim = model.get('state_dim', self.env.observation_space.shape[0])
        #action_dim = checkpoint.get('action_dim', env.action_space.shape[0])
        action_dim = model.get('action_dim', self.env.action_space.n)
        self.checkpoint_term = model.get('checkpoint_term', 100)
        gamma = model.get('gamma', 0.99)
        epsilon = model.get('epsilon', 0.2)
        epochs = model.get('epochs', 10)
        current_episode = model.get('current_episode', 0)
        self.num_episodes = model.get('total_episodes', 2000)
        remaining_episodes = self.num_episodes - current_episode

        # 옵티마이저에서 학습률 가져오기
        optimizer_state = model['optimizer_state_dict']
        lr_actor = optimizer_state['param_groups'][0]['lr']  # actor의 학습률
        lr_critic = optimizer_state['param_groups'][3]['lr']  # critic의 학습률
        
        if further:
           
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
            epochs=epochs
        )

        # 모델 가중치 로드
        ppo_agent.actor_critic.load_state_dict(model['actor_critic_state_dict'])
        
        # 옵티마이저 상태 로드 (새로운 학습률 적용)
        optimizer_state = model['optimizer_state_dict']
        optimizer_state['param_groups'][0]['lr'] = lr_actor  # actor 학습률
        optimizer_state['param_groups'][3]['lr'] = lr_critic  # critic 학습률
        ppo_agent.optimizer.load_state_dict(optimizer_state)


        # 모델 정보 구성
        model_info = {
            'model_name': model.get('model_name', 'ppo'),
            'state_dim': state_dim,
            'action_dim': action_dim,
            'checkpoint_term': self.checkpoint_term,
            'gamma': gamma,
            'epsilon': epsilon,
            'epochs': epochs,
            'current_episode': current_episode,
            'total_episodes': self.num_episodes,
            'lr_actor': optimizer_state['param_groups'][0]['lr'],
            'lr_critic': optimizer_state['param_groups'][3]['lr'],
            'training_stats': model.get('training_stats', {})
        }
        
        

        return ppo_agent, model
        
    def _set_model(self):
        model_name = input('\n모델 저장 이름을 입력해주세요. [default: ppo]: ')
        self.num_episodes = int(input('학습할 에피소드 수를 입력해주세요. [default: 1000]: ') or "1000")
        self.checkpoint_term = int(input('체크포인트 저장 주기를 입력해주세요. [default: 100]: ') or "100")
        lr_actor = float(input('lr_actor [default: 3e-4]: ') or "3e-4")            
        lr_critic = float(input('lr_critic [default: 1e-3]: ') or "1e-3")
        gamma = float(input('gamma [default: 0.99]: ') or "0.99")
        epsilon = float(input('epsilon [default: 0.2]: ') or "0.2")
        epochs = int(input('epochs [default: 10]: ') or "10")

        ppo_agent = algo.PPO(
                state_dim=self.env.observation_space.shape[0],
                action_dim=self.env.action_space.n,
                model_name=model_name,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                epsilon=epsilon,
                epochs=epochs)

        # 새 모델의 초기 정보 구성
        model_info = {
            'model_name': model_name,
            'state_dim': self.env.observation_space.shape[0],
            'action_dim': self.env.action_space.n,
            'checkpoint_term': self.checkpoint_term,
            'gamma': gamma,
            'epsilon': epsilon,
            'epochs': epochs,
            'current_episode': 0,
            'total_episodes': self.num_episodes,
            'lr_actor': lr_actor,
            'lr_critic': lr_critic,
            'training_stats': {
                'total_steps': 0,
                'completed_episodes': 0,
                'win_rate': 0.0
            }
        }

        return ppo_agent, model_info
