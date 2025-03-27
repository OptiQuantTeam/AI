from environment import FuturesEnv
from ppo import PPO
from train import train
from resume import resume
from continue_train import continue_training
from pathlib import Path
import torch
import sys

if __name__ == "__main__":

    env = FuturesEnv(path='/workspace/data/preprocess/BTCUSDT/BTCUSDT-1h.csv')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    while True: 
        print('학습 방법을 선택해주세요.')
        print('(1) 학습')
        print('(2) 재개')
        print('(3) 추가 학습')
        print('(4) 디바이스 확인')
        print('(5) 종료')
        
        choice = int(input('선택: '))
        
        if choice == 1:
            model_name = input('\n모델 저장 이름을 입력해주세요. [default: ppo]: ')
            num_episodes = int(input('학습할 에피소드 수를 입력해주세요. [default: 1000]: ') or "1000")
            checkpoint_term = int(input('체크포인트 저장 주기를 입력해주세요. [default: 100]: ') or "100")
            lr_actor = float(input('lr_actor [default: 3e-4]: ') or "3e-4")            
            lr_critic = float(input('lr_critic [default: 1e-3]: ') or "1e-3")
            gamma = float(input('gamma [default: 0.99]: ') or "0.99")
            epsilon = float(input('epsilon [default: 0.2]: ') or "0.2")
            epochs = int(input('epochs [default: 10]: ') or "10")
            input('학습을 시작합니다 [Enter]')

            ppo_agent = PPO(
                state_dim=state_dim,
                action_dim=action_dim,
                model_name=model_name,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                epsilon=epsilon,
                epochs=epochs)
            
            train(env, ppo_agent, num_episodes, checkpoint_term=checkpoint_term)
            break

        elif choice == 2:
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
                resume(env,latest_checkpoint)

            else:
                print("학습을 재개하지 않습니다.") 
            break

        elif choice == 3:
            # 최신 모델 찾기
            checkpoint_dir = Path('futures_rl/models')
            model_files = list(checkpoint_dir.glob('*.pth'))
            
            if not model_files:
                print("최종 모델 파일을 찾을 수 없습니다.")
                exit(1)
            
            # 수정 시간 기준으로 정렬된 모델 리스트 생성
            sorted_models = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)
            current_index = 0
            
            while True:
                current_model = sorted_models[current_index]
                print(f"\n현재 선택된 모델: {current_model}")
                user_input = input("이 모델로 추가 학습을 진행하시겠습니까? (y/n): ")
                
                if user_input.lower() == 'y':
                    try:
                        episodes = int(input("추가로 학습할 에피소드 수를 입력하세요 (기본값: 1000): "))
                    except ValueError:
                        episodes = 1000
                        print(f"기본값 {episodes}로 설정됩니다.")
                    
                    # 추가 학습 시작
                    continue_training(env, current_model, episodes)
                    break
                elif user_input.lower() == 'n':
                    current_index = (current_index + 1) % len(sorted_models)
                    if current_index == 0:
                        print("\n모든 모델을 확인했습니다. 처음부터 다시 시작합니다.")
                else:
                    print("학습을 취소합니다.")
                    break
            break

        elif choice == 4:
            print("\n현재 사용 가능한 Device : ", "cuda\n" if torch.cuda.is_available() else "cpu\n")

        elif choice == 5:
            print("\n프로그램을 종료합니다.\n")
            exit()

