from train import train
import torch
from Loader import Loader

if __name__ == "__main__":
    
    while True: 
        print('학습 방법을 선택해주세요.')
        print('(1) 학습')
        print('(2) 재개')
        print('(3) 추가 학습')
        print('(4) 디바이스 확인')
        print('(5) 종료')
        
        choice = int(input('선택: '))
        
        if choice == 1:
            loader = Loader()
            train(loader.env, loader.agent, loader.num_episodes, loader.model_info, logger=loader.logger)
            break

        elif choice == 2:
            loader = Loader(further=False)
            train(loader.env, loader.agent, loader.num_episodes, loader.model_info, logger=loader.logger)
            break

        elif choice == 3:
            loader = Loader(further=True)
            train(loader.env, loader.agent, loader.num_episodes, loader.model_info, logger=loader.logger)
            break

        elif choice == 4:
            print("\n현재 사용 가능한 Device : ", "cuda\n" if torch.cuda.is_available() else "cpu\n")

        elif choice == 5:
            print("\n프로그램을 종료합니다.\n")
            exit()

