from environment import FuturesEnv
from ppo import PPO
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from preprocess import preprocess_data


def train_ppo(env, ppo_agent, num_episodes=1000, max_steps=1000):
    episode_rewards = []
    all_episode_returns = []  # 모든 에피소드의 수익률 기록
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        was_liquidated = False  # 현재 에피소드의 청산 여부
        
        print(f"\n에피소드 {episode + 1} 시작")
        step = 0
        update_count = 0
        while True:
        #for step in range(max_steps):
            step += 1
            # 행동 선택
            action, value, log_prob = ppo_agent.select_action(state)
            #print(f"action: {action}")
            # 환경과 상호작용
            next_state, reward, done, info = env.step(action)
            
            # 청산 여부 확인
            if info['liquidated']:
                was_liquidated = True
                
            
            # 트랜지션 저장
            ppo_agent.store_transition((state, action, reward, next_state, log_prob, value, done))
            
            episode_reward += reward
            state = next_state
            
            if len(ppo_agent.memory) >= 512:  # 배치 크기에 도달하면 업데이트
                update_count += 1
                ppo_agent.update()
            
            if done:
                # 마지막 스텝까지 도달한 경우 결과 출력
                env.render()
                break
        print(f"  현재 step: {step}, 에피소드 보상: {episode_reward}, 업데이트 횟수: {update_count}")
        print(f"  청산 여부: {'청산됨' if was_liquidated else '정상 종료'}") 
        episode_rewards.append(episode_reward)
        
        # 현재 에피소드의 최종 수익률
        final_return = env.returns_history[-1]
        all_episode_returns.append(final_return)
    
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
    # 데이터 준비
    
    
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
    
    # 학습 결과 시각화
    plot_results(rewards_history, returns_history, all_episode_returns)
    
    # 모델 저장
    torch.save(ppo_agent.actor_critic.state_dict(), 'futures_trading_model.pth')