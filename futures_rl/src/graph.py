import matplotlib.pyplot as plt
import numpy as np

def plot_balance_history(episode_results, path):
    #plt.figure(figsize=(15, 10))
    
    # 보상 그래프
    #plt.subplot(2, 1, 1)
    plt.plot(episode_results, marker='o')
    plt.title('balance Graph')
    plt.xlabel('episode')
    plt.ylabel('balance')
    
    
    #plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_cumulative_result(cumulative_results, path):
    # 누적합 계산
    cumsum = np.cumsum(cumulative_results)
    
    # 이전 값보다 큰 지점 찾기
    increasing_points = []
    for i in range(1, len(cumsum)):
        if cumsum[i] > cumsum[i-1]:
            increasing_points.append(i)
    
    # 기본 누적합 그래프
    plt.plot(cumsum, label='Cumulative Result')
    '''
    # 증가하는 지점에 마커 추가
    if increasing_points:
        plt.plot(increasing_points, cumsum[increasing_points], 
                'o', color='red', markersize=4, 
                label='Increasing Points', alpha=0.5)
    '''
    plt.title(f'Cumulative Result Graph : [{cumsum[-1]}/{len(cumsum)-1}]')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(path)
    plt.close()

def plot_cumulative_result_log(cumulative_results, path):
    # 누적합 계산
    cumsum = np.cumsum(cumulative_results)
    
    # 이전 값보다 큰 지점 찾기
    increasing_points = []
    for i in range(1, len(cumsum)):
        if cumsum[i] > cumsum[i-1]:
            increasing_points.append(i)
    
    # 기본 누적합 그래프
    plt.plot(cumsum, label='Cumulative Result')
    
    # 증가하는 지점에 마커 추가
    if increasing_points:
        plt.plot(increasing_points, cumsum[increasing_points], 
                'o', color='red', markersize=4, 
                label='Increasing Points', alpha=0.5)
    
    plt.title(f'Cumulative Result Graph : [{cumsum[-1]}/{len(cumsum)-1}]')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Return')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(path)
    plt.close()