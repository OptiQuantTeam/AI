import matplotlib.pyplot as plt
import numpy as np

def plot_balance_history(episode_results, path):
    plt.figure(figsize=(10, 6))
    plt.plot(episode_results, marker='o', markersize=3)
    plt.title('Balance History')
    plt.xlabel('Episode')
    plt.ylabel('Balance')
    plt.grid(True, alpha=0.3)
    plt.savefig(path)
    plt.close()

def plot_cumulative_result(episode_results, save_path):
    # 폰트 설정 제거
    plt.rcParams['axes.unicode_minus'] = False
    
    # 성공/실패 횟수 계산
    total_attempts = len(episode_results) - 1  # 초기 0 제외
    success_count = sum(episode_results)
    failure_count = total_attempts - success_count
    
    # 성공률 계산
    success_rate = (success_count / total_attempts * 100) if total_attempts > 0 else 0
    
    # 성공 횟수 누적 그래프
    cumulative_success = np.cumsum(episode_results)
    
    # 그래프 생성
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 성공 횟수 누적 그래프
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(range(len(cumulative_success)), cumulative_success, 'b-', linewidth=2)
    ax1.set_title('Cumulative Success Count', fontsize=12, pad=20)
    ax1.set_xlabel('Episode', fontsize=10)
    ax1.set_ylabel('Success Count', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 성공/실패 횟수 막대 그래프
    ax2 = plt.subplot(2, 2, 2)
    bars = ax2.bar(['Success', 'Failure'], [success_count, failure_count], 
                   color=['green', 'red'], alpha=0.7)
    ax2.set_title('Success vs Failure Count', fontsize=12, pad=20)
    ax2.set_ylabel('Count', fontsize=10)
    
    # 막대 위에 숫자 표시
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # 3. 성공률 추이 그래프
    ax3 = plt.subplot(2, 2, 3)
    window = 10
    success_rates = []
    for i in range(window, len(episode_results)):
        rates = episode_results[i-window:i]
        success_rates.append(sum(rates) / len(rates) * 100)
    
    ax3.plot(range(window, len(episode_results)), success_rates, 'b-', linewidth=2)
    ax3.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    ax3.set_title(f'Success Rate Trend ({window} episodes)', fontsize=12, pad=20)
    ax3.set_xlabel('Episode', fontsize=10)
    ax3.set_ylabel('Success Rate (%)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 통계 정보
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    stats_text = (
        f"Total Attempts: {total_attempts}\n"
        f"Success Count: {success_count}\n"
        f"Failure Count: {failure_count}\n"
        f"Success Rate: {success_rate:.2f}%\n"
        f"Best Success Rate: {max(success_rates):.2f}%\n"
        f"Worst Success Rate: {min(success_rates):.2f}%"
    )
    ax4.text(0.1, 0.5, stats_text, fontsize=12, 
             verticalalignment='center', linespacing=1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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