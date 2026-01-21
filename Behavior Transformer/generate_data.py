"""
generate_data.py
Pendulum-v1 환경에서 랜덤 정책으로 데이터를 수집하고
BeT 학습을 위한 (N, Context_Len, Dim) 형태로 전처리하여 저장합니다.
"""
import gymnasium as gym
import numpy as np
import os

def get_expert_action(obs):
    """
    Pendulum을 세우기 위한 간단한 PD 제어기 (Rule-based Expert)
    obs: [cos(theta), sin(theta), theta_dot]
    """
    cos_th, sin_th, th_dot = obs
    # atan2를 통해 현재 각도 theta 추출 (0이 수직 상단)
    th = np.arctan2(sin_th, cos_th)
    
    # PD Control: 목표 각도(0)와 속도(0)로 수렴하도록 토크 계산
    # Torque = -Kp * theta - Kd * theta_dot
    torque = -2.0 * th - 1.0 * th_dot
    
    # 행동 범위 [-2, 2]로 클리핑
    return [np.clip(torque, -2.0, 2.0)]

def generate_pendulum_data(
    num_episodes=500, 
    context_len=10, 
    save_dir="data"
):
    env = gym.make("Pendulum-v1")
    
    all_obs_windows = []
    all_action_windows = []
    
    print(f"Collecting data from {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        episode_obs = []
        episode_actions = []
        
        # 1. 에피소드 실행 및 데이터 수집
        while not (done or truncated):
            # 전문가 행동 (Rule-based Policy) 사용
            action = get_expert_action(obs)
            
            episode_obs.append(obs)
            episode_actions.append(action)
            
            obs, reward, done, truncated, _ = env.step(action)
            
        # 2. Sliding Window로 데이터 자르기
        # 데이터 형태: (Time, Dim) -> (N_Windows, Context_Len, Dim)
        ep_obs_arr = np.array(episode_obs)       # (L, Obs_Dim)
        ep_act_arr = np.array(episode_actions)   # (L, Action_Dim)
        
        length = len(ep_obs_arr)
        if length >= context_len:
            for i in range(length - context_len + 1):
                # i부터 i+context_len까지 슬라이싱
                obs_window = ep_obs_arr[i : i + context_len]
                act_window = ep_act_arr[i : i + context_len]
                
                all_obs_windows.append(obs_window)
                all_action_windows.append(act_window)

    env.close()
    
    # 3. Numpy 배열로 변환 및 저장
    all_obs_windows = np.array(all_obs_windows, dtype=np.float32)
    all_action_windows = np.array(all_action_windows, dtype=np.float32)
    
    print(f"Data collection complete.")
    print(f"Obs shape: {all_obs_windows.shape}")     # (Total_N, 10, 3)
    print(f"Action shape: {all_action_windows.shape}") # (Total_N, 10, 1)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    np.save(os.path.join(save_dir, "pendulum_obs.npy"), all_obs_windows)
    np.save(os.path.join(save_dir, "pendulum_actions.npy"), all_action_windows)
    print(f"Saved to {save_dir}/")

if __name__ == "__main__":
    generate_pendulum_data()
