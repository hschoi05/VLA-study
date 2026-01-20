import gym
import torch
import numpy as np
from collections import deque

class BeTGymRunner:
    def __init__(self, env_name, policy, context_len=10, device='cpu'):
        """
        Args:
            env_name: Gym 환경 이름 (예: 'Ant-v4', 'Hopper-v4')
            policy: 앞서 구현한 BeTPolicy 인스턴스
            context_len: 모델이 학습할 때 사용한 시퀀스 길이 (Block Size)
        """
        self.env = gym.make(env_name)
        self.policy = policy
        self.context_len = context_len
        self.device = device
        
        # 최근 N개의 관측값을 저장할 큐 (꽉 차면 가장 오래된 것 자동 삭제)
        self.obs_buffer = deque(maxlen=context_len)

    def reset(self):
        """에피소드 시작 시 버퍼 초기화 (Padding)"""
        obs = self.env.reset()
        
        # Gym 버전에 따라 (obs, info)를 반환할 수도 있으므로 처리
        if isinstance(obs, tuple):
            obs = obs[0]

        # 초기에는 과거 기록이 없으므로, 첫 프레임을 context_len 만큼 복제해서 채움
        # 예: [obs_t0, obs_t0, ..., obs_t0]
        self.obs_buffer.clear()
        for _ in range(self.context_len):
            self.obs_buffer.append(obs)
            
        return obs

    def run_episode(self, render=False):
        """한 에피소드를 실행하고 총 보상을 반환"""
        obs = self.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            if render:
                self.env.render()

            # 1. 모델 입력 데이터 준비 (Numpy -> Tensor)
            # 버퍼에 있는 [obs_t-9, ..., obs_t]를 하나의 배열로 변환
            # shape: (Context_Len, Obs_Dim)
            obs_seq = np.array(self.obs_buffer)
            
            # 배치 차원 추가: (1, Context_Len, Obs_Dim)
            obs_tensor = torch.FloatTensor(obs_seq).unsqueeze(0).to(self.device)

            # 2. BeT 모델 추론 (Action Prediction)
            # policy.get_action은 (Action_Dim,) 형태의 Numpy 배열 반환
            action = self.policy.get_action(obs_tensor, deterministic=True)
            
            # 만약 action이 (1, Dim) 형태라면 squeeze 필요
            if len(action.shape) > 1:
                action = action.squeeze()

            # 3. 환경 Step 실행
            try:
                # Gym 버전에 따라 반환값이 4개 또는 5개일 수 있음
                step_result = self.env.step(action)
                next_obs, reward, done, *_ = step_result # 유연하게 처리
            except Exception as e:
                print(f"Error during env.step: {e}")
                break

            # 4. 버퍼 업데이트 (FIFO)
            # 새로운 관측값을 큐에 넣으면, 가장 오래된 관측값은 자동으로 밀려남
            self.obs_buffer.append(next_obs)
            
            total_reward += reward
            obs = next_obs
            step_count += 1
            
            # 안전장치 (무한 루프 방지)
            if step_count > 1000: 
                break

        print(f"Episode finished. Total Reward: {total_reward}")
        return total_reward

    def close(self):
        self.env.close()