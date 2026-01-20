import numpy as np
import torch
from sklearn.cluster import KMeans

class BeTPreprocessor:
    def __init__(self, n_clusters=24): # 논문에서는 보통 24~100개 사이 사용
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.centers = None

    def fit(self, all_actions):
        """
        학습 데이터셋의 모든 행동을 모아서 K-Means 클러스터링을 수행합니다.
        
        Args:
            all_actions (np.array): (Total_Samples, Action_Dim) 크기의 전체 행동 데이터
        """
        print(f"Fitting KMeans with {self.n_clusters} clusters...")
        self.kmeans.fit(all_actions)
        self.centers = torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float32)
        print("KMeans fitting complete.")

    def process(self, actions):
        """
        개별 행동(또는 배치)을 모델 학습용 타겟(Label, Offset)으로 변환합니다.
        
        Args:
            actions (np.array or torch.Tensor): (Batch, Action_Dim)
            
        Returns:
            target_labels (torch.LongTensor): (Batch,) - 가장 가까운 클러스터 인덱스
            true_offsets (torch.FloatTensor): (Batch, Action_Dim) - (실제 행동 - 클러스터 중심)
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        # 1. 가장 가까운 클러스터 찾기 (predict)
        # labels: (Batch,)
        labels = self.kmeans.predict(actions)
        
        # 2. 해당 클러스터의 중심 좌표 가져오기
        # chosen_centers: (Batch, Action_Dim)
        # self.kmeans.cluster_centers_는 numpy array임
        chosen_centers = self.kmeans.cluster_centers_[labels]

        # 3. Offset 계산 (실제 행동 - 클러스터 중심)
        offsets = actions - chosen_centers

        return torch.tensor(labels, dtype=torch.long), torch.tensor(offsets, dtype=torch.float32)

    def get_centers(self):
        """모델 초기화 시 등록할 버퍼(Buffer)용 센터 반환"""
        return self.centers
    
from torch.utils.data import Dataset, DataLoader

class RobotBehaviorDataset(Dataset):
    def __init__(self, observations, actions, preprocessor):
        """
        observations: (N, Sequence_Length, Obs_Dim)
        actions: (N, Sequence_Length, Action_Dim)
        preprocessor: 학습된 BeTPreprocessor 인스턴스
        """
        self.observations = observations
        self.actions = actions
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        # 1. 원본 데이터 가져오기
        obs_seq = torch.FloatTensor(self.observations[idx]) # (T, Obs_Dim)
        action_seq = torch.FloatTensor(self.actions[idx])   # (T, Action_Dim)

        # 2. 행동 데이터를 BeT 학습용 타겟으로 변환
        # Sequence 전체에 대해 한 번에 처리 (T, Action_Dim) -> (T,), (T, Action_Dim)
        target_labels, true_offsets = self.preprocessor.process(action_seq)

        return {
            "observations": obs_seq,
            "target_labels": target_labels, # Classification 정답
            "true_offsets": true_offsets    # Regression 정답
        }