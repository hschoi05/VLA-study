import torch
import torch.nn as nn
from mingpt.model import GPT

class BehaviorTransformer(nn.Module):
    def __init__(self, config, n_clusters, kmeans_centers):
        super().__init__()
        self.config = config
        # 1. GPT Backbone (Context Encoder)
        self.transformer = GPT(config)

        # K-Means로 미리 계산된 클러스터 센터 (Action Centers)
        # (K, Action_Dim)
        self.register_buffer('cluster_centers', 
                             kmeans_centers if isinstance(kmeans_centers, torch.Tensor) 
                             else torch.tensor(kmeans_centers))
        
        # 2. Prediction Heads
        # (A) Classification Head: predict Action Center
        self.class_head = nn.Linear(config.n_embd, n_clusters)
        
        # (B) Offset Head: predict Offset from Center
        # 입력 차원 당 n_clusters개의 오프셋을 예측 (선택된 클러스터 것만 사용)
        self.offset_head = nn.Linear(config.n_embd, n_clusters * config.action_dim)

    def forward(self, x):
        """
        x shape: (Batch, Sequence_Length, Input_Dim)
        """
        
        # feature extraction via Transformer
        # (B, T, input_dim) -> (B, T, n_embd)
        features = self.transformer(x)

        # Main Logics
        
        # 1. Predict Logits (Focal Loss)
        # (B, T, n_embd) -> (B, T, K)
        class_logits = self.class_head(features) 

        # 2. Predict Offsets (Masked MSE Loss)
        # 모든 클러스터에 대한 오프셋을 예측한 뒤, 나중에 정답 클러스터 것만 선택
        # (B, T, K * action_dim) -> (B, T, K, action_dim)
        all_offsets = self.offset_head(features)
        all_offsets = all_offsets.view(features.shape[0], features.shape[1], -1, self.config.action_dim)

        # Inference 시의 최종 행동 산출:
        # 가장 확률 높은 클러스터 인덱스(k)를 찾고 -> Center[k] + Offset[k]
        
        return class_logits, all_offsets