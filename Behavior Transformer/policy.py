import torch
import torch.nn.functional as F

class BeTPolicy:
    def __init__(self, model):
        self.model = model
        self.model.eval() # 평가 모드로 전환

    def get_action(self, obs_seq, deterministic=True):
        """
        obs_seq: (Batch, Sequence_Length, Obs_Dim) - 현재까지의 관측 히스토리
        deterministic: True면 확률이 가장 높은 행동 선택, False면 확률 분포에 따라 샘플링
        """
        with torch.no_grad():
            # 1. 모델 Forward
            # class_logits: (B, T, K)
            # all_offsets: (B, T, K, Action_Dim)
            class_logits, all_offsets = self.model(obs_seq)

            # 우리는 가장 마지막 타임스텝(T-1)의 예측값만 필요함
            last_logits = class_logits[:, -1, :]        # (B, K)
            last_offsets = all_offsets[:, -1, :, :]     # (B, K, Action_Dim)

            # 2. 클러스터 선택 (Classification)
            probs = F.softmax(last_logits, dim=-1) # 확률로 변환
            
            if deterministic:
                # 가장 확률이 높은 클러스터 선택 (Greedy)
                cluster_idxs = torch.argmax(probs, dim=-1) # (B,)
            else:
                # 확률 분포에 기반하여 샘플링 (Stochastic) - 탐험(Exploration)이 필요할 때 사용
                cluster_idxs = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # 3. 해당 클러스터의 Center와 Offset 가져오기
            
            # (A) Center 가져오기
            # model.cluster_centers는 (K, Action_Dim)
            selected_centers = self.model.cluster_centers[cluster_idxs] # (B, Action_Dim)

            # (B) Offset 가져오기
            # last_offsets 중에서 선택된 cluster_idxs에 해당하는 것만 뽑아야 함
            # indexing: [0~Batch_Size, cluster_idxs]
            batch_range = torch.arange(last_offsets.shape[0], device=last_offsets.device)
            selected_offsets = last_offsets[batch_range, cluster_idxs] # (B, Action_Dim)

            # 4. 최종 행동 복원 (Reconstruction)
            predicted_action = selected_centers + selected_offsets

            return predicted_action.cpu().numpy()