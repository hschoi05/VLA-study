"""
train.py
BeT 모델 학습을 위한 메인 스크립트
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

# 모듈 임포트 (앞서 정리한 파일 구조 가정)
from config import BeTConfig
from model import BehaviorTransformer
from loss import BeTLoss
from data import BeTPreprocessor, RobotBehaviorDataset

def train():
    # 1. 설정 및 디바이스 준비
    cfg = BeTConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}...")

    # 2. 데이터 준비 (Dummy Data 예시)
    # 실제 프로젝트에서는 .npy 파일을 로드하거나 파싱해야 합니다.
    # 예: total_obs (N, T, input_dim), total_actions (N, T, action_dim)
    print("Loading Pendulum data...")
    data_dir = "data"
    obs_path = os.path.join(data_dir, "pendulum_obs.npy")
    act_path = os.path.join(data_dir, "pendulum_actions.npy")

    if not os.path.exists(obs_path) or not os.path.exists(act_path):
        raise FileNotFoundError("Data files not found. Please run generate_data.py first.")

    raw_obs = np.load(obs_path)
    raw_actions = np.load(act_path)
    print(f"Loaded Obs: {raw_obs.shape}, Actions: {raw_actions.shape}")

    # 3. 데이터 전처리 (K-Means Clustering)
    print("Fitting KMeans for action clustering...")
    # Flatten actions to (NT, Action_Dim) for clustering
    flat_actions = raw_actions.reshape(-1, cfg.action_dim)
    
    preprocessor = BeTPreprocessor(n_clusters=cfg.n_clusters)
    preprocessor.fit(flat_actions) # 학습 데이터 전체로 클러스터 센터 찾기

    # 4. Dataset & DataLoader 생성
    train_dataset = RobotBehaviorDataset(raw_obs, raw_actions, preprocessor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        num_workers=4
    )

    # 5. 모델 초기화
    # 중요: Preprocessor가 찾은 클러스터 센터를 모델에 등록해야 함
    # (K, Action_Dim)
    cluster_centers = preprocessor.get_centers().to(device)
    
    model = BehaviorTransformer(
        config=cfg, 
        n_clusters=cfg.n_clusters, 
        kmeans_centers=cluster_centers
    ).to(device)

    # 6. Optimizer & Loss Function 설정
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.learning_rate, 
        weight_decay=cfg.weight_decay
    )
    
    loss_fn = BeTLoss(
        action_dim=cfg.action_dim, 
        n_clusters=cfg.n_clusters,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma
    )

    # 7. 학습 루프 (Training Loop)
    print("Starting training loop...")
    model.train()

    for epoch in range(cfg.epochs):
        total_epoch_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 데이터를 GPU로 이동
            obs = batch['observations'].to(device)       # (B, T, Input_Dim)
            target_labels = batch['target_labels'].to(device) # (B, T)
            true_offsets = batch['true_offsets'].to(device)   # (B, T, Action_Dim)

            # (A) Forward Pass
            # pred_logits: (B, T, K)
            # pred_offsets: (B, T, K, Action_Dim)
            pred_logits, pred_offsets = model(obs)

            # (B) Loss Calculation
            loss, cls_loss, offset_loss = loss_fn(
                pred_logits, 
                pred_offsets, 
                target_labels, 
                true_offsets
            )

            # (C) Backward Pass & Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{cfg.epochs}] Batch {batch_idx}: "
                      f"Loss {loss.item():.4f} (Cls: {cls_loss.item():.4f}, Off: {offset_loss.item():.4f})")

        avg_loss = total_epoch_loss / len(train_loader)
        print(f"==> End of Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    # 8. 모델 저장
    print("Saving model...")
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
        
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': cfg,
        'cluster_centers': cluster_centers.cpu() # 센터값도 같이 저장하는 것이 안전함
    }, "checkpoints/bet_model_final.pth")
    
    print("Training finished successfully.")

if __name__ == "__main__":
    train()