"""
config.py
BeT 모델 및 학습을 위한 하이퍼파라미터 정의
"""

class BeTConfig:
    # ----------------------------------
    # 1. Model Architecture
    # ----------------------------------
    # Context Length (Block Size): 과거 몇 프레임을 볼 것인가?
    # 논문에서는 과제에 따라 5 ~ 60 사이를 사용 (보통 10 권장)
    context_len = 10 
    
    # 관측값 차원 (환경에 따라 다름, 예: 로봇 팔 관절 각도 7개 + 위치 3개 = 10)
    input_dim = 3  # Pendulum: cos(theta), sin(theta), theta_dot
    
    # 행동 차원 (예: 관절 토크 7개)
    action_dim = 1  # Pendulum: Torque

    # Cluster 개수 (K): 논문에서는 24 ~ 100 사이 권장
    n_clusters = 10 # 행동 공간이 단순하므로 클러스터 수를 줄임

    # Transformer 크기 (MinGPT / GPT-2 Small 수준)
    n_layer = 4        # 레이어 깊이
    n_head = 4         # 어텐션 헤드 수
    n_embd = 128       # 임베딩 차원 (논문에서는 128~256 사용)
    
    # Dropout 확률
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    # ----------------------------------
    # 2. Training Hyperparameters
    # ----------------------------------
    batch_size = 128
    learning_rate = 1e-4  # AdamW 기본값
    weight_decay = 1e-4
    epochs = 50           # 데이터 양에 따라 조절
    
    # Loss Weights
    # 논문에서는 오프셋 로스에 가중치를 더 주기도 함 (보통 1.0 ~ 5.0)
    offset_loss_weight = 1.0 
    focal_alpha = 1.0
    focal_gamma = 2.0