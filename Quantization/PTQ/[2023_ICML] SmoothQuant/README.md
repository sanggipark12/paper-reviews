# [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models]

> Links: ArXiv | Official Code
> 
> Tags: #Quantization, #LLM, #PTQ, #W8A8, #Efficient_Inference
> 
> Author: Guangxuan Xiao et al. (MIT, NVIDIA) 
> 
> Date: 2023 (ICML 2023), Updated 2024

## 1. 기존 연구 및 한계 (Related Work & Limitations)

- **기존 접근 방식**:
    
    - 모델 크기가 커짐에 따라 메모리와 연산 비용을 줄이기 위해 양자화가 필수적임.
        
    - **ZeroQuant**: 토큰 단위 활성화 양자화와 그룹 단위 가중치 양자화를 제안했으나, 초대형 모델(OPT-175B 등)에서는 정확도가 깨짐.
        
    - **LLM.int8()**: 이상치를 FP16으로 따로 처리하는 혼합 정밀도 방식을 사용해 정확도는 잡았으나, 연산 오버헤드로 인해 추론 속도가 오히려 느려지는 문제가 있음.
        
- **기존 방식의 한계점**:
    
    - **활성화 양자화의 어려움**: 6.7B 이상의 LLM에서는 활성화값에 100배 이상 큰 이상치가 발생하여, 일반적인 INT8 양자화 시 정보 손실이 큼.
        
    - **하드웨어 제약**: 정확도를 높이려면 채널별 활성화 양자화가 이상적이지만, 이는 **INT8 GEMM 커널(행렬 연산 가속기)과 호환되지 않아** 속도 향상을 볼 수 없음.
        

## 2. 핵심 기여 및 방법론 (Key Contribution & Methodology)

### 핵심 아이디어 (Core Idea)

- **"SmoothQuant offline migrates the quantization difficulty from activations to weights."**
    
- 활성화(Activation)는 양자화하기 어렵고(이상치 때문), 가중치(Weight)는 양자화하기 쉽다(분포가 균일함). 따라서 **활성화의 "양자화 난이도(이상치의 크기)"를 수학적으로 가중치 쪽으로 밀어버리자!**
    

### 관찰 및 방법론 (Observation & Methodology)

- **핵심 관찰**:
    
    1. 이상치는 특정 **채널**에 고정되어 나타난다. (어떤 토큰이 오든 그 채널은 항상 값이 큼)
        
    2. 이상치 채널 내부의 **분산은 작다**. (값이 항상 크기 때문에 스케일링으로 줄이기 쉬움)
        
- **스무딩**:
    
    - 하드웨어 효율적인 **Per-tensor** 또는 **Per-token** 양자화를 쓰면서도, 마치 **Per-channel** 양자화를 한 것 같은 효과를 내기 위해 입력 $X$를 채널별로 스케일링($s$)함.
        
    - 수식: $Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \cdot \hat{W}$
        
        - $X$ (활성화): $s$로 나누어 이상치를 평탄화(Smoothing) → **양자화 쉬워짐**
            
        - $W$ (가중치): $s$를 곱해줌 → 양자화 난이도가 약간 올라가지만 여전히 할만함
            
    - 이때 $s$는 이전 레이어의 파라미터에 미리 융합하여 런타임 오버헤드를 없앰.
        
- **Migration Strength ($\alpha$)**:
    
    - 난이도를 얼마나 옮길지 결정하는 하이퍼파라미터 $\alpha$ 도입.
        
    - $s_j = \max(|X_j|)^\alpha / \max(|W_j|)^{1-\alpha}$
        
    - $\alpha=0.5$는 난이도를 똑같이 나누는 지점이며, 대부분의 모델에서 최적.
        

## 3. 결과 (Results)

- **성능 지표 (Accuracy & Efficiency)**:
    
    - **정확도**: OPT-175B, BLOOM-176B, GLM-130B, MT-NLG 530B 등 초대형 모델에서 FP16과 동등한 정확도 유지 (W8A8 적용 시).
        
    - **속도**: PyTorch 구현 기준 최대 **1.51배**, FasterTransformer 기준 최대 **1.56배** 추론 속도 향상.
        
    - **메모리: FP16 대비 메모리 사용량 **2배 감소**.
        
    - **Scalability**: 8개의 A100 GPU(단일 노드)로 **530B 모델** 서빙 가능 (기존엔 불가능).
        