# [Up or Down? Adaptive Rounding for Post-Training Quantization]

> **Links**: [ArXiv](https://arxiv.org/abs/2004.10568) | [Official Code](https://www.google.com/search?q=https://github.com/qualcomm-ai-research/adaround) **Tags**: #Quantization, #Post-Training_Quantization, #Neural_Network_Compression, #AdaRound
> 
> **Author**: Markus Nagel, Rana Ali Amjad, Mart van Baalen, Christos Louizos, Tijmen Blankevoort (Qualcomm AI Research)
> **Date**: 2020-06-30  

## 1. 기존 연구 (Related Work)

- **기존의 접근 방식**:
    
    - **Rounding-to-nearest**: 신경망의 부동 소수점 가중치를 고정 소수점 값으로 양자화할 때, 가장 가까운 값으로 반올림하는 것이 지배적인 접근 방식이었습니다.  
        
    - **Post-Training Quantization (PTQ)**: 모델 재학습이나 하이퍼파라미터 튜닝이 필요한 Quantization-Aware Training(QAT)과 달리, PTQ는 훈련된 모델을 빠르게 양자화하여 배포할 수 있어 실용적입니다.  
        
- **기존 방식의 한계점**:
    
    - **Rounding-to-nearest의 최적성 결여**: 가중치 행렬의 개별 값 차이를 최소화하는 것이 가장 합리적으로 보이지만, 실제로는 가중치 간의 상호작용(Task Loss의 Hessian off-diagonal term)을 무시하기 때문에 최적의 성능을 보장하지 않습니다.  
        
    - **성능 저하**: 단순히 가장 가까운 값으로 반올림하는 방식은 특히 4비트와 같은 낮은 비트 수에서 치명적인 성능 저하를 초래할 수 있습니다.  
        

## 2. 기존 연구와의 차이점 (Key Contribution)

### 핵심 아이디어 (Core Idea)

- 단순히 가장 가까운 값으로 반올림하는 대신, **입력 데이터와 Task Loss에 적응(Adaptive)하여 가중치를 올림(Up) 또는 내림(Down)할지 결정하는 새로운 메커니즘인 'AdaRound'를 제안**합니다.  
    

### 방법론 및 아키텍처 (Methodology & Architecture)

- **이론적 배경 (Taylor Expansion & QUBO)**:
    
    - 사전 학습된 신경망의 가중치 변동에 따른 Task Loss를 2차 Taylor 급수로 근사하여 분석합니다.  
        
    - 이 문제를 2차 비제약 이진 최적화(Quadratic Unconstrained Binary Optimization, QUBO) 문제로 정식화합니다.  
        
- **Local MSE Loss로의 단순화**:
    
    - 전체 Task Loss를 최적화하는 것은 계산 비용이 높으므로, 이를 레이어별(Layer-wise) Local Loss 최적화 문제로 단순화합니다.  
        
    - 최종적으로는 출력 활성화 값(Pre-activation)의 Mean Squared Error (MSE)를 최소화하는 문제로 귀결됩니다.  
        
- **AdaRound 알고리즘 (Continuous Relaxation)**:
    
    - 이산적인 최적화 문제를 해결하기 위해 **Soft Relaxation**을 적용합니다.  
        
    - 학습 가능한 연속 변수 $V_{i,j}$와 Rectified Sigmoid 함수를 사용하여 최적화하며, 정규화 항(Regularization term)을 통해 최종적으로 가중치가 0 또는 1(올림 또는 내림)로 수렴하도록 유도합니다.  
        
    - 전체 데이터를 사용할 필요 없이 소량의 레이블 없는 데이터(Unlabeled data)만으로 빠르고 효율적으로 최적화가 가능합니다.  
        

## 3. 결과 (Results)

- **성능 지표**:
    
    - ResNet18과 ResNet50 모델을 미세 조정(Fine-tuning) 없이 4비트로 양자화했을 때, 정확도 손실을 1% 이내로 유지하며 기존 Rounding-to-nearest 방식(약 24% 정확도) 대비 압도적인 성능(약 68~69% 정확도)을 보였습니다.  
        
    - MobileNetV2, InceptionV3 등 다양한 모델에서도 새로운 State-of-the-art(SOTA) 성능을 달성했습니다.  
        
- **Ablation Study**:
    
    - **데이터 강건성**: ImageNet 데이터 1024장만 사용하여도 충분한 성능을 내며, 심지어 256장이나 다른 도메인의 데이터를 사용해도 성능 저하가 미미합니다.  
        
    - **재구성 손실(Reconstruction Loss)**: 비대칭 재구성을 사용했을 때 단순 MSE보다 성능이 향상됨을 확인했습니다.

	- Rounding-to-nearest vs AdaRound 정확도 비교 (ImageNet)