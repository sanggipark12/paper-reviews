# [BRECQ] Pushing the Limit of Post-Training Quantization by Block Reconstruction
### Links: ArXiv | Official Code Tags: #Quantization, #PTQ, #Compression Author: Yuhang Li et al. (SenseTime Research) Date: 2021-05-01 (ICLR 2021)

## 1. 기존 연구 (Related Work)
### 기존의 접근 방식:

QAT (Quantization-Aware Training): 전체 데이터셋을 사용하여 모델을 재학습시키는 방식. 성능은 좋으나 학습 비용(시간, 자원)이 큼.   

PTQ (Post-Training Quantization): 재학습 없이 적은 데이터(Calibration data)만으로 양자화하는 방식. 빠르고 가볍지만 낮은 비트(Low-bit)에서 성능 저하가 심함.   


Layer-wise Reconstruction: 기존 PTQ(AdaRound 등)는 각 레이어를 독립적으로 보고 재구성 오차를 최소화함.   

### 기존 방식의 한계점:

Layer-wise: 레이어 간의 상호 의존성(Cross-layer dependency)을 무시하기 때문에 4비트 이하(특히 2비트)로 갈수록 정확도가 급격히 떨어짐.   


Net-wise: 전체 네트워크 출력을 맞추려 하면, 적은 데이터로 인해 과적합(Overfitting)이 발생하여 일반화 성능이 떨어짐.   

## 2. 기존 연구와의 차이점 (Key Contribution)
### 핵심 아이디어 (Core Idea)
Block Reconstruction: Layer 단위는 너무 좁고, Network 단위는 너무 넓다. 신경망의 기본 구성 요소인 '블록(Block, 예: Residual Block)' 단위로 재구성하는 것이 정확도와 일반화 성능 사이의 최적의 균형(Trade-off)이다.   

최초로 PTQ 방식에서 INT2 (2-bit) 양자화 성능을 유의미한 수준으로 끌어올림.   

### 방법론 및 아키텍처 (Methodology & Architecture)
Second-order Error Analysis: 테일러 전개를 통한 손실 함수 분석에서, 전체 Hessian 행렬을 사용하는 것은 불가능하므로 근사가 필요함. 저자들은 Block-diagonal Hessian 근사가 Cross-layer dependency를 적절히 반영하면서도 계산 가능하다는 것을 증명함.   


Fisher Information Matrix (FIM): 재구성 시 단순 MSE가 아닌, FIM을 가중치로 사용하여 기울기(Gradient) 정보가 큰 중요한 특징에 더 집중하도록 함.   

Mixed Precision: 민감도(Sensitivity)에 따라 레이어별로 비트수(2/4/8 bit)를 다르게 할당. 유전 알고리즘(Genetic Algorithm)을 사용하여 정확도와 하드웨어 효율성(Latency/Size)을 동시에 최적화함.   

## 3. 결과 (Results)
### 성능 지표:

ResNet-18 (4-bit): Top-1 Accuracy 69.60% 달성 (Full Precision 71.08% 대비 손실 미비). 이는 QAT 방식(PACT, DSQ)과 동등한 수준임.   


속도: QAT 방식이 100 GPU hours가 걸리는 반면, BRECQ는 0.4 GPU hours 만에 완료됨 (약 240배 빠름).   


INT2 성능: 기존 방법론들이 2-bit에서 정확도가 0~55%로 무너질 때, BRECQ는 ResNet-18에서 **66.30%**를 달성함.   


Ablation Study: Reconstruction 단위를 Layer, Block, Stage, Net으로 비교했을 때 Block 단위가 가장 성능이 좋음을 실험적으로 검증함.   
