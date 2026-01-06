# [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration]

> Links: ArXiv | Official Code
> 
> Tags: #Quantization, #LLM, #W4A16, #Efficient_Inference, #TinyChat
> 
> Author: Ji Lin et al. (MIT, NVIDIA, Tsinghua Univ.)
> 
> Date: 2024 (MLSys Best Paper)

## 1. 기존 연구 및 한계 (Related Work & Limitations)

- **기존 접근 방식 :
    
    - **RTN (Round-to-Nearest)**: 가장 단순한 반올림 방식이나 성능 저하가 심함.
        
    - **GPTQ**: Hessian(2차 미분 정보)을 활용하여 양자화 오차를 보정하는 방식으로 성능이 좋음.
        
- **기존 방식의 한계점**:
    
    - **QAT (Quantization-aware Training)**: LLM 규모에서는 훈련 비용이 너무 비싸 비현실적임.
        
    - **GPTQ의 한계**: 보정 데이터셋에 과적합되어, 훈련 분포 밖의 데이터에 대해서는 일반화 성능이 떨어질 수 있음.
        
    - **하드웨어 비효율성**: 일부 중요한 가중치만 FP16으로 유지하는 혼합 정밀도 방식은 하드웨어 구현이 어렵고 느림.
        

## 2. 핵심 기여 및 방법론 (Key Contribution & Methodology)

### 핵심 아이디어 (Core Idea)

- **"We should protect salient weights by scaling them up."**
    
- 모든 가중치가 동등하게 중요한 것은 아님. **활성화 값이 큰 채널의 가중치가 더 중요**하며, 이들을 수학적으로 스케일링하여 양자화 오차를 줄여야 함.
    
- 역전파나 재구성 없이 스케일링만으로 성능을 보존하므로 과적합이 없음.
    

### 방법론 및 아키텍처 (Methodology & Architecture)

1. **핵심 관찰 :
    
    - **가중치의 중요도**: 전체 가중치의 0.1%~1%만 FP16으로 유지해도 양자화 성능이 비약적으로 상승함.
        
    - **중요도의 기준**: 가중치의 크기가 아니라 활성화의 크기가 큰 채널이 가지고 있음.
        
2. **Activation-aware Scaling**:
    
    - 중요한 가중치를 FP16으로 남기면 하드웨어가 느려지므로, 대신 **중요한 가중치의 값을 키워서 양자화 해상도를 높이는 효과를 냄.
        
    - 수식적 등가성: $Y = (W \cdot s) \cdot (X / s)$
        
        - 가중치 $W$에 스케일 $s (>1)$를 곱해 양자화하고, 입력 $X$는 $s$로 나눔.
            
        - 이렇게 하면 중요한 그룹의 값 범위가 늘어나 **상대적인 양자화 오차가 감소**함.
            
    - **최적의 Scale($s$) 찾기**:
        
        - $s = s_X^{\alpha}$ 공식 사용 ($s_X$는 채널별 활성화의 평균 크기).
            
        - $\alpha$ 값을 Grid Search(0~1 사이)로 찾아 오차를 최소화하는 값을 선택.
            
3. **TinyChat**:
    
    - 이론적인 메모리 절약을 실제 속도 향상으로 바꾸기 위해 개발한 W4A16 추론 프레임워크.
        
    - Kernel Fusion과 On-the-fly Dequantization(실행 직전에 FP16으로 변환)을 사용하여 메모리 대역폭 병목을 해결.
        

## 3. 결과 (Results)

- **성능 지표**:
    
    - LaMA, OPT 모델 등에서 RTN 대비 월등히 높고, GPTQ와 동등하거나 더 나은 PPL달성.
        
    - **일반화 능력(Generalization)**: 보정 데이터에 과적합되지 않아, Instruction-tuned 모델이나 멀티모달 모델(VILA, OpenFlamingo)에서도 성능 저하가 없음.
        
- **속도 향상**:
    
    - **TinyChat** 적용 시, 데스크탑(RTX 4090) 및 모바일 GPU(Jetson Orin)에서 HuggingFace FP16 구현 대비 **3배 이상의 추론 속도 향상**.
        
    - Raspberry Pi 같은 엣지 디바이스에서도 LLaMA-7B 구동 가능.
         

---

- **메모/한줄평**:
    
    - "중요한 놈만 잘 챙기자"는 아이디어를 Mixed-precision 대신 Scaling으로 풀어낸 점이 매우 인상깊음.
        
    - GPTQ가 최적화 문제라면, AWQ는 분포(Distribution) 기반의 휴리스틱 해결책에 가까움.