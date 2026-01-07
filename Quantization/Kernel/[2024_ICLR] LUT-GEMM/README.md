# [LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models]

> Links: ArXiv | Official Code
> 
> Tags: #Quantization, #LLM, #Inference_Acceleration, #LUT, #Weight_Only_Quantization
> 
> Author: Gunho Park et al. (POSTECH, NAVER Cloud)
> 
> Date: 2024 (ICLR 2024)

## 1. 기존 연구 및 한계 (Related Work & Limitations)

- **기존의 접근 방식**:
    
    - **W8A8 양자화 (LLM.int8(), SmoothQuant)**: 가중치와 활성화를 모두 8비트로 줄여 INT8 연산기를 활용함. 계산 속도는 빠르지만 메모리 압축률이 W4A16 방식에 비해 낮음.
        
    - **Weight-Only 양자화 (OPTQ, AWQ)**: 메모리를 많이 차지하는 가중치만 3~4비트로 줄이고, 연산 직전에 FP16으로 복원(Dequantization)하여 연산함.
        
- **기존 방식의 한계점**:
    
    - **메모리 병목**: LLM의 추론 단계는 연산량보다 메모리 대역폭이 병목임. W8A8은 메모리 절감 효과가 부족하여 이 병목을 완전히 해소하지 못함.
        
    - **역양자화 오버헤드**: 기존의 Weight-only 방식은 메모리는 줄이지만, 연산할 때마다 INT에서 FP16으로 변환하는 과정이 자원을 많이 잡아먹어 실제 속도 향상이 제한적임.
        

## 2. 기존 연구와의 차이점 (Key Contribution & Methodology)

### 핵심 아이디어 (Core Idea)

- **"we employ a lookup table (LUT) based computation technique to mitigate redundant calculations caused by digitized binary weights after BCQ"**
    
- 가중치를 FP16으로 복원(Dequantization)하지 않고, **미리 계산된 룩업 테이블(LUT)을 조회하는 방식**으로 대체하여 연산 속도를 획기적으로 높임.
    

### 방법론 및 아키텍처 (Methodology & Architecture)

- **LUT 기반 행렬 곱 :
    
    - 가중치가 저비트로 양자화되어 있다면 가능한 패턴의 수가 적음.
        
    - 따라서 활성화와 가중치 패턴의 곱셈 결과를 **미리 계산하여 LUT에 저장**해둠.
        
    - 실제 매트릭스 연산 시에는 **LUT에서 값을 가져오는 단순 작업**으로 대체하여 역양자화 과정을 생략함.
        
- **확장된 BCQ 형식** :
    
    - 기존의 균일(Uniform) 및 비균일(Non-uniform) 양자화 방식을 모두 지원하기 위해, 가중치를 **이진 벡터의 합($\sum \alpha_i b_i$)과 편향($z$)** 형태로 재정의함.
        
    - 이로 인해 AWQ나 OPTQ 같은 다양한 양자화 기법으로 만들어진 모델도 LUT-GEMM 커널 위에서 돌릴 수 있음.
        
- **GPU 최적화** :
    
    - **병렬성 극대화**: 가능한 많은 스레드를 사용하여 병렬로 LUT 조회 수행.
        
    - **공유 메모리 활용**: 스케일링 팩터($\alpha$)를 공유하여 중복 연산을 줄이고, 그룹 단위 양자화를 통해 압축률과 정확도 간의 트레이드오프 조절.
        

## 3. 결과 (Results)

- **성능 지표**:
    
    - **OPT-175B 모델** 기준, 3-bit 양자화 적용 시 기존 커널(OPTQ) 대비 **2.1배** 토큰 생성 속도 향상.
        
    - 단일 A100 GPU에서 거대 모델 추론이 가능해지며, 8개 GPU가 필요하던 작업을 1개로 줄일 수 있어 비용 효율성 증대.
        
- **Ablation Study**:
    
    - **그룹 크기($g$)의 영향**: 그룹 크기를 줄이면 정확도(PPL)는 올라가지만, 스케일링 팩터 저장 공간이 늘어나 메모리 효율이 약간 떨어짐. $g=128$ 정도가 속도와 정확도의 균형점임을 확인.
        
    - **커널 비교**: 다른 커널과 비교했을 때, LUT-GEMM이 가장 낮은 레이턴시를 기록함 (Table 1).
        

---

- **메모/한줄평**:
    
    - Weight-only 양자화의 '복원은 귀찮다.' 라는 문제를 '미리 계산해두기(LUT)'라는 컴퓨터 구조의 고전적인 아이디어로 멋지게 해결한 논문.
        
    - 단순히 알고리즘을 제안한 것이 아닌 다른 양자화 모델을 수행할 수 있다는 범용성을 확장한 점이 돋보임.