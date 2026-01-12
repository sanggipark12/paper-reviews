#include <cuda_runtime.h>

__global__ void baseline_dequant_gemm_profiled(
    const float* A,
    const int* B,
    float* C,
    const float* alpha,
    int N, int K, int M,
    unsigned long long* debug_compute_cycles
) {
    int row = blockIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int tid = threadIdx.x;

    float acc = 0.0f;
    
    // 로컬 카운터
    unsigned long long local_cycles = 0;

    for (int k = 0; k < K; k+=4){
        
        // 측정
        // lut 생성 과정 없음
        long long t0 = clock64();

        if (col < M) {

            // 2. Input 로드
            float a0 = (k + 0 < K) ? A[row * K + (k + 0)] : 0.0f;
            float a1 = (k + 1 < K) ? A[row * K + (k + 1)] : 0.0f;
            float a2 = (k + 2 < K) ? A[row * K + (k + 2)] : 0.0f;
            float a3 = (k + 3 < K) ? A[row * K + (k + 3)] : 0.0f;

            // B는 [K/8, M] 크기이므로, 현재 k가 속한 패킹된 인덱스를 찾음
            int packed_idx = (k / 8) * M + col;
            int packed_val = B[packed_idx];

            // 비트 추출 로직
            // 4개씩 뽑아쓰기
            int shift_amount = (k % 8) * 4;

            int current_amount = packed_val >> shift_amount;

            // 여기서 스케일링 팩터는 원본 스케일 값이어야함
            // 여기서는 Channel-wise Quantization을 가정
            float scale = alpha[col]; 

            // 비트 연산으로 쪼개고 -> 실수로 변환 -> 스케일 곱하기 -> 입력과 곱하기
            
            // Weight 0
            int b0_int = (current_amount >> 0) & 0xF; 
            float b0_fp = (float)b0_int * scale; // Dequant overhead
            acc += b0_fp * a0;

            // Weight 1
            int b1_int = (current_amount >> 4) & 0xF;
            float b1_fp = (float)b1_int * scale;
            acc += b1_fp * a1;

            // Weight 2
            int b2_int = (current_amount >> 8) & 0xF;
            float b2_fp = (float)b2_int * scale;
            acc += b2_fp * a2;

            // Weight 3
            int b3_int = (current_amount >> 12) & 0xF;
            float b3_fp = (float)b3_int * scale;
            acc += b3_fp * a3;
        }
        
        // [타이머 종료]
        long long t1 = clock64();

        if (tid == 0) {
            local_cycles += t1 - t0;
        }
    }

    if (col < M && row < N) {
        C[row * M + col] = acc;
    }

    if (tid == 0) {
        atomicAdd(debug_compute_cycles, local_cycles);
    }
}