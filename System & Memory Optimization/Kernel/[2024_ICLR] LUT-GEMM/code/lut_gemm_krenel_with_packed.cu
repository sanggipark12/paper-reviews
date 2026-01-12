/*
    Input A: [N, K] (FP32)
    Weight B: [K/8, M] (Packed INT4 -> INT32)
    32비트 정수 하나에 8개의 4비트 가중치가 K 차원을 따라 패킹됨
    Output C: [N, M] (FP32)
    Alpha: [4, M] 
 */

#include <cuda_runtime.h>

__global__ void lut_gemm(
    const float* A,
    const int* B,       // [K/8, M] 크기의 패킹된 가중치
    float* C,
    const float* alpha,
    int N, int K, int M,
    unsigned long long* debug_lut_cycles,
    unsigned long long* debug_compute_cycles
) {

    int row = blockIdx.y; // N ( batch 전체)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // M
    int tid = threadIdx.x;


    // mu = 4 라고 가정 
    __shared__ float lut[16];// (이런 식으로 하면 노는 스레드가 발생할 수 있으나 어차피 간략하게 테스트 용이니 나중에 생각 )

    float acc = 0.0f; // ALU

    // 로컬 카운터
    unsigned long long local_lut_time = 0;
    unsigned long long local_compute_time = 0;

    // 논문에서는 K_TILE_SIZE = 32 이지만 현재 코드에서는 K 전체를 4개 씩 타일링 해서 처리
    for (int k = 0; k < K; k+=4){

        // 측정
        long long t0 = clock64();

        if (tid < 16){
            /*
            Optimization:
                데이터를 하나씩 가져오지 말고 int4로 여러개 같고와서 공유메모리에 저장하기(공유 메모리가 남는다면)
                reinterpret_cast 사용 가능
            */
            float a0 = (k + 0 < K) ? A[row * K + (k + 0)] : 0.0f;
            float a1 = (k + 1 < K) ? A[row * K + (k + 1)] : 0.0f;
            float a2 = (k + 2 < K) ? A[row * K + (k + 2)] : 0.0f;
            float a3 = (k + 3 < K) ? A[row * K + (k + 3)] : 0.0f;

            float pattern_sum =     + (2 * ((tid >> 0) & 1) - 1) * a0
                                    + (2 * ((tid >> 1) & 1) - 1) * a1
                                    + (2 * ((tid >> 2) & 1) - 1) * a2
                                    + (2 * ((tid >> 3) & 1) - 1) * a3;

            lut[tid] = pattern_sum;

        }

        __syncthreads();

        // 분기점 
        long long t1 = clock64();

        if (col < M) {

            // B는 [K/8, M] 크기이므로, 현재 k가 속한 패킹된 인덱스를 찾음
            // 예: k=0~7은 index 0, k=8~15는 index 1 ...
            int packed_idx = (k / 8) * M + col;
            int packed_val = B[packed_idx];

            // 비트 추출 로직
            // 4개씩 뽑아쓰기
            int shift_amount = (k % 8) * 4;

            int current_amount = packed_val >> shift_amount;

            // 4개의 가중치(b0~b3) 추출
            int b0 = (current_amount >> 0)  & 0xF;
            int b1 = (current_amount >> 4)  & 0xF;
            int b2 = (current_amount >> 8)  & 0xF;
            int b3 = (current_amount >> 12) & 0xF;

            for (int b=0; b < 4; ++b) {
                // 각 가중치 b번째 비트
                int bit0 = (b0 >> b) & 1;
                int bit1 = (b1 >> b) & 1;
                int bit2 = (b2 >> b) & 1;
                int bit3 = (b3 >> b) & 1;

                int pattern_idx = (bit3 << 3) | (bit2 << 2) | (bit1 << 1) | bit0;

                /*
                Optimization:
                    alpha를 사용 할 때마다 글로벌 메모리에 접근하게 하지 말고
                    루프 이전에 알파를 레지스터로 이동
                */
                acc += lut[pattern_idx] * alpha[b * M + col];
            }
        }
        __syncthreads();

        // 종료
        long long t2 = clock64();

        // tid == 0 일 때, 대표로 측정
        // 워프 내 실행 흐름은 같으니
        if (tid == 0) {
            local_lut_time += t1 - t0;
            local_compute_time += t2 - t1;
        }
    }

    if (col < M && row < N) {
        C[row * M + col] = acc;
    }

    // 결과 합
    if (tid == 0) {
        atomicAdd(debug_lut_cycles, local_lut_time);
        atomicAdd(debug_compute_cycles, local_compute_time);
    }
}