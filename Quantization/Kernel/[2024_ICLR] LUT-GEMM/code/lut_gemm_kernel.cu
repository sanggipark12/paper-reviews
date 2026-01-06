/*
lut-gemm 간단하게 구현해보기

    input A : [N, K] (fp32 라고 가정)
    weight B : [K, M] (int4가 int32에 담겨있다고 가정)
    output C : [N, M]
    alpha : [bits, M]

*/

#include <cuda_runtime.h>

__global__ void lut_gemm(
    const float* A,
    const int* B,
    float* C,
    const float* alpha,
    int N, int K, int M
) {

    int row = blockIdx.y; // N ( batch 전체)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // M
    int tid = threadIdx.x;


    // mu = 4 라고 가정 
    __shared__ float lut[16];// (이런 식으로 하면 노는 스레드가 발생할 수 있으나 어차피 간략하게 테스트 용이니 나중에 생각 )

    float acc = 0.0f; // ALU

    // 논문에서는 K_TILE_SIZE = 32 이지만 현재 코드에서는 K 전체를 4개 씩 타일링 해서 처리
    for (int k = 0; k < K; k+=4){

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

        if (col < M) {
            // 가중치가 패킹이 안되어 있다고 가정
            int b0 = (k + 0 < K) ? B[(k + 0) * M + col] : 0;
            int b1 = (k + 1 < K) ? B[(k + 1) * M + col] : 0;
            int b2 = (k + 2 < K) ? B[(k + 2) * M + col] : 0;
            int b3 = (k + 3 < K) ? B[(k + 3) * M + col] : 0;

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
    }

    if (col < M && row < N) {
        C[row * M + col] = acc;
    }
}