#include <cuda_runtime.h>
/*
    - brecq를 통해 w4a16으로 양자화된 가중치 연산을 위한 gemmm 커널

        weigt : M x K/8
        input : N x K
        output : M x N

        smem 사용하여 블록당 16x16 타일 계산
        
*/

__global__ void sgemm(
    half* input, int* weight, half* output,
    int N, int K, int M,
    float scale, float zero_point
)
{
    __shared__ int s_w[16][2];
    __shared__ half s_in[16][17]; // bank conflict 방치

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    float acc = 0.0f;

    for (int k = 0; k<K; k+=16) {

        if (row < N && (k + tx) < K) {
            s_in[ty][tx] = input[row * K + (k + tx)];
        }
        else {
            s_in[ty][tx] = __float2half(0.0f);
        }

        if (ty < 2 && col < M && (k/8 + ty) < (K/8)) {
            s_w[tx][ty] = weight[col * (K/8) + (k/8 + ty)];
        }
        else {
            s_w[tx][ty] = 0;
        }

        __syncthreads();

        for (int i=0; i<16; ++i) {
            int packed_val = s_w[tx][i/8];
            int w_int4 = (packed_val >> ((i % 8) * 4)) & 0xF;

            float w_fp = ((float)w_int4 - zero_point) * scale;
            float in_fp = __half2float(s_in[ty][i]);

            acc += w_fp * in_fp;
        }

        __syncthreads();
    }

    if (row < N && col < M) {
        output[row * M + col] = __float2half(acc);
    }
}