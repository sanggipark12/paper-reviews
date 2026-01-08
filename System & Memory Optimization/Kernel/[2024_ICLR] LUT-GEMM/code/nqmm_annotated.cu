#include <cuda_runtime.h>

namespace kernel {


    template<int NUM_BITS, int M_TILE_SIZE, int N_TILE_SIZE, int K_TILE_SIZE>
    __global__ void nqmm_t(
    
        uint32_t* W,   // [Grid_K(blockIdx.y), Inner_K(32/8), Bits(4), Output(M)] -> 패킹된 4D 텐서 형태
        float * alpha, // [Bits(4), Output(M)] -> 비트와 출력 채널에 따라 스케일이 다름
        float* input,  // [Batch(N), Input(K)]
        float * output,// [Batch(N), Output(M)]
        int M,
        int N,
        int K) {

            // mu = 8
            __shared__ float lut[K_TILE_SIZE/8][256][N_TILE_SIZE];

            const int lut_y_size = K_TILE_SIZE/8; // group number
            const int lut_x_size = blockDim.y / (lut_y_size); // per group thread

            // 질문 : threadIdx.y가 K 타일과 관련이 있는데 그럼 y가 k축 방향으로 움직이는 건가요 - 현재 y는 K 차원을 쪼개는 용도로 사용됨
            int lut_y = threadIdx.y / lut_x_size; // 그럼 lut_y는 처리해야할 k 축에서 몇번 째 그룹인지를 나타내나요 - k축이라기 보단 묶인 그룹의 인덱스를 나타냄
            int lut_x = threadIdx.y % lut_x_size; // lut_x는 그 그룹에서 어느 패턴인지를 나타내는거고요 - 패턴의 시작점을 나타냄
            int lut_z = threadIdx.x; // N dim

            // blockIdx.y를 K 차원을 나누는데 사용
            float * _inp = &input[lut_z * K + (blockIdx.y * K_TILE_SIZE + lut_z * 8)]; // 질문 : 8을 곱한 이유가 mu가 8이여서 인가요 - mu가 8이니깐 메모리 주소를 8칸 점프해야함
            
            /*
            질문 : lut_x가 그자체로 패턴 인덱스에 사용된다? - 32개의 패턴이 사용되고 아래 동적으로 다시 처리.
            */
            float base =    + (2 * ((lut_x>>0) & 1) - 1) * _inp[0]
                            + (2 * ((lut_x>>1) & 1) - 1) * _inp[1]
                            + (2 * ((lut_x>>2) & 1) - 1) * _inp[2]
                            + (2 * ((lut_x>>3) & 1) - 1) * _inp[3]
                            + (2 * ((lut_x>>4) & 1) - 1) * _inp[4]
                            + (2 * ((lut_x>>5) & 1) - 1) * _inp[5]
                            + (2 * ((lut_x>>6) & 1) - 1) * _inp[6]
                            + (2 * ((lut_x>>7) & 1) - 1) * _inp[7] ;
             
            lut[lut_y][lut_x][lut_z] = base;

            int s = (lut_x_size==1)  ?0:
                    (lut_x_size==2)  ?1:
                    (lut_x_size==4)  ?2:
                    (lut_x_size==8)  ?3:
                    (lut_x_size==16) ?4:
                    (lut_x_size==32) ?5:
                    (lut_x_size==64) ?6: 
                    (lut_x_size==128)?7: 8;

            for (;s <8; ++s){
                // 질문 : 패턴 p와 p + 2^b가 b비트 번째만 차이가 나는가? - 2, 3 과 0, 1 을 비교하면 2번 째 비트만 차이가 남
                float iValue = 2* _inp[s];

                for (int i = (1 << s); i < (1 << (s+1)); i += lut_x_size) {
                    lut[lut_y][i + lut_x][lut_z] =  lut[lut_y][i +  lut_x - (1 << s)][lut_z] + iValue; 
                }
            }

            __syncthreads();

            // blockIdx.x 는 m 차원에 사용
            int m_start = blockIdx.x * M_TILE_SIZE + threadIdx.y; // 질문 : m은 출력 차원에서 열일텐데 thread y는 k 방향으로 움직이지 않나요? 왜 m축에서 재사용되나요? - 동기화가 끝나고 y가 노니깐 불러와서 재사용
            int m_end = (blockIdx.x + 1) * M_TILE_SIZE; // 질문 : +1 을 한게 왜 end인가요? 한 타일만 처리하겠다는 건가요? - 전체 M 중에서 자기 구역만 처리하겠다.
            
            m_end = (m_end < M) ? m_end : M;
            int m_step = blockDim.y; // 128 -> 128 줄을 병렬 처리

            
            uint32_t *bW = &W[blockIdx.y * K_TILE_SIZE/32 * NUM_BITS * M]; // 질문 : 패킹된 가중치가 1차원으로 쭉 펴진건가요? 그럼 bw는 blockidx y로 나눠진 줄의 가중치의 주소인가요? - 넵
            float *_output = &output[lut_z * M]; // 질문 : 이것도 마찬가지로 threadidx x 로 나눠진 n 차원 주소인가요 - 넵 배치 인덱스입니다
            for(int m = m_start;m < m_end;m += m_step){
                float reg_o = 0;
                for(int b=0;b < NUM_BITS;b++){
                    float   reg_a = alpha[b * M + m];
                    float   reg_t_o = 0;
                    for(int kt=0;kt < K_TILE_SIZE/32;kt++){
                        uint32_t reg_w = bW[kt * NUM_BITS * M + b * M + m];
                        // bw는 int 가중치 
                        // uint32는 4개의 그룹 
                        // 패킹 해제 로직
                        int reg_w0 = (reg_w >> 8 * 0) & 255;   reg_t_o +=  + lut[kt*4 + 0][reg_w0][lut_z];
                        int reg_w1 = (reg_w >> 8 * 1) & 255;   reg_t_o +=  + lut[kt*4 + 1][reg_w1][lut_z];
                        int reg_w2 = (reg_w >> 8 * 2) & 255;   reg_t_o +=  + lut[kt*4 + 2][reg_w2][lut_z];
                        int reg_w3 = (reg_w >> 8 * 3) & 255;   reg_t_o +=  + lut[kt*4 + 3][reg_w3][lut_z]; 
                    }
                    reg_o += reg_a * reg_t_o;
                }
                atomicAdd(&_output[m], reg_o);
    }           
    /* 질문 : 정리하면 K_TILE_SIZE는 32로 32개의 k 를 처리한다. mu=8 이므로 32/8 = 4 개의 그룹으로 처리한다 
    4개의 그룹당 가질 스레드는 lut_x_size로 표현한다. input을 blockidx y(큰 K 차원)을 사용하여 등분하면서 처리한다. 
     lut를 생성하고, 출력의 출력채널 m 을 m_step=128 개 씩 처리한다.(일할 스레드가 128기니깐) 큰 K를 kt로 작은 타일로 나눠서 계산한다.
    */
    }
}