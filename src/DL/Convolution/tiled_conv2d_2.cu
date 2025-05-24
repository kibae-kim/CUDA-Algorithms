#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

// 2D Shared Memory Tiled Convolution 커널
__global__ void sharedmem_conv2d_kernel(
    const float* __restrict__ input,   // [N, C, H, W]
    const float* __restrict__ weight,  // [Kout, C, Kh, Kw]
    const float* __restrict__ bias,    // [Kout]
    float* __restrict__ output,        // [N, Kout, Ho, Wo]
    int N, int C, int H, int W,
    int Kout, int Kh, int Kw,
    int pad, int stride)
{

    // 타일로 나뉘어질 출력 피쳐맵 전체 크기를 먼저 구함
    // 출력 피쳐맵이 그것의 행과 열방향으로 몇개의 픽셀을 가지고 있는지 계산
    // 블록안 하나의 스레드가 출력 피쳐맵에속한 하나의 픽셀을 담당하게됨
    // 패딩은 상하좌우 시작 끝에 동일한 크기만큼 적용하게됨
    const int Ho = ((pad + H + pad) - Kh) / stride + 1;
    const int Wo = ((pad + W + pad) - Kw) / stride + 1;
    
    // 하나의 스레드블록이 하나의 타일안의 모든 출력픽셀을 산출을 담당하는것으로 가정
    // 하나의 스레드블록안의 스레드들과, 담당타일안의 픽셀들이, 일대일대응 하는것으로 가정
    // 그 스레드블록이 수행하는 행방향으로의 컨볼루션 횟수는 행방향으로 그것이 가진 스레드의 개수
    // 그 스레드블록이 수행하는 열방향으로의 컨볼루션 횟수는 열방향으로 그것이 가진 스레드의 개수
    const int TILE_Height_convTime = blockDim.y;
    const int TILE_Width_convTime = blockDim.x; 

    // 마스크(커널)가 순회하는 리셉티브필드(버퍼)의 픽셀(탭)들을 전역메모리로부터 
    // 공유메모리로 가져오기 위해서 필요한 만큼의 공유메모리 영역을 계산하고 동적으로 선언
    // 타일안의 출력픽셀과 인접출력픽셀을 계산하는, 스레드와 인접스레드가, 가져와야하는 리셉티브필드는
    // 스트라이드가 마스크의 폭(혹은 높이)보다 작은이상 겹쳐질 수 밖에 없고, 겹쳐지는 부분의 크기가
    // 마스크의 폭(혹은 높이)에서 스트라이드를 뺀 만큼이고, 겹쳐지지않는 부분이 스트라이드만큼임
    // 블록안에서 제일 좌상단에 위치한 스레드가 컨볼루션을 수행하기위해서 전역메모리의 픽셀들을
    // 마스크의 폭만큼 공유메모리로 로딩, 블록안의 잔여 스레드들도 컨볼루션을 수행하기 위해서
    // 스트라이드 크기만큼 픽셀들을 (블록안의 스레드개수 - 1) 횟수로 공유메모리로 로딩
    const int IN_TILE_Height_pxls = Kh + stride * (TILE_Height_convTime - 1); 
    const int IN_TILE_Width_pxls = Kw + stride * (TILE_Width_convTime - 1); 
    extern __shared__ float input_tile[];
    
    // 각 타일당 공유메모리로 불러들어야할 총 입력픽셀 수를 계산
    // 블록 내부의 모든 스레드가 입력픽셀들의 로딩에 참여(각자의 컨볼루션을 수행하기 이전에)
    // 블록 내부의 평탄화된 인덱스들로 블록안의 스레드들 각각이 한번에 하나씩 픽셀들을 로딩함
    const int IN_TILE_total_pxls = IN_TILE_Height_pxls * IN_TILE_Width_pxls;
    const int loaderNumsPerBlock = TILE_Height_convTime * TILE_Width_convTime;
    const int loaders = threadIdx.y * TILE_Width_convTime + threadIdx.x;

    // 출력타일들의 좌상단 병렬 좌표
    const int tiles_output_top = blockIdx.y * TILE_Height_convTime;
    const int tiles_output_left = blockIdx.x * TILE_Width_convTime; 
    
    // 배치 인덱스
    const int n = blockIdx.z;

    // 모든 출력 채널 루프에서 출력 타일들의 좌상단 병렬 좌표에서
    // 병렬 오프셋을 더해서 각 스레드가 담당하는 출력 픽셀의 위치를 잡음
    for (int kout = 0; kout < Kout; ++kout) {
        const int out_y = tiles_output_top + threadIdx.y;
        const int out_x = tiles_output_left + threadIdx.x;

        float output_sum = 0.0f;
        
        // 입력 채널
        for (int cin = 0; cin < C; ++cin) {
            for (int tidx1d = loaders; tidx1d < IN_TILE_total_pxls; tidx1d += loaderNumsPerBlock) {
                
                // 전역메모리를 참조하기위해 평탄화된 인덱스로부터 
                // 공유메모리를 참조하기위한 2차원 행,열 인덱스를 복원
                const int tile_is = tidx1d / IN_TILE_Width_pxls;
                const int tile_js = tidx1d % IN_TILE_Width_pxls;
                
                // 리셉티브필드(버퍼)안의 블록안의스레드수만큼 픽셀(탭)들을 한번에 로딩
                const int tabIdxs_y = tiles_output_top * stride + tile_is - pad;
                const int tabIdxs_x = tiles_output_left * stride + tile_js - pad;
                float val = 0.0f;
                
                // 전역메모리를 참조
                if (tabIdxs_y >= 0 && tabIdx_y < H && tabIdxs_x >= 0 && tabIdxs_x < W)
                    val = input[n*C*H*W + cin*H*W + tabIdxs_y*W + tabIdxs_x];
                
                // 공유메모리로 로딩
                input_tile[tile_is * IN_TILE_Width_pxls + tile_js] = val;
                
            }
            __syncthreads(); // 공유메모리 타일 완성 대기

            // (5-2) 컨볼루션 연산 (출력 픽셀마다)
            if (out_x < Wo && out_y < Ho) {
                float partial_sum = 0.0f;
                for (int ki = 0; ki < Kh; ++ki) {
                    for (int kj = 0; kj < Kw; ++kj) {
                        // 공유메모리에서 해당 리셉티브필드 위치 값을 읽음
                        const int tile_i = threadIdx.y * stride + ki;
                        const int tile_j = threadIdx.x * stride + kj;
                        float in_val = input_tile[tile_i * IN_TILE_Width_pxls + tile_j];
                        float w = weight[kout*C*Kh*Kw + cin*Kh*Kw + ki*Kw + kj];
                        partial_sum += in_val * w;
                    }
                }
                output_sum += partial_sum;
            }
            __syncthreads(); // 다음 입력채널 타일 로딩 전 동기화
        }

        // (6) 편향 더하고 결과를 전역 출력에 기록
        if (out_x < Wo && out_y < Ho) {
            output_sum += bias[kout];
            output[n*Kout*Ho*Wo + kout*Ho*Wo + out_y*Wo + out_x] = output_sum;
        }
        __syncthreads(); // 다음 출력채널 전환 전 동기화
    }
}
