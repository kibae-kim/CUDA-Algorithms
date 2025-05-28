%%writefile tiled_conv2d.cu

#include <stdio.h>  
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
    extern __shared__ float shared_mem[];
    
    // 각 타일당 공유메모리로 불러들어야할 총 입력픽셀 수를 계산
    // 블록 내부의 모든 스레드가 입력픽셀들의 로딩에 참여(각자의 컨볼루션을 수행하기 이전에)
    // 블록 내부의 평탄화된 인덱스들로 블록안의 스레드들 각각이 한번에 하나씩 픽셀들을 로딩함
    // 한번 로딩할 때, 여러개의 타일안의 스레드들 전부 및 특정 타일의 일부 스레드들을 로딩함
    const int IN_TILE_total_pxls = IN_TILE_Height_pxls * IN_TILE_Width_pxls;
    const int loaderNumsPerBlock = TILE_Height_convTime * TILE_Width_convTime;
    const int loaders = threadIdx.y * TILE_Width_convTime + threadIdx.x; //ex)(2,2)->2*4+2=10 index

    // 출력타일들의 좌상단 병렬 좌표
    // 스레드블록의 차원과 블록안에서의 스레드인덱스를 곱하면 
    // 병렬로 나열된 스레드블록들안의 첫번째 원소의 전역그리드 차원
    // 버퍼(리셉티브영역)안의 선두탭(픽셀)을 구하기 위해서 필요
    const int tiles_output_top = blockIdx.y * TILE_Height_convTime;
    const int tiles_output_left = blockIdx.x * TILE_Width_convTime; 
    
    const int out_y = tiles_output_top + threadIdx.y;
    const int out_x = tiles_output_left + threadIdx.x;
    
    // 배치 인덱스
    const int n = blockIdx.z;

    // 모든 출력 채널 루프에대해 반복 
    for (int kout = 0; kout < Kout; ++kout) {
        float output_sum = 0.0f; //컨볼루션 결과 초기화
        
        // 모든 입력 채널 루프에대해 반복
        for (int cin = 0; cin < C; ++cin) {
        
            // <<1.임의의 블록에 대해서, 각 블록안의 모든스레드가 공유메모리에 필요한 픽셀(탭)들을 로드할 때까지 반복>>
            for (int tidx1d = loaders; tidx1d < IN_TILE_total_pxls; tidx1d += loaderNumsPerBlock) {
                
                // 좌표변환: (블록내부스레드인덱스)->(평탄화인덱스)->(shared_mem내부픽셀인덱스)
                const int tile_is = tidx1d / IN_TILE_Width_pxls;
                const int tile_js = tidx1d % IN_TILE_Width_pxls;
                
                // 스트라이드와 패딩이 주어졌을 때 마스크h[m]과 맞물리는 전역메모리상의 리셉티브 영역부분의 인덱스
                // x[n+m] -> x[n*stride+m-pad], tabIdxs = n*stride+m-pad
                const int tabIdxs_y = tiles_output_top * stride + tile_is - pad;
                const int tabIdxs_x = tiles_output_left * stride + tile_js - pad;
                float receptiveFld = 0.0f;
                
                // 입력피쳐맵의 경계조건은 매번 스레드블록이 탭(픽셀)들을 로딩할 때 검사되어야함
                if (tabIdxs_y >= 0 && tabIdxs_y < H && tabIdxs_x >= 0 && tabIdxs_x < W)
                    receptiveFld = input[n*C*H*W + cin*H*W + tabIdxs_y*W + tabIdxs_x]; // [n,cin,tabIdxs_y,tabIdx_x]
                
                // 마스크와 맞물리는 공유메모리영역으로 로딩(tile_is, tile_js를 이용해서 공유메모리 참조)
                shared_mem[tile_is * IN_TILE_Width_pxls + tile_js] = receptiveFld; //[tile_is,tile_js]
                
            } // <<1. 블록안의 모든스레드가 공유메모리에 필요한 픽셀들을 로드 후 종료>>
            __syncthreads(); // 공유메모리 타일 완성 대기


            // <<2. 출력피쳐맵의 경계조건>>
            // 출력피쳐맵의 전역좌표가 커널초반에 계산된 조건을 벗어나는지 여부를 단 한번만 검사
            if (out_x < Wo && out_y < Ho) {
                float partial_sum = 0.0f;
                
                // 마스크의 모든 픽셀에 대해서 순회(방향오프셋으로 순회)
                for (int ki = 0; ki < Kh; ++ki) {
                    for (int kj = 0; kj < Kw; ++kj) {
                        
                        // 공유메모리shared_mem에서 해당 리셉티브필드 위치 값을 읽음
                        // 출력피쳐맵의 타일안에서 인접픽셀들을 계산하는 threadIdx 끼리 
                        // 전역메모리의 입력단에서 리셉티브필드가 스트라이드 차이가 나는것처럼
                        // 그들의 공유메모리에서 리셉티브필드차이도 여전히 스트라이드만큼 차이가 남
                        // (기준점오프셋)+(방향오프셋)
                        const int tile_i = threadIdx.y * stride + ki;
                        const int tile_j = threadIdx.x * stride + kj;
                        
                        float receptiveFld_eles = shared_mem[tile_i * IN_TILE_Width_pxls + tile_j];
                        float mask_eles = weight[kout*C*Kh*Kw + cin*Kh*Kw + ki*Kw + kj];
                        
                        partial_sum += receptiveFld_eles * mask_eles;
                    }
                } // 마스크의 모든 픽셀에 대해서 순회종료 후 특정입력채널cin에서 컨볼루션결과누적 
                output_sum += partial_sum; 
                
            } // <<2. 출력피쳐맵의 경계조건 종료>>
            __syncthreads(); // 다음 입력채널 타일 로딩 전 동기화
            
        } // 입력채널순회종료

        // 특정출력채널kout에서 편향 더하고 결과를 전역 출력에 기록
        if (out_x < Wo && out_y < Ho) {
            output_sum += bias[kout];
            output[n*Kout*Ho*Wo + kout*Ho*Wo + out_y*Wo + out_x] = output_sum;
        } 
        __syncthreads(); // 다음 출력채널 전환 전 동기화
        
    } // 출력채널순회종료
    
} //커널종료

int main() {
    /* -------------------- 1. 파라미터 -------------------- */
    constexpr int N = 1, C = 1, H = 3, W = 3;
    constexpr int Kout = 1, Kh = 2, Kw = 2;
    constexpr int pad = 0, stride = 1;
    constexpr int Ho  = (H - Kh) / stride + 1;
    constexpr int Wo  = (W - Kw) / stride + 1;

    /* -------------------- 2. 호스트 버퍼 ------------------- */
    float h_in [H * W]   = { 1,2,3,4,5,6,7,8,9 };
    float h_w  [Kh * Kw] = { 1,0,0,1 };
    float h_b  [Kout]    = { 0 };
    float h_out[Ho * Wo] = { 0 };

    /* -------------------- 3. 디바이스 메모리 ---------------- */
    float *d_in, *d_w, *d_b, *d_out;
    cudaMalloc(&d_in , sizeof(h_in ));
    cudaMalloc(&d_w  , sizeof(h_w  ));
    cudaMalloc(&d_b  , sizeof(h_b  ));
    cudaMalloc(&d_out, sizeof(h_out));

    cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w , h_w , sizeof(h_w ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b , h_b , sizeof(h_b ), cudaMemcpyHostToDevice);

    /* -------------------- 4. 커널 실행 --------------------- */
    dim3 block(Wo, Ho);      // 2×2 = 4 threads
    dim3 grid (1, 1, N);     // 배치 N개 → blockIdx.z

    size_t smem = (Kh + stride*(block.y-1)) *
                  (Kw + stride*(block.x-1)) * sizeof(float);

    sharedmem_conv2d_kernel<<<grid, block, smem>>>(
        d_in, d_w, d_b, d_out,
        N, C, H, W, Kout, Kh, Kw, pad, stride);
    cudaDeviceSynchronize();

    /* -------------------- 5. 결과 확인 -------------------- */
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);

    printf("Output (%d×%d):\n", Ho, Wo);
    for (int i = 0; i < Ho; ++i) {
        for (int j = 0; j < Wo; ++j)
            printf("%6.1f ", h_out[i*Wo + j]);
        printf("\n");
    }

    /* -------------------- 6. 자원 반환 -------------------- */
    cudaFree(d_in); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
    return 0;
}
