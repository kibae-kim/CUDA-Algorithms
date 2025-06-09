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
    // 타일안의 출력픽셀과 인접출력픽셀을 산출하는, 스레드와 인접스레드가, 가져와야하는 리셉티브필드는
    // 스트라이드가 마스크의 폭(혹은 높이)보다 작은이상 겹쳐질 수 밖에 없고, 겹쳐지는 부분의 크기가
    // 마스크의 폭(혹은 높이)에서 스트라이드를 뺀 만큼이고, 겹쳐지지않는 부분이 스트라이드만큼임
    // 겹쳐지는 부분은 단 한번만 공유메모리로 가져와야하고, 순회되지않는 경계부분의 픽셀들은 모두 누락시킴
    // 블록안에서 제일 좌상단에 위치한 스레드가 컨볼루션을 수행하기위해서 전역메모리의 픽셀들을
    // 마스크의 폭만큼 공유메모리로 로딩, 블록안의 잔여 스레드들도 컨볼루션을 수행하기 위해서
    // 스트라이드 크기만큼 픽셀들을 (블록안의 스레드개수 - 1) 횟수로 공유메모리로 로딩
    const int numSharedMEM_Height_pxls = Kh + stride * (TILE_Height_convTime - 1); 
    const int numSharedMEM_Width_pxls = Kw + stride * (TILE_Width_convTime - 1); 
    extern __shared__ float shared_mem[];
    
    // 블록 내부의 모든 스레드가 입력픽셀들의 로딩에 참여(각자의 컨볼루션을 수행하기 이전에)
    // 블록 내부의 평탄화된 인덱스들로 블록안의 스레드들 각각이 한번에 하나씩 픽셀들을 로딩함
    const int numSharedMEM_total_pxls = numSharedMEM_Height_pxls * numSharedMEM_Width_pxls;
    const int numLoadersPerBlock = TILE_Height_convTime * TILE_Width_convTime;
    
    // 임베딩: ((by),bx,bz,(ty,tx),tz)[dim6] -> (bx,bz,(xyPlaneIdx_flatten),tz)[dim4]
    const int xyPlaneIdx_flatten = threadIdx.y * blockDim.x + threadIdx.x; //ex)(2,2)->2*4+2=10 index
    
    // 쿠다커널 스레드의 일반적인 좌표형태 (by,bx,bz,ty,tx,tz)의 6차원 s.t (by,bx,bz),(ty,tx,tz)는 서로독립
    // 임베딩: (행방향블록들, 블록안에서행방향스레드들) -> 행방향의전역스레드들 ((by,ty),bx,tx,bz,tz)[dim6]->((gy),bx,bz,tx,tz)[dim5]
    // 임베딩: (열방향블록들, 블록안에서열방향스레드들) -> 열방향의전역스레드들 (by,ty,(bx,tx),bz,tz)[dim6]->(by,ty,(gx),bz,tz)[dim5]
    // such that gy = gy(f_b(by),cardinality(ty),f_t(ty)) gx = gx(f_b(bx),cardinality(tx),f_t(tx))
    // f_b와 f_t가 리니어이면, gy = gy(stride*by+offset,card(ty),stride*ty+offset)로 블록/스레드들 간격일정
    
    // 공유메모리 버퍼(리셉티브영역)안에 제일먼저 유입되는 탭(픽셀)의 전역좌표 gy = gy(ty=0), gx = gx(tx=0)
    const int tiles_output_top_gIdx = blockIdx.y * blockDim.y; // case s.t threadIdx.y=0
    const int tiles_output_left_gIdx = blockIdx.x * blockDim.x; // case s.t threadIdx.x=0
    
    // 배치 인덱스
    const int n = blockIdx.z;

    // 모든 출력 채널 루프에대해 반복 
    for (int kout = 0; kout < Kout; ++kout) {
    
        // 컨볼루션 결과 초기화, 입력채널별 컨볼루션결과를 합친 후 받아야함
        float output_sum = 0.0f; 
        
        // 모든 입력 채널 루프에대해 반복
        for (int cin = 0; cin < C; ++cin) {
        
            // <<1.임의의 블록에 대해서, 각 블록안의 모든스레드가 공유메모리에 필요한 픽셀(탭)들을 로드할 때까지 반복>>
            // 스레드블록안의 스레드들이 평탄화된 인덱스의 형태로 커널이 훑고 지나가는 리셉티브영역들의 모든 픽셀들을 로드해야함
            // loadIdxs+(offset) 스레드들이 다음에 로드할 전역메모리 위치로, 이전에 가져왔던 위치와 겹치지 않게해야하는 오프셋값을 설정
            for (int loadIngIdx = xyPlaneIdx_flatten; loadIngIdx < numSharedMEM_total_pxls; loadIngIdx += numLoadersPerBlock) {
                
                // 좌표변환: (블록내부스레드2차원인덱스)->(평탄화된 1차원인덱스)->(공유메모리내부2차원픽셀인덱스)
                // 동일하게 평면이지만,행과 열의 차원값이 다른 공간의 인덱스를 재좌표하는 과정
                const int recoordinated_loaded_idx = loadIngIdx / numSharedMEM_Width_pxls;
                const int recoordinated_loaded_jdx = loadIngIdx % numSharedMEM_Width_pxls;
                
                // 스트라이드와 패딩이 주어졌을 때 마스크(충격응답들)h[m]과 맞물리는 전역메모리상의 리셉티브 영역부분의 전역인덱스는
                // x[n+m] -> x[n*stride+m-pad], tabIdxs = n*stride+m-pad 로 버퍼안으로 m번째에 유입되는 신호의 인덱스
                const int loadIng_tabIdxs_y = tiles_output_top_gIdx * stride + recoordinated_loaded_idx - pad;
                const int loadIng_tabJdxs_x = tiles_output_left_gIdx * stride + recoordinated_loaded_jdx - pad;
                float receptiveFld = 0.0f;
                
                // 전역메모리에서 공유메모리로 로딩할 때 "정확하게 필요한 수 만큼"의 스레드들만 참여해서 계산리소스 절약
                // 전역메모리에서 공유메모리로 로딩 때는 매 오프셋(스레드이동을 위한)마다 경계조건을 검사
                // 매번 스레드블록이 탭(픽셀)들을 로딩할 때마다 전역메모리상의 인덱스들에 대해서 경계조건이 검사되어야함
                if (loadIng_tabIdxs_y >= 0 && loadIng_tabIdxs_y < H && loadIng_tabJdxs_x >= 0 && loadIng_tabJdxs_x < W)
                    receptiveFld = input[n*C*H*W + cin*H*W + loadIng_tabIdxs_y*W + loadIng_tabJdxs_x]; 
                
                // 마스크와 맞물리는 공유메모리영역으로 로딩(recoordinated_loaded_idx/jdx를 이용해서 참조)
                // 공유메모리안의 recoordinated_loaded_idx/jdx는 전역메모리의 버퍼안의탭들의인덱스 tabIdxs_x/y
                // 출력단에서 인접픽셀들이 전역메모리단에서처럼 공유메모리단에서도 스트라이드차이가 남
                // 단지 스트라이드가 마스크의 폭보다 작으면 인풋피쳐맵에서 경계에 있는 패딩의 픽셀들이 누락되는것이고
                // 스트라이드가 마스크의 폭보다 크면 인풋피쳐맵에서 이웃 리셉티브필드 사이에 있는 픽셀들이 누락되는것임
                shared_mem[recoordinated_loaded_idx * numSharedMEM_Width_pxls + recoordinated_loaded_jdx] = receptiveFld; 
                
            } // <<1. 블록안의 모든스레드가 공유메모리에 필요한 픽셀들을 로드 후 종료>>
            __syncthreads(); // 공유메모리 타일 완성 대기


            // <<2. 출력피쳐맵의 경계조건>>
            // "정확하게 필요한 수 만큼"의  스레드들만 참여해서 계산리소스를 절약해야하기 때문에 병렬계산 전에 "인덱스들"의 경계조건을 따지는것이 중요함
            // 각 입력채널당 마스크를 출력채널의 횟수만큼 적용(컨볼루션)하고 합치는 형태이기 때문에 
            // 각 출력채널에 대한 컨볼루션 결과를 받을 부분합을 먼저 선언
            if (tiles_output_left_gIdx + threadIdx.x < Wo && tiles_output_top_gIdx + threadIdx.y < Ho) {
                float partial_sum = 0.0f;
                
                // 하나의 스레드가 하나의 출력픽셀을 계산하기로 되어있으므로, 
                // 각 스레드는 마스크의 모든 픽셀에 대해서 순회(방향오프셋으로 순회)
                for (int ki = 0; ki < Kh; ++ki) {
                    for (int kj = 0; kj < Kw; ++kj) {
                        
                        // 출력피쳐맵의 타일안에서 인접픽셀들을 계산하는 threadIdx 끼리 
                        // 전역메모리에서 가져와야할 리셉티브필드간 거리차가 스트라이드 차이가 나는것처럼
                        // 그들의 공유메모리에서 리셉티브필드간 거리차이도 여전히 스트라이드만큼 차이가 남
                        // 스트라이드만큼 차이나는 공유메모리안 시작점에서 스레드들이 어떻게 이동할지 정의
                        const int tidxMov_y_inTile = threadIdx.y * stride + ki;
                        const int tidxMov_x_inTile = threadIdx.x * stride + kj;
                        
                        // 공유메모리shared_mem에서 해당 리셉티브필드 위치 값을 읽음
                        // 
                        float receptiveFld_eles = shared_mem[tidxMov_y_inTile * numSharedMEM_Width_pxls + tidxMov_x_inTile];
                        // 병렬스레드들에 대해서 고정된 단일 스레드가 참조
                        float mask_eles = weight[kout*C*Kh*Kw + cin*Kh*Kw + ki*Kw + kj];
                        
                        partial_sum += receptiveFld_eles * mask_eles;
                    }
                } // 마스크의 모든 픽셀에 대해서 순회종료 후 특정입력채널cin에서 컨볼루션결과누적 
                output_sum += partial_sum; 
                
            } // <<2. 출력피쳐맵의 경계조건 종료>>
            __syncthreads(); // 다음 입력채널 타일 로딩 전 동기화
            
        } // 입력채널순회종료

        // 출력피쳐맵의 결과를 적어놓는곳은 전역메모리이므로 전역스레드에 대해서 경계조건 따지고, 전역스레드로 출력피쳐맵을 참조해서 결과를 기록
        if (tiles_output_left_gIdx + threadIdx.x < Wo && tiles_output_top_gIdx + threadIdx.y < Ho) {
            output_sum += bias[kout]; // 출력채널별로 임계값(=-편향)을 따짐
            output[n*Kout*Ho*Wo + kout*Ho*Wo + (tiles_output_top_gIdx + threadIdx.y)*Wo + (tiles_output_left_gIdx + threadIdx.y)] = output_sum;
        } 
        __syncthreads(); // 다음 출력채널 전환 전 동기화
        
    } // 출력채널순회종료
    
} //커널종료