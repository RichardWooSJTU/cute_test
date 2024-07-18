#include <iostream>
#include <vector> 
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
// #include <cute/layout.hpp>
// #include "cutlass/util/print_error.hpp"

template<typename TA, typename TB, typename TC,
         int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_device(const TA* A, const TB* B, TC* C, int m, int n, int k) {
    using namespace cute;
    // 1. Make global tensor, TN or NT can be chosen
    // For this case, A is row major, B should be col major
    // TODO: 对比TN NT性能
    Tensor At = make_tensor(make_gmem_ptr(A), make_shape(m, k), make_stride(k, Int<1>{})); // T
    Tensor Bt = make_tensor(make_gmem_ptr(B), make_shape(n, k), make_stride(k, Int<1>{})); // N
    Tensor Ct = make_tensor(make_gmem_ptr(C), make_shape(m, n), make_stride(n, Int<1>{}));


    // 2. Make block tensor (Tile)
    // TODO: 对比xy yx性能
    int ix = blockIdx.x;
    int iy = blockIdx.y;
    Tensor gA = local_tile(At, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _)); // [kTileM, kTilek, num_k_tiles]
    Tensor gB = local_tile(Bt, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gC = local_tile(Ct, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));

    // 3. Make threadmma and thread tensor
    
    // 3.1
    TiledMMA mma;
    auto thr_mma = mma.get_slice(threadIdx.x);

    // 3.2 make global memory partition
    auto tAgA = thr_mma.partition_A(gA); // [MMA, MMA_M, MMA_K, num_k_tiles] MMA_M指在M方向重复的次数 MMA指一个thread在当前MMA配置下的数据 这里MMA（第一维）我终于搞清楚了，第一个就是一个mmaAtom（一个warp）去读取mmaOp的matrice，一个thread要读取的shape
    auto tBgB = thr_mma.partition_B(gB);
    auto tCgC = thr_mma.partition_C(gC);
# if 0
    if (thread0()) {
        print("tAgA");print_tensor(tAgA);print("\n");
        print("tBgB");print_tensor(tBgB);print("\n");
    }
# endif
    // 3.3 make registers memory partition
    auto tArA = thr_mma.partition_fragment_A(gA(_,_,0)); // [MMA, MMA_M, MMA_K]
    auto tBrB = thr_mma.partition_fragment_B(gB(_,_,0));
    auto tCrC = thr_mma.partition_fragment_C(gC(_,_));

# if 0
    if (thread0()) {
        print("tArA");print_tensor(tArA);print("\n");
        print("tBrB");print_tensor(tBrB);print("\n");
    }
#endif

    clear(tCrC);

    // 4. Main loop
    int num_tiles_k = size<2>(gA);

    for (int itile = 0; itile < num_tiles_k; itile++) {
        copy(tAgA(_,_,_,itile), tArA);
        copy(tBgB(_,_,_,itile), tBrB);

        // __sync_thread(); ??
        gemm(mma, tCrC, tArA, tBrB, tCrC);

        // __sync_thread(); ??
    }
    copy(tCrC, tCgC);
}

template<typename TA, typename TB, typename TC>
void LaunchCuteGEMM(const TA* A, const TB* B, TC* C, int m, int n, int k) {

    using namespace cute;
    using mma_op = cute::SM80_16x8x32_S32S8S8S32_TN;
    // using mma_op = cute::SM80_16x8x16_S32S8S8S32_TN; // k 必须是这里K的整数倍
    using mma_traits = cute::MMA_Traits<mma_op>;
    using mma_atom = cute::MMA_Atom<mma_traits>;
    /*
    make_tiled_mma(MMA_Atom<MMA_Op> const& mma_atom,
               MMAThrLayout     const& thr_layout   = {},
               Permutations     const& permutations = {})
    */
    auto mma = cute::make_tiled_mma(mma_atom{},
                                       cute::make_layout(Shape<_1, _2, _1>{}),
                                       cute::make_layout(Shape<_1, _1, _1>{}));
    using MMA = decltype(mma); // N方向 permute， 具体做法不是很清楚，这个不会自动做吗，还是说这个定义一个tiledMMA可以做完(32 32 16)，剩下的需要通过循环来做


    // constexpr int kTileM = 128; //m n k 小于Tile shape的情况都有风险
    constexpr int kTileN = 128;

    constexpr int kTileM = 32;
    // constexpr int kTileN = 16;
    constexpr int kTileK = 32;

    print("size(mma)");print(size(mma));print("\n");

    dim3 block(size(mma));
    dim3 grid(ceil_div(n, kTileN), 
              ceil_div(m, kTileM));

    gemm_device<TA, TB, TC, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(A, 
                                                                          B, 
                                                                          C, 
                                                                          m, n, k);
    auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::cout  << "cute runtime error " << status << "\n";
        exit(0);
    }

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout  << "cute runtime error " << err << "\n";
        exit(0);
    }

}
