

#include <string>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <cublas_v2.h>

template<typename TA, typename TB, typename TC>
void LaunchCublasLtGEMM(const TA* A_dev, const TB* B_dev, TC* C_dev, 
                        int m, int n, int k,
                        bool is_read_from_csv=false,
                        std::string name = "search") {

    char* workspace;

    //init data structure

    cublasLtMatmulDesc_t matmul_desc_;
    cublasLtMatrixLayout_t A_desc_;
    cublasLtMatrixLayout_t B_desc_;
    cublasLtMatrixLayout_t C_desc_;
    int32_t alpha_ = 1;
    int32_t beta_ = 0;


    cublasComputeType_t cudaComputeType = CUBLAS_COMPUTE_32I;
    cublasLtMatmulDescCreate(
        &matmul_desc_, cudaComputeType, CUDA_R_32I);
    cublasOperation_t op_transpose = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(matmul_desc_,
                                                 CUBLASLT_MATMUL_DESC_TRANSA,
                                                 &op_transpose,
                                                 sizeof(op_transpose));
    cublasLtMatrixLayoutCreate(&B_desc_, CUDA_R_8I, k, n, k);
    cublasLtMatrixLayoutCreate(&A_desc_, CUDA_R_8I, k, m, k);
    cublasLtMatrixLayoutCreate(&C_desc_, CUDA_R_32I, n, m, n);

    cublasLtMatmulAlgo_t algo;
    int algoId;
    int swizzle;
    int customOption;
    int tile ;
    int splitK_val;
    int reductionScheme;
    int stages;
    size_t work_space_size = 0;
    float time_ref;

    auto using_default_config = [&](){
        algoId = 21;
        swizzle = 0;
        customOption = 0;
        tile = 15;
        splitK_val = 0;
        reductionScheme = 0;
        stages = 23;
        if (m >= 128) {
            tile = 20;
            stages = 17;
        }
    };
    if (is_read_from_csv) {
        int m_tmp, k_tmp, n_tmp;
        FILE *fp;
        std::string full_name = name + ".csv";
        fp=fopen(full_name.c_str(), "r");
        if (!fp) {
            using_default_config();
        } else {
            bool match = false;
            int find_cnt = 0;
            while(1) {
                fscanf(fp,"%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f",
                    &m_tmp,&k_tmp, &n_tmp, &algoId, &swizzle, &customOption,  &tile, &splitK_val, 
                    &reductionScheme,&stages, &work_space_size, &time_ref);
                if (feof(fp))break;
                if (k_tmp == k && n_tmp == n && m <= m_tmp) {
                    match = true;
                    break;
                }
                find_cnt++;
            }
            if (find_cnt==0) {
                std::cout << "Please use test mode to select\n, Now we use default params\n" ;
                using_default_config();
            }
        } 
    } else {
        std::cout << "Please use test mode to select\n, Now we use default params\n" ;
        using_default_config();
    }

    cublasLtHandle_t handle_;
    cublasLtCreate(&handle_);

    cudaMalloc((void**)&workspace, work_space_size);
    cublasLtMatmulAlgoInit(handle_,
                                cudaComputeType,
                                CUDA_R_32I,
                                CUDA_R_8I,
                                CUDA_R_8I,
                                CUDA_R_32I,
                                CUDA_R_32I,
                                algoId,
                                &algo);
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo,
        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
        &(customOption),
        sizeof(customOption));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                              CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                              &(splitK_val),
                                              sizeof(splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        &(reductionScheme),
        sizeof(int));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));

    cublasStatus_t status;
    const int repeats = 1;
    for (int loop = 0; loop < repeats; loop++) {
        status = cublasLtMatmul(handle_,
                                    matmul_desc_,
                                    &alpha_,
                                    B_dev,
                                    B_desc_,
                                    A_dev,
                                    A_desc_,
                                    &beta_,
                                    C_dev,
                                    C_desc_,
                                    C_dev,
                                    C_desc_,
                                    &algo,
                                    //  nullptr,
                                    (void*)workspace,
                                    // 0,
                                    work_space_size,
                                    0);
        cudaDeviceSynchronize();
    }
    if (status != cudaSuccess) {
        std::cout  << "CUBLASLT runtime error " << status << "\n";
        std::cout  << "using default gemm\n";
        status = cublasLtMatmul(handle_,
                                    matmul_desc_,
                                    &alpha_,
                                    B_dev,
                                    B_desc_,
                                    A_dev,
                                    A_desc_,
                                    &beta_,
                                    C_dev,
                                    C_desc_,
                                    C_dev,
                                    C_desc_,
                                    &algo,
                                    nullptr,
                                    0,
                                    0);
        cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            std::cout  << "CUBLASLT runtime error " << status << "\n";
            exit(0);
        }
    }
}