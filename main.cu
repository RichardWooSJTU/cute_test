#include <limits>
#include "cublaslt_gemm.cuh"
#include "cute_gemm.cuh"

template<typename T>
void print_matrix(const thrust::host_vector<T>& X, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << X[i*n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<>
void print_matrix(const thrust::host_vector<int8_t>& X, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << static_cast<int>(X[i*n + j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<typename T>
bool is_matrix_equal(const thrust::host_vector<T>& A, const thrust::host_vector<T>& B, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (A[i * n + j] != B[i * n + j]) {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char **argv) {
    using namespace cute;
    assert(argc == 4);

    int m = 16;
    int n = 16;
    int k = 16;

    sscanf(argv[1], "%d", &m);
    sscanf(argv[2], "%d", &n);
    sscanf(argv[3], "%d", &k);

    std::cout << "M = " << m << std::endl;
    std::cout << "N = " << n << std::endl;
    std::cout << "K = " << k << std::endl;

    using TA = int8_t;
    using TB = int8_t;
    using TC = int32_t;


    thrust::host_vector<TA> h_A(m*k);
    thrust::host_vector<TB> h_B(n*k);
    thrust::host_vector<TC> cute_h_C(m*n);
    thrust::host_vector<TC> cublaslt_h_C(m*n);

    // for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( (2 * (rand() / double(RAND_MAX)) - 1) * std::numeric_limits<int8_t>::max() );
    // for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( (2 * (rand() / double(RAND_MAX)) - 1) * std::numeric_limits<int8_t>::max() );

    for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( j);
    for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( j );
    for (int j = 0; j < m*n; ++j) cute_h_C[j] = static_cast<TC>(-1);
    for (int j = 0; j < m*n; ++j) cublaslt_h_C[j] = static_cast<TC>(-1);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = cute_h_C;

    // print_matrix(h_A, m, k);
    // print_matrix(h_B, n, k);


    std::cout << "===============cute gemm===================\n";

    LaunchCuteGEMM(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);

    cute_h_C = d_C;


    // print_matrix(h_C, m, n);


    std::cout << "===============cublaslt gemm===================\n";
    d_C = cublaslt_h_C;
    LaunchCublasLtGEMM(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);

    cublaslt_h_C = d_C;
    // print_matrix(h_C, m, n);
    if (is_matrix_equal(cute_h_C, cublaslt_h_C, m, n)) {
        std::cout << "=========SUCCESS==========\n";
    } else {
        std::cout << "=========FAILED==========\n";
    }


}