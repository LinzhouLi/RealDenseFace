#include "utils.h"
#include "flame_tracking.h"
#include <cassert>


__global__ void AddLambdaKernel(float* H, int n, float lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // H[idx * n + idx] += lambda;

        float h_val = H[idx * n + idx];
        H[idx * n + idx] = h_val + lambda * h_val + 1e-6f;
    }
}


// void FlameTracking::cholesky_solve(
//     cublasHandle_t cublas_handle,
//     cusolverDnHandle_t cusolver_handle,
//     std::function<void*(size_t N)> mem_func,
//     const float* d_J,   // [R, X]
//     const float* d_r,   // [R]
//     float* d_x,         // [X]  
//     const int m,        // num_residuals
//     const int n,        // num_unknowns
//     const float lambda
// ) {
//     float alpha_one = 1.0f;
//     float alpha_zero = 0.0f;
//     float alpha_neg_one = -1.0f;

//     int work_size = 0;
//     cusolverDnSpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, NULL, n, &work_size);

//     char* mem_ptr = (char*)mem_func((n * n + n + work_size) * sizeof(float) + sizeof(int));
//     float* d_H = (float*)mem_ptr;                                                   // [n x n]
//     float* d_g = (float*)(mem_ptr + n * n * sizeof(float));                         // [n]
//     int* d_info = (int*)(mem_ptr + (n * n + n) * sizeof(float));                    // [1]
//     float* d_work = (float*)(mem_ptr + (n * n + n) * sizeof(float) + sizeof(int));  // [work_size]

//     cublasSsyrk(
//         cublas_handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 
//         n, m, 
//         &alpha_one, d_J, m, 
//         &alpha_zero, d_H, n
//     );

//     cublasSgemv(
//         cublas_handle, CUBLAS_OP_T, 
//         m, n, 
//         &alpha_neg_one, d_J, m, 
//         d_r, 1, 
//         &alpha_zero, d_g, 1
//     );

//     int threads = 256;
//     int blocks = (n + threads - 1) / threads;
//     AddLambdaKernel<<<blocks, threads>>>(d_H, n, lambda);

//     cusolverDnSpotrf(
//         cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, 
//         d_H, n, 
//         d_work, work_size, 
//         d_info
//     );

//     cusolverDnSpotrs(
//         cusolver_handle, CUBLAS_FILL_MODE_UPPER, 
//         n, 1, 
//         d_H, n, 
//         d_g, n, 
//         d_info
//     );

//     cudaMemcpyAsync(d_x, d_g, n * sizeof(float), cudaMemcpyDeviceToDevice);
// }


void FlameTracking::cholesky_solve(
    cublasHandle_t cublas_handle,
    cusolverDnHandle_t cusolver_handle,
    std::function<void*(size_t)> mem_func,
    const float* d_J,   // [M, N] (Row-Major) -> cuBLAS sees [N, M] (Col-Major)
    const float* d_r,   // [M]
    float* d_x,         // [N] Output
    const int m,        // num_residuals
    const int n,        // num_unknowns
    const float lambda
) {
    float alpha_one = 1.0f;
    float alpha_zero = 0.0f;
    float alpha_neg_one = -1.0f;

    int work_size = 0;
    cusolverDnSpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, NULL, n, &work_size);

    char* mem_ptr = (char*)mem_func((n * n + n + work_size) * sizeof(float) + sizeof(int));
    float* d_H = (float*)mem_ptr;                           // [n x n]
    float* d_g = (float*)(mem_ptr + n * n * sizeof(float)); // [n]
    int* d_info = (int*)(mem_ptr + (n * n + n) * sizeof(float));
    float* d_work = (float*)(mem_ptr + (n * n + n) * sizeof(float) + sizeof(int));

    // 1. 构建 Hessian: H = J^T * J
    // 内存 J 是 [M, N]。cuBLAS 视为 A [N, M]。
    // 我们要算 J^T * J = A * A^T
    // Ssyrk(OP_N) 计算 A * A^T (当 k=m, lda=n 时)
    cublasSsyrk(
        cublas_handle, CUBLAS_FILL_MODE_UPPER, 
        CUBLAS_OP_N, // 注意这里变成了 OP_N
        n, m,        // n=N (结果维度), k=M (消去维度)
        &alpha_one, d_J, n, // lda=N (因为 CuBLAS 看来有 N 行)
        &alpha_zero, d_H, n
    );

    // 2. 构建 Gradient: g = -J^T * r
    // g = -A * r
    // Sgemv(OP_N) 计算 A * x
    cublasSgemv(
        cublas_handle, 
        CUBLAS_OP_N, // 注意这里变成了 OP_N
        n, m,        // A 的行数=N, 列数=M
        &alpha_neg_one, d_J, n, // lda=N
        d_r, 1,      // incx=1
        &alpha_zero, d_g, 1
    );

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    AddLambdaKernel<<<blocks, threads>>>(d_H, n, lambda);

    cusolverDnSpotrf(
        cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, 
        d_H, n, 
        d_work, work_size, 
        d_info
    );

    cusolverDnSpotrs(
        cusolver_handle, CUBLAS_FILL_MODE_UPPER, 
        n, 1, 
        d_H, n, 
        d_g, n, 
        d_info
    );

    cudaMemcpyAsync(d_x, d_g, n * sizeof(float), cudaMemcpyDeviceToDevice);
}


__global__ void AddLambdaBatchedKernel(float* H_batch, int n, int batch_size, int stride_H, float lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * batch_size; 
    if (idx >= total_elements) return;

    int b = idx / n;
    int i = idx % n;
    
    if (b < batch_size) {
        long long offset = (long long)b * stride_H + i * n + i;
        // H_batch[offset] += lambda;

        float h_val = H_batch[offset];
        H_batch[offset] = h_val + lambda * h_val + 1e-6f;
    }
}


__global__ void SetupPointerArray(
    float* data_base, 
    int stride, 
    float** ptr_array, 
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        ptr_array[idx] = data_base + idx * stride;
    }
}


// void FlameTracking::cholesky_solve_batched(
//     cublasHandle_t cublas_handle,
//     cusolverDnHandle_t cusolver_handle,
//     std::function<void*(size_t)> mem_func,
//     const float* d_J,   // [B, R, X]
//     const float* d_r,   // [B, X]
//     float* d_x,         // [B, X]
//     const int m,        // num_residuals
//     const int n,        // num_unknowns
//     const int batch_size,
//     const float lambda
// ) {
//     float alpha_one = 1.0f;
//     float alpha_zero = 0.0f;
//     float alpha_neg_one = -1.0f;

//     long long stride_J = (long long)m * n;
//     long long stride_r = m;
//     long long stride_x = n;
//     long long stride_H = (long long)n * n;
//     long long stride_g = n;

//     size_t size_H_batch = batch_size * stride_H * sizeof(float);
//     size_t size_g_batch = batch_size * stride_g * sizeof(float);
//     size_t size_info    = batch_size * sizeof(int);
//     size_t size_ptrs    = batch_size * sizeof(float*) * 2; // H_ptrs + g_ptrs
    
//     char* mem_ptr = (char*)mem_func(size_H_batch + size_g_batch + size_info + size_ptrs);
    
//     float* d_H_batch = (float*)mem_ptr;
//     float* d_g_batch = (float*)(mem_ptr + size_H_batch);
//     int* d_info    = (int*)  (mem_ptr + size_H_batch + size_g_batch);
//     float** d_H_ptrs = (float**)(mem_ptr + size_H_batch + size_g_batch + size_info);
//     float** d_g_ptrs = d_H_ptrs + batch_size;

//     cublasSgemmStridedBatched(
//         cublas_handle, 
//         CUBLAS_OP_T,
//         CUBLAS_OP_N,
//         n, n,  m, 
//         &alpha_one,
//         d_J, m, stride_J,
//         d_J, m, stride_J,
//         &alpha_zero,
//         d_H_batch, n, stride_H,
//         batch_size
//     );

//     cublasSgemmStridedBatched(
//         cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
//         n, 1, m,
//         &alpha_neg_one,
//         d_J, m, stride_J,
//         d_r, m, stride_r,
//         &alpha_zero,
//         d_g_batch, n, stride_g,
//         batch_size
//     );

//     int total_threads = n * batch_size;
//     AddLambdaBatchedKernel<<<(total_threads + 255)/256, 256>>>(d_H_batch, n, batch_size, stride_H, lambda);

//     SetupPointerArray<<<(batch_size + 255)/256, 256>>>(d_H_batch, stride_H, d_H_ptrs, batch_size);
//     SetupPointerArray<<<(batch_size + 255)/256, 256>>>(d_g_batch, stride_g, d_g_ptrs, batch_size);

//     cusolverDnSpotrfBatched(
//         cusolver_handle, CUBLAS_FILL_MODE_UPPER,
//         n, 
//         d_H_ptrs, n,
//         d_info, 
//         batch_size
//     );

//     cusolverDnSpotrsBatched(
//         cusolver_handle, CUBLAS_FILL_MODE_UPPER,
//         n, 1, // nrhs = 1
//         d_H_ptrs, n, 
//         d_g_ptrs, n,
//         d_info,
//         batch_size
//     );

//     cudaMemcpyAsync(d_x, d_g_batch, batch_size * n * sizeof(float), cudaMemcpyDeviceToDevice);
// }


size_t align_up(size_t x, size_t a) {
    return (x + a - 1) / a * a;
}


void FlameTracking::cholesky_solve_batched(
    cublasHandle_t cublas_handle,
    cusolverDnHandle_t cusolver_handle,
    std::function<void*(size_t)> mem_func,
    const float* d_J,   // [Batch, M, N] (Row-Major)
    const float* d_r,   // [Batch, M]
    float* d_x,         // [Batch, N] Output
    const int m,        // num_residuals
    const int n,        // num_unknowns
    const int batch_size,
    const float lambda
) {
    float alpha_one = 1.0f;
    float alpha_zero = 0.0f;
    float alpha_neg_one = -1.0f;

    long long stride_J = (long long)m * n;
    long long stride_r = m;
    // long long stride_x = n;
    long long stride_H = (long long)n * n;
    long long stride_g = n;

    size_t size_H_batch = batch_size * stride_H * sizeof(float);
    size_t size_g_batch = batch_size * stride_g * sizeof(float);
    size_t size_info    = batch_size * sizeof(int);
    size_t size_ptrs    = batch_size * sizeof(float*) * 2; 
    
    // char* mem_ptr = (char*)mem_func(size_H_batch + size_g_batch + size_info + size_ptrs);
    
    // float* d_H_batch = (float*)mem_ptr;
    // float* d_g_batch = (float*)(mem_ptr + size_H_batch);
    // int* d_info    = (int*)  (mem_ptr + size_H_batch + size_g_batch);
    // float** d_H_ptrs = (float**)(mem_ptr + size_H_batch + size_g_batch + size_info);
    // float** d_g_ptrs = d_H_ptrs + batch_size;

    size_t off = 0;
    off = align_up(off, 16);
    size_t off_H = off; off += size_H_batch;
    off = align_up(off, 16);
    size_t off_g = off; off += size_g_batch;
    off = align_up(off, alignof(int));
    size_t off_info = off; off += size_info;
    off = align_up(off, alignof(float*));
    size_t off_Hptrs = off; off += batch_size * sizeof(float*);
    off = align_up(off, alignof(float*));
    size_t off_gptrs = off; off += batch_size * sizeof(float*);

    char* mem_ptr = (char*)mem_func(off);

    float*  d_H_batch = (float*)(mem_ptr + off_H);
    float*  d_g_batch = (float*)(mem_ptr + off_g);
    int*    d_info    = (int*)(mem_ptr + off_info);
    float** d_H_ptrs  = (float**)(mem_ptr + off_Hptrs);
    float** d_g_ptrs  = (float**)(mem_ptr + off_gptrs);

    // 1. 构建 Hessian: H = J^T * J = A * A^T
    // A 是 [N, M] (Col-Major view)
    // Sgemm: C = Op(A) * Op(B)
    // Op(A)=N -> N x M
    // Op(B)=T -> M x N
    // Result -> N x N
    cublasSgemmStridedBatched(
        cublas_handle, 
        CUBLAS_OP_N, // Op(A)
        CUBLAS_OP_T, // Op(B)
        n, n, m,     // m=N, n=N, k=M
        &alpha_one,
        d_J, n, stride_J, // lda=N
        d_J, n, stride_J, // ldb=N
        &alpha_zero,
        d_H_batch, n, stride_H,
        batch_size
    );

    // 2. 构建 Gradient: g = -J^T * r = -A * r
    // A 是 [N, M]
    // r 是 [M, 1]
    // Result -> [N, 1]
    cublasSgemmStridedBatched(
        cublas_handle, 
        CUBLAS_OP_N, // Op(A) = A (N x M)
        CUBLAS_OP_N, // Op(B) = r (M x 1)
        n, 1, m,     // m=N, n=1, k=M
        &alpha_neg_one,
        d_J, n, stride_J, // lda=N
        d_r, m, stride_r, // ldb=M (对于向量列，ldb至少是行数)
        &alpha_zero,
        d_g_batch, n, stride_g,
        batch_size
    );

    int total_threads = n * batch_size;
    AddLambdaBatchedKernel<<<(total_threads + 255)/256, 256>>>(d_H_batch, n, batch_size, stride_H, lambda);

    SetupPointerArray<<<(batch_size + 255)/256, 256>>>(d_H_batch, stride_H, d_H_ptrs, batch_size);
    SetupPointerArray<<<(batch_size + 255)/256, 256>>>(d_g_batch, stride_g, d_g_ptrs, batch_size);

    cusolverDnSpotrfBatched(
        cusolver_handle, CUBLAS_FILL_MODE_UPPER,
        n, 
        d_H_ptrs, n,
        d_info, 
        batch_size
    );

    cusolverDnSpotrsBatched(
        cusolver_handle, CUBLAS_FILL_MODE_UPPER,
        n, 1, 
        d_H_ptrs, n, 
        d_g_ptrs, n,
        d_info,
        batch_size
    );

    cudaMemcpyAsync(d_x, d_g_batch, batch_size * n * sizeof(float), cudaMemcpyDeviceToDevice);
}