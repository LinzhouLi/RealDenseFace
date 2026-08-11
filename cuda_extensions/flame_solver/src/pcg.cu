#include "utils.h"
#include "flame_tracking.h"
#include <cassert>


__global__ void ComputeDiagJtJCuda(
    const float* __restrict__ J, 
    float* __restrict__ diag, 
    const int m, const int n
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < m; i += blockDim.x) {
        float v = J[i + bid * m];
        sum += v * v;
    }

    __shared__ float shared[WARP_SIZE]; // assuming blockDim.x <= 1024 (32 * 32)
    block_shuffle_sum<float>(shared, &sum, tid);

    if (tid == 0) diag[bid] = float(sum);
}


__global__ void ComputeMInvCuda(
    float* __restrict__ diag,
    const float lam,
    const int n
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    diag[j] = 1.0f / std::max(diag[j] + lam, 1e-6f);
}


__global__ void ApplyPreconditionCuda(
    const float* __restrict__ M_inv, 
    const float* __restrict__ src, 
    float* __restrict__ out, 
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = M_inv[idx] * src[idx];
}


void pcg_solve_legacy(
    cublasHandle_t handle,
    const float* J,
    const float* f,
    float* x,
    const int m, 
    const int n,
    const float lambda,
    const int iters
) {
    float one = 1.0f, zero = 0.0f, neg_one = -1.0f;

    float* M, *z, *r, *p, *Ap, *tmp_m;
    cudaMalloc((void**)&M, sizeof(float) * n);
    cudaMalloc((void**)&z, sizeof(float) * n);
    cudaMalloc((void**)&r, sizeof(float) * n);
    cudaMalloc((void**)&p, sizeof(float) * n);
    cudaMalloc((void**)&Ap, sizeof(float) * n);
    cudaMalloc((void**)&tmp_m, sizeof(float) * m);

    ComputeDiagJtJCuda<<<n, 1024>>>(J, M, m, n);
    ComputeMInvCuda<<<1, n>>>(M, lambda, n);

    // x = 0
    cudaMemsetAsync(x, 0, n * sizeof(float));

    // r = -JTf
    cublasSgemv(handle, CUBLAS_OP_T, m, n, &neg_one, J, m, f, 1, &zero, r, 1);

    // z = M * r
    ApplyPreconditionCuda<<<1, n>>>(M, r, z, n);

    // p = z
    cudaMemcpyAsync(p, z, n * sizeof(float), cudaMemcpyDeviceToDevice);

    // rz_old = dot(r, z)
    float rz_old = 0; float rz_new = 0;
    cublasSdot(handle, n, r, 1, z, 1, &rz_old);

    for (int i = 0; i < iters; i++) {
        // Ap = A * p = J^T * (J * p) + lambda * p
        cublasSgemv(handle, CUBLAS_OP_N, m, n, &one, J, m, p, 1, &zero, tmp_m, 1);
        cublasSgemv(handle, CUBLAS_OP_T, m, n, &one, J, m, tmp_m, 1, &zero, Ap, 1);
        cublasSaxpy(handle, n, &lambda, p, 1, Ap, 1);

        // denom = dot(p, Ap)
        float denom = 0;
        cublasSdot(handle, n, p, 1, Ap, 1, &denom);

        // x = x + alpha * p
        // r = r - alpha * Ap
        float alpha = rz_old / std::max(denom, 1e-6f);
        float neg_alpha = -alpha;
        cublasSaxpy(handle, n, &alpha, p, 1, x, 1);
        cublasSaxpy(handle, n, &neg_alpha, Ap, 1, r, 1);

        // z = M * r
        ApplyPreconditionCuda<<<1, n>>>(M, r, z, n);

        // rz_new = dot(r, z)
        cublasSdot(handle, n, r, 1, z, 1, &rz_new);

        // p = z + beta * p
        float beta = rz_new / std::max(rz_old, 1e-6f);
        cublasSscal(handle, n, &beta, p, 1);
        cublasSaxpy(handle, n, &one, z, 1, p, 1);

        rz_old = rz_new;
    }

    cudaFree(M);
    cudaFree(z);
    cudaFree(r);
    cudaFree(p);
    cudaFree(Ap);
    cudaFree(tmp_m);
}


__global__ void KernelA(
    const float* __restrict__ r,
    float* __restrict__ M,
    float* __restrict__ p,
    float* __restrict__ x,
    float* __restrict__ rz_old,
    const float lambda,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float r_val = 0.0f, M_val = 0.0f, z_val = 0.0f;
    if (idx < n) {
        r_val = r[idx];
        M_val = M[idx];

        float inv_M = 1.0f / std::max(M_val + lambda, 1e-6f); // M_inv = 1.0 / (M + lambda)
        z_val = inv_M * r_val; // z = M * r

        x[idx] = 0.0f; // x = 0
        p[idx] = z_val; // p = z
        M[idx] = inv_M;
    }

    // rz_old = dot(r, z)
    float rz = r_val * z_val;
    __shared__ float shared[WARP_SIZE]; // assuming blockDim.x <= 1024 (32 * 32)
    block_shuffle_sum<float>(shared, &rz, tid);
    if (tid == 0) { *rz_old = rz; }
}


__global__ void KernelB(
    const float* __restrict__ M,
    float* __restrict__ Ap,
    float* __restrict__ p,
    float* __restrict__ x,
    float* __restrict__ r,
    float* __restrict__ rz_old,
    const float lambda,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ float shared[WARP_SIZE]; // assuming blockDim.x <= 1024 (32 * 32)
    __shared__ float alpha;
    __shared__ float beta;

    float rz_old_val = 0.0f;
    if (tid == 0) rz_old_val = *rz_old;

    float p_val = 0.0f, Ap_val = 0.0f;
    if (idx < n) {
        p_val = p[idx];
        Ap_val = Ap[idx];
        Ap_val += lambda * p_val; // Ap = Ap + lambda * p
    }

    // pAp = dot(p, Ap)
    float pAp = p_val * Ap_val;
    block_shuffle_sum<float>(shared, &pAp, tid);
    if (tid == 0) {
        alpha = rz_old_val / std::max(pAp, 1e-6f); // alpha = rz_old / dot(p, Ap)
    }
    __syncthreads();

    float x_val = 0.0f, r_val = 0.0f, z_val = 0.0f;
    if (idx < n) {
        x_val = x[idx];
        r_val = r[idx];

        x_val += alpha * p_val; // x = x + alpha * p
        r_val -= alpha * Ap_val; // r = r - alpha * Ap
        z_val = M[idx] * r_val; // z = M * r
    }

    // rz = dot(r, z)
    float rz = r_val * z_val;
    block_shuffle_sum<float>(shared, &rz, tid);
    if (tid == 0) {
        beta = rz / std::max(rz_old_val, 1e-6f); // beta = rz_new / rz_old
    }
    __syncthreads();

    if (idx < n) {
        p_val = beta * p_val + z_val; // p = z + beta * p

        x[idx] = x_val;
        r[idx] = r_val;
        p[idx] = p_val;
    }

    if (tid == 0) { *rz_old = rz; }
}


void pcg_solve(
    cublasHandle_t handle,
    std::function<float*(size_t N)> mem_func,
    const float* J,
    const float* f,
    float* x,
    const int m, 
    const int n,
    const float lambda,
    const int iters
) {
    float one = 1.0f, zero = 0.0f, neg_one = -1.0f;

    float* mem_ptr = mem_func(4 * n + m + 1);
    float* M = mem_ptr;
    float* r = M + n;
    float* p = r + n;
    float* Ap = p + n;
    float* tmp_m = Ap + n;
    float* rz_old = tmp_m + m;

    ComputeDiagJtJCuda<<<n, 512>>>(J, M, m, n);

    // r = -JTf
    cublasSgemv(handle, CUBLAS_OP_T, m, n, &neg_one, J, m, f, 1, &zero, r, 1);

    assert(n <= WARP_SIZE * WARP_SIZE);
    int threads = (n + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    KernelA<<<1, threads>>>(r, M, p, x, rz_old, lambda, n);

    for (int i = 0; i < iters; i++) {
        // Ap = A * p = J^T * (J * p)
        cublasSgemv(handle, CUBLAS_OP_N, m, n, &one, J, m, p, 1, &zero, tmp_m, 1);
        cublasSgemv(handle, CUBLAS_OP_T, m, n, &one, J, m, tmp_m, 1, &zero, Ap, 1);
        KernelB<<<1, threads>>>(M, Ap, p, x, r, rz_old, lambda, n);
    }
}