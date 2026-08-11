#include "utils.h"
#include "flame_tracking.h"
#include <cassert>


void lm_solve(
    cublasHandle_t handle,
    std::function<void(const float* x, float* r)> compute_residual,
    std::function<void(const float* x, float* r, float* J)> compute_jacobian,
    float* x, 
    float* r, 
    float* J,
    float* H,
    const int m, // num_residuals
    const int n, // num_unknowns
    const int max_iters
) {
    float alpha_one = 1.0f;
    float alpha_zero = 0.0f;
    float alpha_neg_one = -1.0f;

    compute_jacobian(x, r, J);

    float error_curr = 0.0f;
    cublasSnrm2(handle, m, r, 1, &error_curr);
    error_curr = 0.5f * error_curr * error_curr;

    for (int iter = 0; iter < max_iters; ++iter) {
        cublasSsyrk(
            cublas_handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 
            n, m, 
            &alpha_one, d_J, m,
            &alpha_zero, d_H, n
        );

        // g = -J^T * r
        // g = alpha * op(A) * x + beta * y
        cublasSgemv(
            cublas_handle, CUBLAS_OP_T, 
            m, n, 
            &alpha_neg_one, d_J, m, 
            d_r, 1, 
            &alpha_zero, d_g, 1
        );

        bool step_accepted = false;
        int reject_count = 0;
    }
}


///// gemini
void lm_solve(
    cublasHandle_t cublas_handle,
    cusolverDnHandle_t cusolver_handle, // 需要传入 cuSOLVER 句柄
    std::function<void(const float* x, float* r)> compute_residual,
    std::function<void(const float* x, float* r, float* J)> compute_jacobian,
    float* d_x,         // [IN/OUT] Params (N)
    float* d_r,         // [Buffer] Residuals (M)
    float* d_J,         // [Buffer] Jacobian (N x M in memory -> M x N in cuBLAS view)
    float* d_H,         // [Buffer] Hessian (N x N)
    const int m,        // num_residuals
    const int n,        // num_unknowns
    const int max_iters
) {
    // 常量定义
    float alpha_one = 1.0f;
    float alpha_zero = 0.0f;
    float alpha_neg_one = -1.0f;

    // LM 参数
    float lambda = 1e-4f;
    const float lambda_min = 1e-6f;
    const float lambda_max = 1e10f;

    // -------------------------------------------------------
    // 1. 临时内存分配 (建议在类成员中预分配，这里为了演示直接Malloc)
    // -------------------------------------------------------
    float *d_g, *d_dx, *d_x_new, *d_r_new, *d_H_damp;
    int *d_info;
    void *d_work;
    int work_size = 0;

    cudaMalloc(&d_g, n * sizeof(float));
    cudaMalloc(&d_dx, n * sizeof(float));       // 存放求解出的 delta_x
    cudaMalloc(&d_x_new, n * sizeof(float));    // 存放试探步 x_new
    cudaMalloc(&d_r_new, m * sizeof(float));    // 存放试探步 r_new
    cudaMalloc(&d_H_damp, n * n * sizeof(float)); // 存放加了阻尼的 H
    cudaMalloc(&d_info, sizeof(int));

    // 获取 cuSolver 需要的 Workspace 大小 (Cholesky: potrf)
    cusolverDnSpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, d_H_damp, n, &work_size);
    cudaMalloc(&d_work, work_size);

    // -------------------------------------------------------
    // 2. 初始化 (Iter 0)
    // -------------------------------------------------------
    // 计算初始 J 和 r
    compute_jacobian(d_x, d_r, d_J);

    // 计算初始 Error
    float error_curr = 0.0f;
    cublasSnrm2(cublas_handle, m, d_r, 1, &error_curr);
    error_curr = 0.5f * error_curr * error_curr;

    // -------------------------------------------------------
    // 3. Optimization Loop
    // -------------------------------------------------------
    for (int iter = 0; iter < max_iters; ++iter) {
        
        // Step A: 构建线性系统 (Build System)
        // ---------------------------------------------------
        // H = J^T * J
        // J shape: [num_unknowns, num_residuals] (Rows=N, Cols=M)
        // cuBLAS 默认列主序，所以视作 [M, N] 矩阵 (M rows, N cols)
        // cublasSsyrk(OP_T, N, M, A, lda=M) => A^T * A => (M x N)^T * (M x N) => (N x M) * (M x N) => N x N
        // 结果 H 是 N x N
        cublasSsyrk(
            cublas_handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, 
            n, m, 
            &alpha_one, d_J, m,
            &alpha_zero, d_H, n
        );

        // g = -J^T * r
        // Sgemv(OP_T, M, N, A, lda=M, x) => A^T * x => (M x N)^T * (M x 1) => N x 1
        cublasSgemv(
            cublas_handle, CUBLAS_OP_T, 
            m, n, 
            &alpha_neg_one, d_J, m, 
            d_r, 1, 
            &alpha_zero, d_g, 1
        );

        bool step_accepted = false;
        int reject_count = 0;

        // ---------------------------------------------------
        // Step B: Inner Loop (Adjust Lambda)
        // ---------------------------------------------------
        while (!step_accepted && reject_count < 10) {
            
            // 1. 准备 H_damp = H
            cudaMemcpy(d_H_damp, d_H, n * n * sizeof(float), cudaMemcpyDeviceToDevice);

            // 2. 加阻尼: H_damp += lambda * I
            int blocks = (n + 255) / 256;
            AddLambdaKernel<<<blocks, 256>>>(d_H_damp, n, lambda);

            // 3. Cholesky 分解: H = L * L^T
            cusolverDnSpotrf(
                cusolver_handle, CUBLAS_FILL_MODE_UPPER, 
                n, d_H_damp, n, 
                d_work, work_size, d_info
            );

            // 检查分解是否成功 (检查正定性)
            int h_info = 0;
            cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost); // 同步点

            if (h_info != 0) {
                // H 非正定 (Lambda 太小)，加大阻尼重试
                lambda *= 10.0f;
                reject_count++;
                continue;
            }

            // 4. 求解: H * dx = g -> dx = H^-1 * g
            // 结果直接覆盖在 d_dx 中 (初始 d_dx = d_g)
            cudaMemcpy(d_dx, d_g, n * sizeof(float), cudaMemcpyDeviceToDevice);
            
            cusolverDnSpotrs(
                cusolver_handle, CUBLAS_FILL_MODE_UPPER, 
                n, 1, 
                d_H_damp, n, 
                d_dx, n, 
                d_info
            );

            // 5. 试探更新: x_new = x + dx
            UpdateParamKernel<<<blocks, 256>>>(d_x, d_dx, d_x_new, n);

            // 6. 评估: 计算 r_new
            compute_residual(d_x_new, d_r_new);

            float error_new = 0.0f;
            cublasSnrm2(cublas_handle, m, d_r_new, 1, &error_new);
            error_new = 0.5f * error_new * error_new;

            // 7. 接受/拒绝策略
            if (error_new < error_curr) {
                // [ACCEPT]
                step_accepted = true;
                
                // 更新状态
                cudaMemcpy(d_x, d_x_new, n * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_r, d_r_new, m * sizeof(float), cudaMemcpyDeviceToDevice);
                error_curr = error_new;

                // 减小阻尼 (Trust Region 扩大)
                lambda = std::max(lambda_min, lambda * 0.1f);
            } else {
                // [REJECT]
                // 增大阻尼 (Trust Region 缩小)
                lambda = std::min(lambda_max, lambda * 10.0f);
                reject_count++;
            }
        } // End Inner Loop

        // 如果步长被接受，且不是最后一次迭代，重新计算 Jacobian 供下一次使用
        if (step_accepted && iter < max_iters - 1) {
            compute_jacobian(d_x, d_r, d_J);
        } else if (!step_accepted) {
            // 如果尝试多次仍无法下降，提前终止
            break;
        }

        // 简单的收敛判断 (可选)
        if (error_curr < 1e-6f) break;

    } // End Optimization Loop

    // 释放内存
    cudaFree(d_g); cudaFree(d_dx); cudaFree(d_x_new);
    cudaFree(d_r_new); cudaFree(d_H_damp); cudaFree(d_info); cudaFree(d_work);
}