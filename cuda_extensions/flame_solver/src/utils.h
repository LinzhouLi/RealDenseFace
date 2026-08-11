#pragma once
#include <cmath>
#include <iostream>
#include <glm/glm.hpp>
#include "constants.h"


__host__ __device__ __forceinline__ glm::mat4 make_affine(const glm::mat3& R, const glm::vec3& t) {
    glm::mat4 M(1.0f); // identity

    // rotation
    M[0][0] = R[0][0]; M[1][0] = R[1][0]; M[2][0] = R[2][0];
    M[0][1] = R[0][1]; M[1][1] = R[1][1]; M[2][1] = R[2][1];
    M[0][2] = R[0][2]; M[1][2] = R[1][2]; M[2][2] = R[2][2];

    // translation
    M[3][0] = t.x;
    M[3][1] = t.y;
    M[3][2] = t.z;

    return M;
}


__host__ __device__ __forceinline__ glm::mat4 inverse_affine(const glm::mat4& M) {
    glm::mat4 M_inv(1.0f); // identity

    // rotation
    M_inv[0][0] = M[0][0]; M_inv[1][0] = M[0][1]; M_inv[2][0] = M[0][2];
    M_inv[0][1] = M[1][0]; M_inv[1][1] = M[1][1]; M_inv[2][1] = M[1][2];
    M_inv[0][2] = M[2][0]; M_inv[1][2] = M[2][1]; M_inv[2][2] = M[2][2];

    // translation
    glm::vec3 t(-M[3][0], -M[3][1], -M[3][2]);
    glm::vec3 t_new = glm::vec3(
        t.x * M_inv[0][0] + t.y * M_inv[1][0] + t.z * M_inv[2][0],
        t.x * M_inv[0][1] + t.y * M_inv[1][1] + t.z * M_inv[2][1],
        t.x * M_inv[0][2] + t.y * M_inv[1][2] + t.z * M_inv[2][2]
    );
    M_inv[3][0] = t_new.x;
    M_inv[3][1] = t_new.y;
    M_inv[3][2] = t_new.z;

    return M_inv;
}


__host__ __device__ __forceinline__ glm::mat3 rotvec_to_rotmat(const glm::vec3& rotvec, float epsilon = 1e-6f) {
    float theta = glm::length(rotvec);

    if (theta < epsilon) {
        // --- First-order approximation ---
        float x = rotvec.x;
        float y = rotvec.y;
        float z = rotvec.z;

        glm::mat3 R(1.0f); // identity
        R[0][1] = -z; R[0][2] =  y;
        R[1][0] =  z; R[1][2] = -x;
        R[2][0] = -y; R[2][1] =  x;
        return R;
    } else {
        // --- Rodrigues formula ---
        glm::vec3 axis = rotvec / glm::max(theta, epsilon);
        float kx = axis.x, ky = axis.y, kz = axis.z;

        float sin_theta = std::sin(theta);
        float cos_theta = std::cos(theta);
        float one_minus_cos = 1.0f - cos_theta;

        float xs = kx * sin_theta;
        float ys = ky * sin_theta;
        float zs = kz * sin_theta;

        float xyc = kx * ky * one_minus_cos;
        float xzc = kx * kz * one_minus_cos;
        float yzc = ky * kz * one_minus_cos;

        float xxc = kx * kx * one_minus_cos;
        float yyc = ky * ky * one_minus_cos;
        float zzc = kz * kz * one_minus_cos;

        glm::mat3 R;
        R[0][0] = 1 - yyc - zzc; R[1][0] = xyc - zs;      R[2][0] = xzc + ys;
        R[0][1] = xyc + zs;      R[1][1] = 1 - xxc - zzc; R[2][1] = yzc - xs;
        R[0][2] = xzc - ys;      R[1][2] = yzc + xs;      R[2][2] = 1 - xxc - yyc;

        return R;
    }
}


__host__ __device__ __forceinline__ glm::mat3 skew_matrix(const glm::vec3& v) {
    return glm::mat3(
        0.0f,   v.z,   -v.y,
        -v.z,   0.0f,   v.x,
        v.y,    -v.x,   0.0f
    );
}


__host__ __device__ __forceinline__ glm::mat3 calc_right_jacobian(const glm::vec3& omega, float epsilon = 1e-6f) {
    float theta = glm::length(omega);
    glm::mat3 identity(1.0f);

    glm::mat3 hat = skew_matrix(omega);
    glm::mat3 hat2 = hat * hat;

    float theta2 = theta * theta;
    float theta3 = theta2 * theta;

    float a, b;

    if (theta < epsilon) {
        return identity - 0.5f * hat + (1.0f / 12.0f) * hat2;
    } else {
        a = (1.0f - std::cos(theta)) / theta2;
        b = (theta - std::sin(theta)) / theta3;
        return identity - a * hat + b * hat2;
    }
}


// TODO: check this
__host__ __device__ __forceinline__ void calc_drotmat_drotvec(const glm::vec3& rotvec, glm::mat3 dR_drot[3], float epsilon = 1e-6f) {
    float theta = glm::length(rotvec);

    if (theta < epsilon) {
        // --- First-order approximation ---
        dR_drot[0] = glm::mat3(
            0.0f,    0.0f,    0.0f,
            0.0f,    0.0f,   -1.0f,
            0.0f,    1.0f,    0.0f
        );
        dR_drot[1] = glm::mat3(
            0.0f,    0.0f,    1.0f,
            0.0f,    0.0f,    0.0f,
           -1.0f,    0.0f,    0.0f
        );
        dR_drot[2] = glm::mat3( 
            0.0f,   -1.0f,    0.0f,
            1.0f,    0.0f,    0.0f,
            0.0f,    0.0f,    0.0f
        );
    } else {
        glm::vec3 axis = rotvec / theta;
        glm::mat3 K = glm::mat3(
            0, -axis.z, axis.y,
            axis.z, 0, -axis.x,
            -axis.y, axis.x, 0
        );

        float sin_t = sinf(theta);
        float cos_t = cosf(theta);
        // float one_minus_cos = 1.0f - cos_t;
        float inv_theta = 1.0f / theta;

        glm::mat3 daxis_drot = (glm::mat3(1.0f) - glm::outerProduct(axis, axis)) * inv_theta;

        for (int i = 0; i < 3; i++) {
            glm::vec3 e(0.0f);
            e[i] = 1.0f;

            float dtheta = axis[i];
            glm::vec3 daxis = daxis_drot * e;
            glm::mat3 dK = glm::mat3(
                0, -daxis.z, daxis.y,
                daxis.z, 0, -daxis.x,
                -daxis.y, daxis.x, 0
            );

            glm::mat3 term1 = sin_t * dK + (1 - cos_t) * (dK * K + K * dK);
            glm::mat3 term2 = (cos_t * dtheta) * K + (sin_t * dtheta) * (K * K);
            dR_drot[i] = term1 + term2;
        }
    }
}


template <typename T>
__forceinline__ __device__ void warp_shuffle_sum(T *value) {
    for (int i = WARP_SIZE / 2; i > 0; i /= 2)
        *value += __shfl_down_sync(0xFFFFFFFF, *value, i);
}


template <typename T>
__forceinline__ __device__ void block_shuffle_sum(T *shared, T *value, int tid) {
    warp_shuffle_sum<T>(value);
    if (tid % WARP_SIZE == 0) shared[tid / WARP_SIZE] = *value;
    __syncthreads();
    if (tid < WARP_SIZE) {
        *value = (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared[tid] : 0;
        warp_shuffle_sum<T>(value);
    }
}


template <typename T>
__forceinline__ __device__ void global_shuffle_sum(T *shared, T *value, int tid, T *global) {
    block_shuffle_sum<T>(shared, value, tid);
    if (tid == 0) atomicAdd(global, *value);
}


inline void print_mat4(const glm::mat4& M) {
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            std::cout << M[col][row] << " "; // 注意 column-major
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

inline void print_mat3(const glm::mat3& M) {
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            std::cout << M[col][row] << " "; // 注意 column-major
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}