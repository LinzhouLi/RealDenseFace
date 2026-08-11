#include "utils.h"
#include "flame_tracking.h"


namespace CalcCanonical {

template<int N_VERTS>
__global__ void FlameIdentityForwardCuda(
    const float* __restrict__ identity,      // [300]
    const glm::vec3* __restrict__ v_template,   // [V, 3]
    const glm::vec3* __restrict__ shape_dirs,   // [B, V, 3]
    glm::vec3* v_output                         // [V, 3]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_VERTS) return;

    glm::vec3 vertex = v_template[idx];

    for (int l = 0; l < NUM_FLAME_IDENTITY_BASIS; l++) {
        glm::vec3 shape_vec = shape_dirs[l * N_VERTS + idx];
        vertex += identity[l] * shape_vec;
    }

    v_output[idx] = vertex;
}


__global__ void FlameJointRegressionCuda(
    const int num_nonzero,
    const glm::vec3* __restrict__ vertices,            // [V, 3]
    const int* __restrict__ J_regressor_row,        // [NNZ]
    const int* __restrict__ J_regressor_col,        // [NNZ]
    const float* __restrict__ J_regressor_values,   // [NNZ]
    glm::vec3* joints                                  // [J, 3]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nonzero) return;

    glm::vec3 vertex = vertices[J_regressor_col[idx]];
    float weight = J_regressor_values[idx];
    int joint_id = J_regressor_row[idx];
    glm::vec3 joint = weight * vertex;

    atomicAdd(&joints[joint_id][0], joint[0]);
    atomicAdd(&joints[joint_id][1], joint[1]);
    atomicAdd(&joints[joint_id][2], joint[2]);
}

}


template<int N_VERTS>
static void launch_flame_calc_canonical_impl(
    const float* identity,
    const glm::vec3* v_template,
    const glm::vec3* shape_dirs,

    const int num_J_regressor_nonzero,
    const int* J_regressor_row,
    const int* J_regressor_col,
    const float* J_regressor_values,

    glm::vec3* v_output,
    glm::vec3* J_output
) {
    int threads = 256;

    int blocks = (N_VERTS + threads - 1) / threads;
    CalcCanonical::FlameIdentityForwardCuda<N_VERTS><<<blocks, threads>>>(
        identity, v_template, shape_dirs, v_output
    );

    blocks = (num_J_regressor_nonzero + threads - 1) / threads;
    CalcCanonical::FlameJointRegressionCuda<<<blocks, threads>>>(
        num_J_regressor_nonzero, v_output, 
        J_regressor_row, J_regressor_col, J_regressor_values, 
        J_output
    );
}


void FlameTracking::flame_calc_canonical(
    const int num_vertices,
    const float* identity,
    const glm::vec3* v_template,
    const glm::vec3* shape_dirs,

    const int num_J_regressor_nonzero,
    const int* J_regressor_row,
    const int* J_regressor_col,
    const float* J_regressor_values,

    glm::vec3* v_output,
    glm::vec3* J_output
) {
    switch (num_vertices) {
#define DISPATCH_CASE(N) \
        case N: \
            launch_flame_calc_canonical_impl<N>( \
                identity, v_template, shape_dirs, \
                num_J_regressor_nonzero, J_regressor_row, J_regressor_col, J_regressor_values, \
                v_output, J_output \
            ); \
            break;
        FLAME_FOREACH_NVERTS(DISPATCH_CASE)
#undef DISPATCH_CASE
        default:
            throw std::runtime_error("flame_calc_canonical: unsupported num_vertices");
    }
}
