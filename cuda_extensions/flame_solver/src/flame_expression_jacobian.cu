#include "utils.h"
#include "flame_tracking.h"


namespace ExpressionJacobian {

template<int N_EXPR>
__device__ __forceinline__ void write_pose_effect(
    float* jacobian_base,
    int output_joint_k,
    int input_pose_j,
    const glm::vec3& output_pos,
    const glm::vec3& pivot_pos,
    const glm::mat3& axis_j
) {
    constexpr int DIM_X = dim_x_for(N_EXPR);
    glm::vec3 lever = output_pos - pivot_pos;

    glm::vec3 col0 = glm::cross(glm::vec3(axis_j[0]), lever);
    glm::vec3 col1 = glm::cross(glm::vec3(axis_j[1]), lever);
    glm::vec3 col2 = glm::cross(glm::vec3(axis_j[2]), lever);

    int p_pose = N_EXPR;
    int offset = (output_joint_k * DIM_X + p_pose + input_pose_j * 3) * 3;

    jacobian_base[offset + 0] = col0.x; jacobian_base[offset + 1] = col0.y; jacobian_base[offset + 2] = col0.z;
    jacobian_base[offset + 3] = col1.x; jacobian_base[offset + 4] = col1.y; jacobian_base[offset + 5] = col1.z;
    jacobian_base[offset + 6] = col2.x; jacobian_base[offset + 7] = col2.y; jacobian_base[offset + 8] = col2.z;
}


template<int N_EXPR>
__global__ void SkeletonTreeCuda(
    const bool calc_jacobian,
    const int batch_size,
    const glm::vec3* __restrict__ j_canonical,
    const float* __restrict__ batched_x,

    float* __restrict__ batched_pose_feat,
    glm::mat4* __restrict__ batched_affine_mats,
    glm::mat3* __restrict__ batched_right_jacobian,
    glm::mat4* __restrict__ batched_B_masked,
    glm::mat3* __restrict__ batched_Ar,

    glm::vec3* __restrict__ batched_j_output,            // [B, J, 3]
    float* __restrict__ batched_j_jacobian               // [B, J, DIM_X, 3]
) {
    constexpr int DIM_X = dim_x_for(N_EXPR);

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    const float* x = batched_x + batch_idx * DIM_X;
    float* pose_feat = batched_pose_feat + batch_idx * NUM_FLAME_POSEFEAT_BASIS;
    glm::mat4* affine_mats = batched_affine_mats + batch_idx * NUM_FLAME_JOINTS;
    glm::vec3* j_output = batched_j_output + batch_idx * NUM_FLAME_JOINTS;

    const glm::vec3 translation = glm::vec3(
        x[N_EXPR + NUM_FLAME_JOINTS * 3 + 0],
        x[N_EXPR + NUM_FLAME_JOINTS * 3 + 1],
        x[N_EXPR + NUM_FLAME_JOINTS * 3 + 2]
    );

    glm::vec3 rot_vecs[NUM_FLAME_JOINTS];
    glm::mat3 rot_mats[NUM_FLAME_JOINTS];
    for (int i = 0; i < NUM_FLAME_JOINTS; i++) {
        rot_vecs[i] = glm::vec3(
            x[N_EXPR + i * 3],
            x[N_EXPR + i * 3 + 1],
            x[N_EXPR + i * 3 + 2]
        );

        rot_mats[i] = rotvec_to_rotmat(rot_vecs[i]);

        if (i > 0) {
            glm::mat3 pose_feat_mat = rot_mats[i] - glm::mat3(1.0f);
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    pose_feat[(i - 1) * 9 + r * 3 + c] = pose_feat_mat[r][c];
        }
    }

    // G
    affine_mats[0] = make_affine(rot_mats[0], j_canonical[0]); // rel_affine_mats[0], rel_joints[0]
    affine_mats[1] = affine_mats[0] * make_affine(rot_mats[1], j_canonical[1] - j_canonical[0]); // rel_affine_mats[1], rel_joints[1]
    affine_mats[2] = affine_mats[1] * make_affine(rot_mats[2], j_canonical[2] - j_canonical[1]); // rel_affine_mats[2], rel_joints[2]
    affine_mats[3] = affine_mats[1] * make_affine(rot_mats[3], j_canonical[3] - j_canonical[1]); // rel_affine_mats[3], rel_joints[3]
    affine_mats[4] = affine_mats[1] * make_affine(rot_mats[4], j_canonical[4] - j_canonical[1]); // rel_affine_mats[4], rel_joints[4]

    // j output
    glm::vec3 joints[NUM_FLAME_JOINTS];
    for (int j = 0; j < NUM_FLAME_JOINTS; j++) {
        joints[j] = glm::vec3(affine_mats[j][3]);
        j_output[j] = joints[j] + translation;
    }

    if (calc_jacobian) {
        glm::mat3* right_jacobian = batched_right_jacobian + batch_idx * NUM_FLAME_JOINTS;
        glm::mat4* B_masked = batched_B_masked + batch_idx * NUM_FLAME_JOINTS * NUM_FLAME_JOINTS;
        glm::mat3* Ar = batched_Ar + batch_idx * NUM_FLAME_JOINTS;
        float* jacobian = batched_j_jacobian + batch_idx * NUM_FLAME_JOINTS * DIM_X * 3;

        for (int i = 0; i < NUM_FLAME_JOINTS; i++) {
            right_jacobian[i] = calc_right_jacobian(rot_vecs[i]);
        }

        // G_inv
        glm::mat4 inv_affine_mats[NUM_FLAME_JOINTS];
        for (int i = 0; i < NUM_FLAME_JOINTS; i++) {
            inv_affine_mats[i] = inverse_affine(affine_mats[i]);
        }

        // B_masked
        B_masked[0 * NUM_FLAME_JOINTS + 0] = inv_affine_mats[0] * affine_mats[0];
        B_masked[0 * NUM_FLAME_JOINTS + 1] = inv_affine_mats[0] * affine_mats[1];
        B_masked[0 * NUM_FLAME_JOINTS + 2] = inv_affine_mats[0] * affine_mats[2];
        B_masked[0 * NUM_FLAME_JOINTS + 3] = inv_affine_mats[0] * affine_mats[3];
        B_masked[0 * NUM_FLAME_JOINTS + 4] = inv_affine_mats[0] * affine_mats[4];
        B_masked[1 * NUM_FLAME_JOINTS + 1] = inv_affine_mats[1] * affine_mats[1];
        B_masked[1 * NUM_FLAME_JOINTS + 2] = inv_affine_mats[1] * affine_mats[2];
        B_masked[1 * NUM_FLAME_JOINTS + 3] = inv_affine_mats[1] * affine_mats[3];
        B_masked[1 * NUM_FLAME_JOINTS + 4] = inv_affine_mats[1] * affine_mats[4];
        B_masked[2 * NUM_FLAME_JOINTS + 2] = inv_affine_mats[2] * affine_mats[2];
        B_masked[3 * NUM_FLAME_JOINTS + 3] = inv_affine_mats[3] * affine_mats[3];
        B_masked[4 * NUM_FLAME_JOINTS + 4] = inv_affine_mats[4] * affine_mats[4];

        // A_r
        Ar[0] = rot_mats[0];
        Ar[1] = glm::mat3(affine_mats[0]) * rot_mats[1];
        Ar[2] = glm::mat3(affine_mats[1]) * rot_mats[2];
        Ar[3] = glm::mat3(affine_mats[1]) * rot_mats[3];
        Ar[4] = glm::mat3(affine_mats[1]) * rot_mats[4];

        // jacobian wrt pose (rot vec)
        glm::mat3 axis_0 = glm::mat3(affine_mats[0]) * right_jacobian[0];
        glm::mat3 axis_1 = glm::mat3(affine_mats[1]) * right_jacobian[1];

        // neck
        write_pose_effect<N_EXPR>(jacobian, 1, 0, joints[1], joints[0], axis_0);

        // jaw
        write_pose_effect<N_EXPR>(jacobian, 2, 0, joints[2], joints[0], axis_0);
        write_pose_effect<N_EXPR>(jacobian, 2, 1, joints[2], joints[1], axis_1);

        // eye
        write_pose_effect<N_EXPR>(jacobian, 3, 0, joints[3], joints[0], axis_0);
        write_pose_effect<N_EXPR>(jacobian, 3, 1, joints[3], joints[1], axis_1);

        // eye
        write_pose_effect<N_EXPR>(jacobian, 4, 0, joints[4], joints[0], axis_0);
        write_pose_effect<N_EXPR>(jacobian, 4, 1, joints[4], joints[1], axis_1);


        // jacobian wrt translation
        for (int j = 0; j < NUM_FLAME_JOINTS; j++) {
            int base = j * DIM_X * 3;
            jacobian[base + (N_EXPR + (3 * NUM_FLAME_JOINTS) + 0) * 3 + 0] = 1.0f;
            jacobian[base + (N_EXPR + (3 * NUM_FLAME_JOINTS) + 1) * 3 + 1] = 1.0f;
            jacobian[base + (N_EXPR + (3 * NUM_FLAME_JOINTS) + 2) * 3 + 2] = 1.0f;
        }
    }
}

template<int N_VERTS, int N_EXPR>
__global__ void FlameExpressionForwardWiJacobianCuda(
    const bool calc_jacobian,
    const int batch_size,
    const glm::vec3* __restrict__ v_canonical,  // [V, 3]
    const glm::vec3* __restrict__ j_canonical,

    const float* __restrict__ batched_x,
    const float* __restrict__ batched_pose_feat,
    const glm::mat4* __restrict__ batched_affine_mats,
    const glm::mat3* __restrict__ batched_right_jacobian,
    const glm::mat4* __restrict__ batched_B_masked,
    const glm::mat3* __restrict__ batched_Ar,

    const glm::vec3* __restrict__ expr_dirs,    // [N_EXPR, V, 3]
    const glm::vec3* __restrict__ pose_dirs,    // [36, V, 3]
    const float* __restrict__ lbs_weights,      // [V, J]

    glm::vec3* __restrict__ batched_v_output,            // [B, V, 3]
    float* __restrict__ batched_v_jacobian               // [B, V, DIM_X, 3]
) {
    constexpr int DIM_X = dim_x_for(N_EXPR);

    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int v_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size || v_idx >= N_VERTS) return;

    glm::vec3 vertex = v_canonical[v_idx];
    const float* expression = batched_x + batch_idx * DIM_X;
    const glm::vec3 translation = glm::vec3(
        batched_x[batch_idx * DIM_X + N_EXPR + NUM_FLAME_JOINTS * 3 + 0],
        batched_x[batch_idx * DIM_X + N_EXPR + NUM_FLAME_JOINTS * 3 + 1],
        batched_x[batch_idx * DIM_X + N_EXPR + NUM_FLAME_JOINTS * 3 + 2]
    );
    const float* pose_feat = batched_pose_feat + batch_idx * NUM_FLAME_POSEFEAT_BASIS;
    const glm::mat4* affine_mats = batched_affine_mats + batch_idx * NUM_FLAME_JOINTS;
    glm::vec3* v_output = batched_v_output + batch_idx * N_VERTS;

    for (int l = 0; l < N_EXPR; l++) {
        vertex += expression[l] * expr_dirs[l * N_VERTS + v_idx];
    }

    if (USE_POSE_BS) {
        for (int l = 0; l < NUM_FLAME_POSEFEAT_BASIS; l++) {
            vertex += pose_feat[l] * pose_dirs[l * N_VERTS + v_idx];
        }
    }

    float weights[NUM_FLAME_JOINTS];
    glm::vec4 v_deltas[NUM_FLAME_JOINTS];
    glm::vec3 v_skinned = glm::vec3(0.0f);
    glm::mat3 affine_skinned = glm::mat3(0.0f);

    for (int j = 0; j < NUM_FLAME_JOINTS; j++) {
        weights[j] = lbs_weights[v_idx * NUM_FLAME_JOINTS + j];
        v_deltas[j] = glm::vec4(vertex - j_canonical[j], 1.0f);
        glm::vec4 v_transformed = affine_mats[j] * v_deltas[j];
        v_skinned += weights[j] * glm::vec3(v_transformed);
        affine_skinned += weights[j] * glm::mat3(affine_mats[j]);
    }

    v_output[v_idx] = v_skinned + translation;

    if (calc_jacobian) {
        const glm::mat3* right_jacobian = batched_right_jacobian + batch_idx * NUM_FLAME_JOINTS;
        const glm::mat4* B_masked = batched_B_masked + batch_idx * NUM_FLAME_JOINTS * NUM_FLAME_JOINTS;
        const glm::mat3* Ar = batched_Ar + batch_idx * NUM_FLAME_JOINTS;
        float* jacobian = batched_v_jacobian + batch_idx * N_VERTS * DIM_X * 3;

        glm::vec3 p_sum[NUM_FLAME_JOINTS];
        p_sum[0] =  weights[0] * B_masked[0 * NUM_FLAME_JOINTS + 0] * v_deltas[0] +
                    weights[1] * B_masked[0 * NUM_FLAME_JOINTS + 1] * v_deltas[1] +
                    weights[2] * B_masked[0 * NUM_FLAME_JOINTS + 2] * v_deltas[2] +
                    weights[3] * B_masked[0 * NUM_FLAME_JOINTS + 3] * v_deltas[3] +
                    weights[4] * B_masked[0 * NUM_FLAME_JOINTS + 4] * v_deltas[4];
        p_sum[1] =  weights[1] * B_masked[1 * NUM_FLAME_JOINTS + 1] * v_deltas[1] +
                    weights[2] * B_masked[1 * NUM_FLAME_JOINTS + 2] * v_deltas[2] +
                    weights[3] * B_masked[1 * NUM_FLAME_JOINTS + 3] * v_deltas[3] +
                    weights[4] * B_masked[1 * NUM_FLAME_JOINTS + 4] * v_deltas[4];
        p_sum[2] =  weights[2] * B_masked[2 * NUM_FLAME_JOINTS + 2] * v_deltas[2];
        p_sum[3] =  weights[3] * B_masked[3 * NUM_FLAME_JOINTS + 3] * v_deltas[3];
        p_sum[4] =  weights[4] * B_masked[4 * NUM_FLAME_JOINTS + 4] * v_deltas[4];

        int j_offset = v_idx * DIM_X * 3;

        // jacobian wrt pose (rot vec)
        for (int j = 0; j < NUM_FLAME_JOINTS; j++) {
            glm::mat3 J_dtheta = -Ar[j] * skew_matrix(p_sum[j]);
            glm::mat3 J_domega = J_dtheta * right_jacobian[j];

            jacobian[j_offset + N_EXPR * 3 + j * 9 + 0] = J_domega[0][0];
            jacobian[j_offset + N_EXPR * 3 + j * 9 + 1] = J_domega[0][1];
            jacobian[j_offset + N_EXPR * 3 + j * 9 + 2] = J_domega[0][2];
            jacobian[j_offset + N_EXPR * 3 + j * 9 + 3] = J_domega[1][0];
            jacobian[j_offset + N_EXPR * 3 + j * 9 + 4] = J_domega[1][1];
            jacobian[j_offset + N_EXPR * 3 + j * 9 + 5] = J_domega[1][2];
            jacobian[j_offset + N_EXPR * 3 + j * 9 + 6] = J_domega[2][0];
            jacobian[j_offset + N_EXPR * 3 + j * 9 + 7] = J_domega[2][1];
            jacobian[j_offset + N_EXPR * 3 + j * 9 + 8] = J_domega[2][2];
        }

        // jacobian wrt expression
        for (int l = 0; l < N_EXPR; l++) {
            glm::vec3 J_dexp = affine_skinned * expr_dirs[l * N_VERTS + v_idx]; // TODO
            jacobian[j_offset + l * 3 + 0] = J_dexp.x;
            jacobian[j_offset + l * 3 + 1] = J_dexp.y;
            jacobian[j_offset + l * 3 + 2] = J_dexp.z;
        }

        // jacobian wrt translation
        jacobian[j_offset + (N_EXPR + (3 * NUM_FLAME_JOINTS) + 0) * 3 + 0] = 1.0f;
        jacobian[j_offset + (N_EXPR + (3 * NUM_FLAME_JOINTS) + 1) * 3 + 1] = 1.0f;
        jacobian[j_offset + (N_EXPR + (3 * NUM_FLAME_JOINTS) + 2) * 3 + 2] = 1.0f;
    }
}

}  // namespace ExpressionJacobian


// ---------- host-side templated launchers + dispatch ----------

template<int N_VERTS, int N_EXPR>
static void launch_flame_expression_jacobian_impl(
    const int batch_size,
    const float* batched_x,
    const glm::vec3* v_canonical,
    const glm::vec3* j_canonical,
    const glm::vec3* expr_dirs,
    const glm::vec3* pose_dirs,
    const float* lbs_weights,
    float* batched_pose_feat,
    glm::mat4* batched_affine_mats,
    glm::mat3* batched_right_jacobian,
    glm::mat4* batched_B_masked,
    glm::mat3* batched_Ar,
    glm::vec3* batched_v_output,
    glm::vec3* batched_j_output,
    float* batched_v_jacobian,
    float* batched_j_jacobian
) {
    dim3 block_a(256);
    dim3 grid_a((batch_size + block_a.x - 1) / block_a.x);
    ExpressionJacobian::SkeletonTreeCuda<N_EXPR><<<grid_a, block_a>>>(
        true,
        batch_size, j_canonical, batched_x,
        batched_pose_feat, batched_affine_mats,
        batched_right_jacobian, batched_B_masked, batched_Ar,
        batched_j_output, batched_j_jacobian
    );

    dim3 block_b(1, 256);
    dim3 grid_b(batch_size, (N_VERTS + block_b.y - 1) / block_b.y);
    ExpressionJacobian::FlameExpressionForwardWiJacobianCuda<N_VERTS, N_EXPR><<<grid_b, block_b>>>(
        true,
        batch_size, v_canonical, j_canonical, batched_x,
        batched_pose_feat, batched_affine_mats, batched_right_jacobian, batched_B_masked, batched_Ar,
        expr_dirs, pose_dirs, lbs_weights,
        batched_v_output, batched_v_jacobian
    );
}


template<int N_VERTS, int N_EXPR>
static void launch_flame_expression_forward_impl(
    const int batch_size,
    const float* batched_x,
    const glm::vec3* v_canonical,
    const glm::vec3* j_canonical,
    const glm::vec3* expr_dirs,
    const glm::vec3* pose_dirs,
    const float* lbs_weights,
    float* batched_pose_feat,
    glm::mat4* batched_affine_mats,
    glm::vec3* batched_v_output,
    glm::vec3* batched_j_output
) {
    dim3 block_a(256);
    dim3 grid_a((batch_size + block_a.x - 1) / block_a.x);
    ExpressionJacobian::SkeletonTreeCuda<N_EXPR><<<grid_a, block_a>>>(
        false,
        batch_size, j_canonical, batched_x,
        batched_pose_feat, batched_affine_mats,
        nullptr, nullptr, nullptr,
        batched_j_output, nullptr
    );

    dim3 block_b(1, 256);
    dim3 grid_b(batch_size, (N_VERTS + block_b.y - 1) / block_b.y);
    ExpressionJacobian::FlameExpressionForwardWiJacobianCuda<N_VERTS, N_EXPR><<<grid_b, block_b>>>(
        false,
        batch_size, v_canonical, j_canonical, batched_x,
        batched_pose_feat, batched_affine_mats, nullptr, nullptr, nullptr,
        expr_dirs, pose_dirs, lbs_weights,
        batched_v_output, nullptr
    );
}

template<int N_VERTS>
static void dispatch_flame_expression_jacobian_by_expr(
    const int num_expressions,
    const int batch_size,
    const float* batched_x,
    const glm::vec3* v_canonical,
    const glm::vec3* j_canonical,
    const glm::vec3* expr_dirs,
    const glm::vec3* pose_dirs,
    const float* lbs_weights,
    float* batched_pose_feat,
    glm::mat4* batched_affine_mats,
    glm::mat3* batched_right_jacobian,
    glm::mat4* batched_B_masked,
    glm::mat3* batched_Ar,
    glm::vec3* batched_v_output,
    glm::vec3* batched_j_output,
    float* batched_v_jacobian,
    float* batched_j_jacobian
) {
    switch (num_expressions) {
#define DISPATCH_CASE(N) \
        case N: \
            launch_flame_expression_jacobian_impl<N_VERTS, N>( \
                batch_size, batched_x, v_canonical, j_canonical, \
                expr_dirs, pose_dirs, lbs_weights, \
                batched_pose_feat, batched_affine_mats, \
                batched_right_jacobian, batched_B_masked, batched_Ar, \
                batched_v_output, batched_j_output, \
                batched_v_jacobian, batched_j_jacobian \
            ); \
            break;
        FLAME_FOREACH_NEXPR(DISPATCH_CASE)
#undef DISPATCH_CASE
        default:
            throw std::runtime_error("flame_expression_jacobian: unsupported num_expressions");
    }
}

template<int N_VERTS>
static void dispatch_flame_expression_forward_by_expr(
    const int num_expressions,
    const int batch_size,
    const float* batched_x,
    const glm::vec3* v_canonical,
    const glm::vec3* j_canonical,
    const glm::vec3* expr_dirs,
    const glm::vec3* pose_dirs,
    const float* lbs_weights,
    float* batched_pose_feat,
    glm::mat4* batched_affine_mats,
    glm::vec3* batched_v_output,
    glm::vec3* batched_j_output
) {
    switch (num_expressions) {
#define DISPATCH_CASE(N) \
        case N: \
            launch_flame_expression_forward_impl<N_VERTS, N>( \
                batch_size, batched_x, v_canonical, j_canonical, \
                expr_dirs, pose_dirs, lbs_weights, \
                batched_pose_feat, batched_affine_mats, \
                batched_v_output, batched_j_output \
            ); \
            break;
        FLAME_FOREACH_NEXPR(DISPATCH_CASE)
#undef DISPATCH_CASE
        default:
            throw std::runtime_error("flame_expression_forward: unsupported num_expressions");
    }
}


void FlameTracking::flame_expression_jacobian(
    const int num_vertices,
    const int num_expressions,
    const int batch_size,
    const float* batched_x,
    const glm::vec3* v_canonical,
    const glm::vec3* j_canonical,

    const glm::vec3* expr_dirs,         // [B, V, 3]
    const glm::vec3* pose_dirs,         // [36, V, 3]
    const float* lbs_weights,           // [V, J]

    float* batched_pose_feat,
    glm::mat4* batched_affine_mats,
    glm::mat3* batched_right_jacobian,
    glm::mat4* batched_B_masked,
    glm::mat3* batched_Ar,

    glm::vec3* batched_v_output,
    glm::vec3* batched_j_output,
    float* batched_v_jacobian,
    float* batched_j_jacobian
) {
#define DISPATCH_VERTS_CASE(N_VERTS) \
        case N_VERTS: \
            dispatch_flame_expression_jacobian_by_expr<N_VERTS>( \
                num_expressions, batch_size, batched_x, v_canonical, j_canonical, \
                expr_dirs, pose_dirs, lbs_weights, batched_pose_feat, batched_affine_mats, \
                batched_right_jacobian, batched_B_masked, batched_Ar, \
                batched_v_output, batched_j_output, batched_v_jacobian, batched_j_jacobian \
            ); \
            break;
    switch (num_vertices) {
        FLAME_FOREACH_NVERTS(DISPATCH_VERTS_CASE)
        default:
            throw std::runtime_error("flame_expression_jacobian: unsupported num_vertices");
    }
#undef DISPATCH_VERTS_CASE
}


void FlameTracking::flame_expression_forward(
    const int num_vertices,
    const int num_expressions,
    const int batch_size,
    const float* batched_x,
    const glm::vec3* v_canonical,
    const glm::vec3* j_canonical,

    const glm::vec3* expr_dirs,         // [B, V, 3]
    const glm::vec3* pose_dirs,         // [36, V, 3]
    const float* lbs_weights,           // [V, J]

    float* batched_pose_feat,
    glm::mat4* batched_affine_mats,

    glm::vec3* batched_v_output,
    glm::vec3* batched_j_output
) {
#define DISPATCH_VERTS_CASE(N_VERTS) \
        case N_VERTS: \
            dispatch_flame_expression_forward_by_expr<N_VERTS>( \
                num_expressions, batch_size, batched_x, v_canonical, j_canonical, \
                expr_dirs, pose_dirs, lbs_weights, batched_pose_feat, batched_affine_mats, \
                batched_v_output, batched_j_output \
            ); \
            break;
    switch (num_vertices) {
        FLAME_FOREACH_NVERTS(DISPATCH_VERTS_CASE)
        default:
            throw std::runtime_error("flame_expression_forward: unsupported num_vertices");
    }
#undef DISPATCH_VERTS_CASE
}
