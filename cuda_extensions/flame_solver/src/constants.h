#pragma once

constexpr int WARP_SIZE = 32;

constexpr int NUM_FLAME_VERTICES_BASE = 5023;
constexpr int NUM_FLAME_VERTICES_ORAL = 5768;
constexpr int NUM_FLAME_VERTICES = NUM_FLAME_VERTICES_BASE;
constexpr int NUM_FLAME_JOINTS = 5;
constexpr int NUM_FLAME_IDENTITY_BASIS = 300;
constexpr int NUM_FLAME_POSEFEAT_BASIS = (NUM_FLAME_JOINTS - 1) * 9; // 36

constexpr bool USE_POSE_BS = false;

// Layout of x: [expression(N) | pose(NUM_FLAME_JOINTS * 3) | translation(3)]
constexpr int dim_x_for(int n_expr) { return n_expr + NUM_FLAME_JOINTS * 3 + 3; }

// Single source of truth for the supported expression basis sizes.
// Adding a new size only requires extending this macro; the dispatchers in
// the .cu files and binding.cpp pick it up automatically.
#define FLAME_FOREACH_NEXPR(MACRO) \
    MACRO(50)  \
    MACRO(100)

#define FLAME_FOREACH_NVERTS(MACRO) \
    MACRO(NUM_FLAME_VERTICES_BASE)  \
    MACRO(NUM_FLAME_VERTICES_ORAL)
