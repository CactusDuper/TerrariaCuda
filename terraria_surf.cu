#include <cmath>
#include <cstdint>
#include <chrono>
#include <vector>
#include <cstdio>
#include <fstream>

enum class EDungeonSide {
    Left,
    Right
};
enum class TerrainFeatureType {
    Plateau,
    Hill,
    Dale,
    Mountain,
    Valley
};

enum class EWorldSize {
    Small,
    Medium,
    Large,
    INVALID
};

constexpr int SMALL_X = 4200;
constexpr int SMALL_Y = 1200;
constexpr int MEDIUM_X = 6400;
constexpr int MEDIUM_Y = 1800;
constexpr int LARGE_X = 8400;
constexpr int LARGE_Y = 2400;
static constexpr int SPAWN_X_OFFSET_CONST = 40;
static constexpr int SPAWN_REGION_WIDTH_CONST = (2 * SPAWN_X_OFFSET_CONST + 1);

template <EWorldSize SIZE>
constexpr int getMaxX() {
    if constexpr (SIZE == EWorldSize::Small) {
        return SMALL_X;
    } else if constexpr (SIZE == EWorldSize::Medium) {
        return MEDIUM_X;
    } else {
        return LARGE_X;
    }
}

template <EWorldSize SIZE>
constexpr int getMaxY() {
    if constexpr (SIZE == EWorldSize::Small) {
        return SMALL_Y;
    } else if constexpr (SIZE == EWorldSize::Medium) {
        return MEDIUM_Y;
    } else {
        return LARGE_Y;
    }
}

template <EWorldSize SIZE>
struct WorldDimParams {
    static constexpr int MaxTilesX = getMaxX<SIZE>();
    static constexpr int MaxTilesY = getMaxY<SIZE>();
};

static constexpr int MSEED = 161803398;

struct DeviceUnifiedRandomState {
    int32_t inext;
    int32_t inextp;
    int32_t SeedArray[56];
};

__device__ inline int32_t unified_random_internal_sample(DeviceUnifiedRandomState *state) {
    int32_t retVal;
    int32_t locINext = state->inext;
    int32_t locINextp = state->inextp;

    if (++locINext >= 56) locINext = 1;
    if (++locINextp >= 56) locINextp = 1;

    retVal = state->SeedArray[locINext] - state->SeedArray[locINextp];

    if (retVal == INT_MAX) retVal--;
    if (retVal < 0) retVal += INT_MAX;

    state->SeedArray[locINext] = retVal;
    state->inext = locINext;
    state->inextp = locINextp;

    return retVal;
}

__device__ void unified_random_init_seed(DeviceUnifiedRandomState *state, int32_t seed) {
    int32_t ii;
    int32_t mj, mk;

    int32_t subtraction = (seed == INT_MIN) ? INT_MAX : abs(seed);

    mj = MSEED - subtraction;
    state->SeedArray[55] = mj;
    mk = 1;

    for (int8_t i = 1; i < 55; i++) {
        ii = (21 * i) % 55;
        state->SeedArray[ii] = mk;
        mk = mj - mk;
        if (mk < 0) mk += INT_MAX;
        mj = state->SeedArray[ii];
    }

    for (int k = 1; k < 5; k++) {
        for (int i = 1; i < 56; i++) {
            state->SeedArray[i] -= state->SeedArray[1 + ((i + 30) % 55)];
            if (state->SeedArray[i] < 0) {
                state->SeedArray[i] += INT_MAX;
            }
        }
    }

    state->inext = 0;
    state->inextp = 21;
}

__device__ inline double unified_random_sample(DeviceUnifiedRandomState *state) {
    return static_cast<double>(unified_random_internal_sample(state)) * 4.656612875245797E-10;
}

__device__ inline int32_t unified_random_next(DeviceUnifiedRandomState *state) {
    return unified_random_internal_sample(state);
}

__device__ inline int32_t unified_random_next_max(DeviceUnifiedRandomState *state, int32_t maxValue) {
    return static_cast<int32_t>((unified_random_sample(state) * static_cast<double>(maxValue)));
}

__device__ inline int32_t unified_random_next_range(DeviceUnifiedRandomState *state, int32_t minValue, int32_t maxValue) {
    int64_t range = static_cast<int64_t>(maxValue) - static_cast<int64_t>(minValue);
    return static_cast<int32_t>(unified_random_sample(state) * static_cast<double>(range)) + minValue;
}

template <EWorldSize SIZE>
struct DeviceWorldState {
    using Dim = WorldDimParams<SIZE>;
    DeviceUnifiedRandomState rng_state;
    int16_t effective_world_surface_for_spawn_check;
    int16_t leftBeachEnd;
    int16_t rightBeachStart;
    int16_t spawn_region_surface_y[SPAWN_REGION_WIDTH_CONST];
    double world_surface_high_calculated;
};

template <EWorldSize SIZE>
__device__ void RandomizeTreeStyle_Device(DeviceWorldState<SIZE> &ws) {
    int32_t treeStyle[4];

    if constexpr (SIZE == EWorldSize::Small) {
        unified_random_sample(&ws.rng_state);
        treeStyle[0] = unified_random_next_max(&ws.rng_state, 6);
        treeStyle[1] = unified_random_next_max(&ws.rng_state, 6);
        while (treeStyle[1] == treeStyle[0]) {
            treeStyle[1] = unified_random_next_max(&ws.rng_state, 6);
        }

        for (int i = 0; i < 2; i++) {
            if (treeStyle[i] == 0 && unified_random_next_max(&ws.rng_state, 3) != 0) {
                treeStyle[i] = 4;
            }
        }
        return;
    }

    if constexpr (SIZE == EWorldSize::Medium) {
        unified_random_sample(&ws.rng_state);
        unified_random_sample(&ws.rng_state);

        treeStyle[0] = unified_random_next_max(&ws.rng_state, 6);
        treeStyle[1] = unified_random_next_max(&ws.rng_state, 6);
        treeStyle[2] = unified_random_next_max(&ws.rng_state, 6);
        while (treeStyle[1] == treeStyle[0]) {
            treeStyle[1] = unified_random_next_max(&ws.rng_state, 6);
        }
        while (treeStyle[2] == treeStyle[0] || treeStyle[2] == treeStyle[1]) {
            treeStyle[2] = unified_random_next_max(&ws.rng_state, 6);
        }

        for (int j = 0; j < 3; j++) {
            if (treeStyle[j] == 0 && unified_random_next_max(&ws.rng_state, 3) != 0) {
                treeStyle[j] = 4;
            }
        }
        return;
    }

    // Large
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);

    treeStyle[0] = unified_random_next_max(&ws.rng_state, 6);
    treeStyle[1] = unified_random_next_max(&ws.rng_state, 6);
    treeStyle[2] = unified_random_next_max(&ws.rng_state, 6);
    treeStyle[3] = unified_random_next_max(&ws.rng_state, 6);

    while (treeStyle[1] == treeStyle[0]) {
        treeStyle[1] = unified_random_next_max(&ws.rng_state, 6);
    }
    while (treeStyle[2] == treeStyle[0] || treeStyle[2] == treeStyle[1]) {
        treeStyle[2] = unified_random_next_max(&ws.rng_state, 6);
    }
    while (treeStyle[3] == treeStyle[0] || treeStyle[3] == treeStyle[1] || treeStyle[3] == treeStyle[2]) {
        treeStyle[3] = unified_random_next_max(&ws.rng_state, 6);
    }

    for (int k_idx = 0; k_idx < 4; ++k_idx) {
        if (treeStyle[k_idx] == 0 && unified_random_next_max(&ws.rng_state, 3) != 0) {
            treeStyle[k_idx] = 4;
        }
    }
}

template <EWorldSize SIZE>
__device__ void RandomizeCaveBackgrounds_Device(DeviceWorldState<SIZE> &ws) {
    int32_t caveStyle[4];

    const int maxValue = 8;
    if constexpr (SIZE == EWorldSize::Small) {
        unified_random_sample(&ws.rng_state);
        caveStyle[0] = unified_random_next_max(&ws.rng_state, maxValue);
        caveStyle[1] = unified_random_next_max(&ws.rng_state, maxValue);
        while (caveStyle[1] == caveStyle[0]) {
            caveStyle[1] = unified_random_next_max(&ws.rng_state, maxValue);
        }
    }
    else if constexpr (SIZE == EWorldSize::Medium) {
        unified_random_sample(&ws.rng_state);
        unified_random_sample(&ws.rng_state);
        caveStyle[0] = unified_random_next_max(&ws.rng_state, maxValue);
        caveStyle[1] = unified_random_next_max(&ws.rng_state, maxValue);
        caveStyle[2] = unified_random_next_max(&ws.rng_state, maxValue);
        while (caveStyle[1] == caveStyle[0]) {
            caveStyle[1] = unified_random_next_max(&ws.rng_state, maxValue);
        }
        while (caveStyle[2] == caveStyle[0] || caveStyle[2] == caveStyle[1]) {
            caveStyle[2] = unified_random_next_max(&ws.rng_state, maxValue);
        }
    }
    else { // Large
        unified_random_sample(&ws.rng_state);
        unified_random_sample(&ws.rng_state);
        unified_random_sample(&ws.rng_state);
        caveStyle[0] = unified_random_next_max(&ws.rng_state, maxValue);
        caveStyle[1] = unified_random_next_max(&ws.rng_state, maxValue);
        caveStyle[2] = unified_random_next_max(&ws.rng_state, maxValue);
        caveStyle[3] = unified_random_next_max(&ws.rng_state, maxValue);
        while (caveStyle[1] == caveStyle[0]) {
            caveStyle[1] = unified_random_next_max(&ws.rng_state, maxValue);
        }
        while (caveStyle[2] == caveStyle[0] || caveStyle[2] == caveStyle[1]) {
            caveStyle[2] = unified_random_next_max(&ws.rng_state, maxValue);
        }
        while (caveStyle[3] == caveStyle[0] || caveStyle[3] == caveStyle[1] || caveStyle[3] == caveStyle[2]) {
            caveStyle[3] = unified_random_next_max(&ws.rng_state, maxValue);
        }
    }
}

__device__ int RollRandomForestBGStyle_Device(DeviceUnifiedRandomState *rng_state) {
    int num = unified_random_next_max(rng_state, 11);
    if ((num == 1 || num == 2) && unified_random_next_max(rng_state, 2) == 0) {
        num = unified_random_next_max(rng_state, 11);
    }
    if (num == 0) {
        num = unified_random_next_max(rng_state, 11);
    }
    if (num == 3 && unified_random_next_max(rng_state, 3) == 0) {
        num = 31;
    }
    if (num == 5 && unified_random_next_max(rng_state, 2) == 0) {
        num = 51;
    }
    if (num == 7 && unified_random_next_max(rng_state, 4) == 0) {
        num = unified_random_next_range(rng_state, 71, 74);
    }
    return num;
}

__device__ void RandomizeBackgrounds_Device(DeviceUnifiedRandomState *rng_state) {
    int tree1 = RollRandomForestBGStyle_Device(rng_state);
    int tree2;
    for (tree2 = RollRandomForestBGStyle_Device(rng_state); tree2 == tree1; tree2 = RollRandomForestBGStyle_Device(rng_state)) { }

    int tree3 = RollRandomForestBGStyle_Device(rng_state);
    while (tree3 == tree1 || tree3 == tree2) {
        tree3 = RollRandomForestBGStyle_Device(rng_state);
    }

    int tree4 = RollRandomForestBGStyle_Device(rng_state);
    while (tree4 == tree1 || tree4 == tree2 || tree4 == tree3) {
        tree4 = RollRandomForestBGStyle_Device(rng_state);
    }

    unified_random_sample(rng_state);
    unified_random_sample(rng_state);

    int snow = unified_random_next_max(rng_state, 8);
    if (snow == 2 && unified_random_next_max(rng_state, 2) == 0) {
        unified_random_sample(rng_state);
    }
    else if (snow == 3 && unified_random_next_max(rng_state, 2) == 0) {
        unified_random_sample(rng_state);
    }
    else if (snow == 4 && unified_random_next_max(rng_state, 2) == 0) {
        unified_random_sample(rng_state);
    }
}

__device__ void RandomizeWeather_Device(DeviceUnifiedRandomState *rng_state) {
    unified_random_sample(rng_state);
    float wind = 0.0f;

    while (wind == 0.0f) {
        wind = (float)unified_random_next_range(rng_state, -400, 401) * 0.001f;
    }
}

template <EWorldSize SIZE>
__device__ void ResetGenerationPass_Device(DeviceWorldState<SIZE> &ws) {
    constexpr int maxX = WorldDimParams<SIZE>::MaxTilesX;
    unified_random_sample(&ws.rng_state);
    RandomizeWeather_Device(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    RandomizeTreeStyle_Device(ws);
    RandomizeCaveBackgrounds_Device(ws);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    RandomizeBackgrounds_Device(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);

    EDungeonSide dungeonSide = (unified_random_next_max(&ws.rng_state, 2) != 0) ? EDungeonSide::Right : EDungeonSide::Left;

    unified_random_sample(&ws.rng_state);

    int tempVal = unified_random_next_max(&ws.rng_state, maxX);
    if (dungeonSide == EDungeonSide::Right) {
        while ((double)tempVal < (double)maxX * 0.6 || (double)tempVal > (double)maxX * 0.75) {
            tempVal = unified_random_next_max(&ws.rng_state, maxX);
        }
    }
    else { // Left
        while ((double)tempVal < (double)maxX * 0.25 || (double)tempVal > (double)maxX * 0.4) {
            tempVal = unified_random_next_max(&ws.rng_state, maxX);
        }
    }
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);

    ws.leftBeachEnd = unified_random_next_range(&ws.rng_state, 300, 340);
    if (dungeonSide == EDungeonSide::Right) {
        ws.leftBeachEnd += 40;
    }
    else { // Left
        ws.leftBeachEnd += 20;
    }

    ws.rightBeachStart = maxX - unified_random_next_range(&ws.rng_state, 300, 340);
    if (dungeonSide == EDungeonSide::Left) {
        ws.rightBeachStart -= 40;
    }
    else { // Right
        ws.rightBeachStart -= 20;
    }
    unified_random_sample(&ws.rng_state);
}

__device__ double GenerateWorldSurfaceOffset_Device(TerrainFeatureType featureType, DeviceUnifiedRandomState *rng_state) {
    double num = 0.0;
    switch (featureType) {
    case TerrainFeatureType::Plateau:
        while (unified_random_next_range(rng_state, 0, 7) == 0) {
            num += (double)unified_random_next_range(rng_state, -1, 2);
        }
        break;
    case TerrainFeatureType::Hill:
        while (unified_random_next_range(rng_state, 0, 4) == 0) {
            num -= 1.0;
        }
        while (unified_random_next_range(rng_state, 0, 10) == 0) {
            num += 1.0;
        }
        break;
    case TerrainFeatureType::Dale:
        while (unified_random_next_range(rng_state, 0, 4) == 0) {
            num += 1.0;
        }
        while (unified_random_next_range(rng_state, 0, 10) == 0) {
            num -= 1.0;
        }
        break;
    case TerrainFeatureType::Mountain:
        while (unified_random_next_range(rng_state, 0, 2) == 0) {
            num -= 1.0;
        }
        while (unified_random_next_range(rng_state, 0, 6) == 0) {
            num += 1.0;
        }
        break;
    case TerrainFeatureType::Valley:
        while (unified_random_next_range(rng_state, 0, 2) == 0) {
            num += 1.0;
        }
        while (unified_random_next_range(rng_state, 0, 5) == 0) {
            num -= 1.0;
        }
        break;
    }

    return num;
}

template <EWorldSize SIZE>
__device__ double EstimateSurfaceHighRoughly_Device(DeviceWorldState<SIZE> &ws) {
    constexpr int maxX = WorldDimParams<SIZE>::MaxTilesX;
    constexpr int maxY = WorldDimParams<SIZE>::MaxTilesY;

    for (int i = 0; i < SPAWN_REGION_WIDTH_CONST; i++) {
        ws.spawn_region_surface_y[i] = -1;
    }

    const int num_const = 5;
    const int world_center_x = maxX / 2;
    const int spawn_region_min_x_world = world_center_x - SPAWN_X_OFFSET_CONST;
    const int spawn_region_max_x_world = world_center_x + SPAWN_X_OFFSET_CONST;

    TerrainFeatureType terrainFeatureType = TerrainFeatureType::Plateau;
    int num2_countdown = 0;
    double surface = (double)maxY * 0.3;
    surface *= (double)unified_random_next_range(&ws.rng_state, 90, 110) * 0.005;
    double rock = surface + (double)maxY * 0.2;
    rock *= (double)unified_random_next_range(&ws.rng_state, 90, 110) * 0.01;

    double surfaceHigh = surface;
    double rockLow = rock;

    const double num9_upper_sand_limit = (double)maxY * 0.23;
    num2_countdown = ws.leftBeachEnd + 5;
    for (int i = 0; i < maxX; i++) {
        surfaceHigh = fmax(surface, surfaceHigh);
        rockLow = fmin(rock, rockLow);

        if (num2_countdown <= 0) {
            terrainFeatureType = (TerrainFeatureType)unified_random_next_range(&ws.rng_state, 0, 5);
            num2_countdown = unified_random_next_range(&ws.rng_state, 5, 40);
            if (terrainFeatureType == TerrainFeatureType::Plateau) {
                num2_countdown *= (int)((double)unified_random_next_range(&ws.rng_state, 5, 30) * 0.2);
            }
        }
        num2_countdown--;

        if ((double)i > (double)maxX * 0.45 && (double)i < (double)maxX * 0.55 &&
            (terrainFeatureType == TerrainFeatureType::Mountain || terrainFeatureType == TerrainFeatureType::Valley)) {
            terrainFeatureType = (TerrainFeatureType)unified_random_next_max(&ws.rng_state, 3);
        }
        if ((double)i > (double)maxX * 0.48 && (double)i < (double)maxX * 0.52) {
            terrainFeatureType = TerrainFeatureType::Plateau;
        }

        surface += GenerateWorldSurfaceOffset_Device(terrainFeatureType, &ws.rng_state);
        const double num10_lower_limit = 0.17;
        const double num11_upper_limit = 0.26;

        if (i < ws.leftBeachEnd + num_const || i > ws.rightBeachStart - num_const) {
            surface = fmin(fmax(surface, (double)maxY * 0.17), num9_upper_sand_limit); // clamp
        }
        else if (surface < (double)maxY * num10_lower_limit) {
            surface = (double)maxY * num10_lower_limit;
            num2_countdown = 0;
        }
        else if (surface > (double)maxY * num11_upper_limit) {
            surface = (double)maxY * num11_upper_limit;
            num2_countdown = 0;
        }

        while (unified_random_next_max(&ws.rng_state, 3) == 0) {
            rock += (double)unified_random_next_range(&ws.rng_state, -2, 3);
        }
        if (rock < surface + (double)maxY * 0.06){
            rock += 1.0;
        }
        if (rock > surface + (double)maxY * 0.35) {
            rock -= 1.0;
        }
        if (i >= spawn_region_min_x_world && i <= spawn_region_max_x_world) {
            int x_relative_to_spawn_region = i - spawn_region_min_x_world;
            ws.spawn_region_surface_y[x_relative_to_spawn_region] = static_cast<int16_t>(surface);
        }

        if (i == ws.rightBeachStart - num_const) {
            terrainFeatureType = TerrainFeatureType::Plateau;
            num2_countdown = maxX - i;
        }
    }

    ws.effective_world_surface_for_spawn_check = (int)(surfaceHigh + 25.0);

    unified_random_sample(&ws.rng_state);
    unified_random_sample(&ws.rng_state);
    const int num14_rock_surface_gap = 20;
    if (rockLow < surfaceHigh + (double)num14_rock_surface_gap) {
        const double num15_mid = (rockLow + surfaceHigh) / 2.0;
        double num16_abs_diff = fabs(rockLow - surfaceHigh);
        if (num16_abs_diff < (double)num14_rock_surface_gap) {
            num16_abs_diff = num14_rock_surface_gap;
        }
        surfaceHigh = num15_mid - num16_abs_diff / 2.0;
    }
    return surfaceHigh;
}

template <EWorldSize SIZE>
__device__ int EstimateSpawnPosSet_Device(DeviceWorldState<SIZE> &ws) {
    constexpr int maxX = WorldDimParams<SIZE>::MaxTilesX;
    int spawnRange = 5;
    bool invalidSpawnPoint = true;
    const int world_center_x = maxX / 2;
    const int spawn_region_min_x_world = world_center_x - SPAWN_X_OFFSET_CONST;

    int spawnY = 0;

    while (invalidSpawnPoint) {
        const int x_world = world_center_x + unified_random_next_range(&ws.rng_state, -spawnRange, spawnRange + 1);
        int x_relative_to_spawn_region = x_world - spawn_region_min_x_world;

        if (x_relative_to_spawn_region >= 0 && x_relative_to_spawn_region < SPAWN_REGION_WIDTH_CONST) {
            int16_t surface_y_for_this_x = ws.spawn_region_surface_y[x_relative_to_spawn_region];
            if (surface_y_for_this_x != -1) {
                spawnY = static_cast<int>(surface_y_for_this_x);
            }
        }
        invalidSpawnPoint = false;
        spawnRange++;
        if ((double)spawnY > ws.effective_world_surface_for_spawn_check) {
            invalidSpawnPoint = true;
        }
    }

    int spawnRange2 = 10;
    while ((double)spawnY > ws.effective_world_surface_for_spawn_check) {
        const int x_world = unified_random_next_range(&ws.rng_state, world_center_x - spawnRange2, world_center_x + spawnRange2 + 1);
        int x_relative_to_spawn_region = x_world - spawn_region_min_x_world;

        if (x_relative_to_spawn_region >= 0 && x_relative_to_spawn_region < SPAWN_REGION_WIDTH_CONST) {
            int16_t surface_y_for_this_x = ws.spawn_region_surface_y[x_relative_to_spawn_region];
            if (surface_y_for_this_x != -1) {
                spawnY = static_cast<int>(surface_y_for_this_x);
            }
        }
        spawnRange2++;
        if (spawnRange2 > maxX / 2){
            break;
        }
    }
    return spawnY;
}

template <EWorldSize SIZE>
__global__ void SimplifiedWorldGenCheck_Kernel(int32_t *seeds, DeviceWorldState<SIZE> *d_world_states, bool *d_results, int num_total_seeds) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_total_seeds) {
        return;
    }

    int32_t current_seed = seeds[tid];
    DeviceWorldState<SIZE> &current_ws = d_world_states[tid]; // Use pre-allocated state

    unified_random_init_seed(&current_ws.rng_state, current_seed);

    ResetGenerationPass_Device(current_ws);

    unified_random_init_seed(&current_ws.rng_state, current_seed);
    current_ws.world_surface_high_calculated = EstimateSurfaceHighRoughly_Device(current_ws);

    unified_random_init_seed(&current_ws.rng_state, current_seed);
    int est_spawn_y = EstimateSpawnPosSet_Device(current_ws);

    // Check
    const double SURFACE_SPAWN_DELTA = 30.0;
    if ((double)est_spawn_y - current_ws.world_surface_high_calculated < SURFACE_SPAWN_DELTA) {
        d_results[tid] = false; // Failed
    }
    else {
        d_results[tid] = true; // Passed
        // Can log here but it's super slow
        //printf("Seed: %d, SpawnY: %d, Surf: %lf, Delta: %lf\n", current_seed, est_spawn_y, current_ws.world_surface_high_calculated, ((double)est_spawn_y - current_ws.world_surface_high_calculated));
    }
}

int main(int argc, char **argv) {
    std::vector<int32_t> passedSeeds;
    long long seeds_to_check_per_size_param = static_cast<long long>(INT_MAX) + 1;

    size_t per_seed_state_approx_size = sizeof(DeviceWorldState<EWorldSize::Large>); // All are the same size
    size_t total_gpu_mem_bytes;
    size_t free_gpu_mem_bytes;
    cudaError_t cuda_err = cudaMemGetInfo(&free_gpu_mem_bytes, &total_gpu_mem_bytes);

    int BATCH_SIZE = 65536 * 16;
    if (cuda_err == cudaSuccess && per_seed_state_approx_size > 0)
    {
        // size_t usable_gpu_mem = static_cast<size_t>(free_gpu_mem_bytes * 0.9);
        // BATCH_SIZE = usable_gpu_mem / per_seed_state_approx_size;
        // if (BATCH_SIZE == 0) BATCH_SIZE = 1;
        // if (BATCH_SIZE > 65536) BATCH_SIZE = 65536;
    }
    else
    {
        if (cuda_err != cudaSuccess)
            printf("cudaMemGetInfo failed: %s. ", cudaGetErrorString(cuda_err));
        printf("Using default batch size: %d\n", BATCH_SIZE);
    }
    printf("Calculated GPU Batch Size: %d (Per-seed state est.: %zu bytes)\n", BATCH_SIZE, per_seed_state_approx_size);

    long long total_seeds_processed_count = 0;
    long long total_passed_check_count = 0;
    auto overall_start_time = std::chrono::high_resolution_clock::now();

    std::ofstream outfile("passed_seeds.txt", std::ios_base::app);
    if (!outfile.is_open())
    {
        printf("Error: Could not open passed_seeds.txt for writing.\n");
        // Cleanup and exit or proceed without writing to file
    }

    int32_t *seeds_batch;
    void *d_world_states_batch_generic = nullptr; // void* for generic handling
    bool *d_results_batch;
    bool *h_results_batch = new bool[BATCH_SIZE];
    std::vector<int32_t> seeds(BATCH_SIZE);

    cuda_err = cudaMalloc(&seeds_batch, BATCH_SIZE * sizeof(int32_t));
    if (cuda_err != cudaSuccess)
    {
        printf("cudaMalloc seeds_batch failed: %s\n", cudaGetErrorString(cuda_err));
        delete[] h_results_batch;
        return 1;
    }
    cuda_err = cudaMalloc(&d_results_batch, BATCH_SIZE * sizeof(bool));
    if (cuda_err != cudaSuccess)
    {
        printf("cudaMalloc d_results_batch failed: %s\n", cudaGetErrorString(cuda_err));
        cudaFree(seeds_batch);
        delete[] h_results_batch;
        return 1;
    }

    long long grand_total_seeds_processed = 0;
    long long grand_total_passed_count = 0;
    auto overall_program_start_time = std::chrono::high_resolution_clock::now();

    std::vector<EWorldSize> world_sizes_to_iterate = {EWorldSize::Small, EWorldSize::Medium, EWorldSize::Large};

    for (EWorldSize current_world_size_enum : world_sizes_to_iterate) {
        printf("\n===== Processing World Size: %d =====\n", static_cast<int>(current_world_size_enum));

        size_t current_world_state_struct_size = 0;
        switch (current_world_size_enum) {
        case EWorldSize::Small:
            current_world_state_struct_size = sizeof(DeviceWorldState<EWorldSize::Small>);
            break;
        case EWorldSize::Medium:
            current_world_state_struct_size = sizeof(DeviceWorldState<EWorldSize::Medium>);
            break;
        case EWorldSize::Large:
            current_world_state_struct_size = sizeof(DeviceWorldState<EWorldSize::Large>);
            break;
        default:
            printf("Error: Invalid world size in loop for allocation.\n");
            continue;
        }
        cuda_err = cudaMalloc(&d_world_states_batch_generic, BATCH_SIZE * current_world_state_struct_size);
        if (cuda_err != cudaSuccess) {
            printf("cudaMalloc for d_world_states_batch (size %d) failed (%zu bytes needed): %s\n",
                   static_cast<int>(current_world_size_enum), BATCH_SIZE * current_world_state_struct_size, cudaGetErrorString(cuda_err));
            continue;
        }

        long long seeds_processed_this_size_type = 0;
        long long passed_this_size_type = 0;
        auto size_type_start_time = std::chrono::high_resolution_clock::now();

        for (long long seed_batch_start_value = 0; seed_batch_start_value < seeds_to_check_per_size_param; seed_batch_start_value += BATCH_SIZE) {

            long long remaining_seeds_for_this_size_type = seeds_to_check_per_size_param - seed_batch_start_value;
            int current_batch_actual_size = static_cast<int>(std::min((long long)BATCH_SIZE, remaining_seeds_for_this_size_type));

            if (current_batch_actual_size <= 0) {
                break;
            }

            for (int i = 0; i < current_batch_actual_size; ++i) {
                seeds[i] = static_cast<int>(seed_batch_start_value + i);
            }

            cuda_err = cudaMemcpy(seeds_batch, seeds.data(), current_batch_actual_size * sizeof(int32_t), cudaMemcpyHostToDevice);
            if (cuda_err != cudaSuccess) {
                break;
            }
            

            int blockSize = 1024;
            int numBlocks = (current_batch_actual_size + blockSize - 1) / blockSize;

            switch (current_world_size_enum) {
            case EWorldSize::Small:
                SimplifiedWorldGenCheck_Kernel<EWorldSize::Small><<<numBlocks, blockSize>>>(
                    seeds_batch, static_cast<DeviceWorldState<EWorldSize::Small> *>(d_world_states_batch_generic),
                    d_results_batch, current_batch_actual_size);
                break;
            case EWorldSize::Medium:
                SimplifiedWorldGenCheck_Kernel<EWorldSize::Medium><<<numBlocks, blockSize>>>(
                    seeds_batch, static_cast<DeviceWorldState<EWorldSize::Medium> *>(d_world_states_batch_generic),
                    d_results_batch, current_batch_actual_size);
                break;
            case EWorldSize::Large:
                SimplifiedWorldGenCheck_Kernel<EWorldSize::Large><<<numBlocks, blockSize>>>(
                    seeds_batch, static_cast<DeviceWorldState<EWorldSize::Large> *>(d_world_states_batch_generic),
                    d_results_batch, current_batch_actual_size);
                break;
            default:
                printf("Critical Error: Attempting to launch kernel for unhandled EWorldSize.\n");
                cuda_err = cudaErrorInvalidValue;
                break;
            }
            if (cuda_err == cudaErrorInvalidValue) {
                break;
            }

            cuda_err = cudaGetLastError();
            if (cuda_err != cudaSuccess) {
                break;
            }

            cuda_err = cudaDeviceSynchronize();
            if (cuda_err != cudaSuccess) {
                break;
            }

            cuda_err = cudaMemcpy(h_results_batch, d_results_batch, current_batch_actual_size * sizeof(bool), cudaMemcpyDeviceToHost);
            if (cuda_err != cudaSuccess) {
                break;
            }


            for (int i = 0; i < current_batch_actual_size; ++i) {
                if (h_results_batch[i]) {
                    passed_this_size_type++;
                    grand_total_passed_count++;

                    if (outfile.is_open()) {
                        outfile << (static_cast<int>(current_world_size_enum) + 1) << "." << seeds[i] <<"\n";
                        outfile.flush();
                    }
                    printf("%d.%d\n", (static_cast<int>(current_world_size_enum) + 1), seeds[i]);
                }
            }
            seeds_processed_this_size_type += current_batch_actual_size;
            grand_total_seeds_processed += current_batch_actual_size;

            if ((seed_batch_start_value / BATCH_SIZE) % 10 == 0 || seed_batch_start_value + current_batch_actual_size >= seeds_to_check_per_size_param) {
                auto ct = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> ems = ct - overall_program_start_time;
                double sps = (ems.count() > 0) ? (grand_total_seeds_processed / (ems.count() / 1000.0)) : 0;
                printf("  Prog (Size %d): %lld/%lld. Passed this size: %lld | Total Overall: %lld. Total Passed: %lld (%.2f seeds/sec overall)\n",
                       static_cast<int>(current_world_size_enum),
                       seeds_processed_this_size_type, seeds_to_check_per_size_param, passed_this_size_type,
                       grand_total_seeds_processed, grand_total_passed_count, sps);
            }
            if (cuda_err != cudaSuccess) {
                break;
            }
        }

        if (d_world_states_batch_generic) {
            cudaFree(d_world_states_batch_generic);
            d_world_states_batch_generic = nullptr;
        }

        auto size_type_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_this_size_ms = size_type_end_time - size_type_start_time;
        double seeds_per_sec_this_size = (duration_this_size_ms.count() > 0 && seeds_processed_this_size_type > 0) ? 
        (seeds_processed_this_size_type / (duration_this_size_ms.count() / 1000.0)) : 0.0;
        printf("  Finished Size %d: Processed %lld seeds, %lld passed. Time: %.2f ms (%.2f seeds/sec for this size)\n", 
               static_cast<int>(current_world_size_enum), 
               seeds_processed_this_size_type, 
               passed_this_size_type, 
               duration_this_size_ms.count(),
               seeds_per_sec_this_size);

        if (cuda_err != cudaSuccess) {
            break;
        }
    }

    if (outfile.is_open()) {
        outfile.close();
    }

    auto overall_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration_ms = overall_end_time - overall_start_time;

    printf("\n--- Processing Complete ---\n");
    printf("Total seeds checked: %lld\n", total_seeds_processed_count);
    printf("Total seeds passed check: %lld\n", total_passed_check_count);
    printf("Total execution time: %.3f ms (%.3f seconds)\n", total_duration_ms.count(), total_duration_ms.count() / 1000.0);
    if (total_seeds_processed_count > 0 && total_duration_ms.count() > 0) {
        printf("Average time per seed (overall): %.6f ms\n", total_duration_ms.count() / total_seeds_processed_count);
        printf("Overall Seeds processed per second: %.2f\n", (double)total_seeds_processed_count / (total_duration_ms.count() / 1000.0));
    }

    cudaFree(seeds_batch);
    if (d_world_states_batch_generic) { // Should be free already
        cudaFree(d_world_states_batch_generic);
    }
    cudaFree(d_results_batch);
    delete[] h_results_batch;

    return 0;
}