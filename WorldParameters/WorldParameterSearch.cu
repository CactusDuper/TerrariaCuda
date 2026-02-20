#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <vector>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <map>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

constexpr int32_t INT_MAX_VAL = 2147483647;

constexpr int32_t MAX_HITS_PER_BATCH = 102400;
constexpr int BLOCKS_PER_GRID = 8192;
constexpr int THREADS_PER_BLOCK = 128;
constexpr int64_t BATCH_SIZE = (int64_t)BLOCKS_PER_GRID * THREADS_PER_BLOCK;
constexpr int64_t TOTAL_SEEDS = (int64_t)INT_MAX_VAL + 1;

enum class EDungeonSide {
    Left,
    Right
};

enum class EWorldSize {
    Small,
    Medium,
    Large,
    INVALID
};

// -1 is any
// More comments are in config file
struct SearchConfig {
    // RandomizeTreeStyle
    int treeStyle[4] = {-1, -1, -1, -1};

    // RandomizeCaveBackgrounds
    int caveStyle[4] = {-1, -1, -1, -1};
    int iceBackStyle = -1;
    int hellBackStyle = -1;
    int jungleBackStyle = -1;

    // RandomizeBackgrounds
    int32_t treeBG[4] = {-1, -1, -1, -1};
    int corruptBG = -1;
    int jungleBG = -1;
    int snowBG = -1;
    int hallowBG = -1;
    int crimsonBG = -1;
    int desertBG = -1;
    int oceanBG = -1;
    int mushroomBG = -1;
    int underworldBG = -1;

    int maxSeeds = 10000;
    int worldSize = 0;
    int dungeonSide = -1;
    int oreTier1 = -1;
    int oreTier2 = -1;
    int oreTier3 = -1;
    int oreTier4 = -1;
    int evil = -1;
    int moonStyle = -1;
};

__constant__ SearchConfig cConfig;

static constexpr int MSEED = 161803398;

__device__ __forceinline__ int32_t SubAndNorm(int32_t a, int32_t b) {
    int32_t ret = a - b;
    if (ret == INT_MAX_VAL) { ret--; } // Can be removed for more speed but might result in invalid results
    if (ret < 0) { ret += INT_MAX_VAL; } 
    return ret; // return ret + ((ret >> 31) & INT_MAX_VAL); is slower???
}

__device__ __forceinline__ int32_t RandomInternalSample(int32_t* seedArray, uint32_t& inext) {
    [[assume(inext <= 56)]];
    uint32_t locINext = (inext == 55) ? 1 : inext + 1;

    uint32_t locINextp = locINext + 21;
    if (locINextp > 55) {
        locINextp -= 55;
    }

    int32_t retVal = seedArray[locINext] - seedArray[locINextp];
    if (retVal == INT_MAX_VAL) {
        retVal--;
    }

    retVal += (retVal >> 31) & INT_MAX_VAL;

    [[assume(retVal >= 0 && retVal < INT_MAX_VAL)]];
    seedArray[locINext] = retVal;
    inext = locINext;
    return retVal;
}

__device__ __forceinline__ void RandomInitBase(int32_t* seedArray, int32_t seed) {
    int32_t mj = MSEED - seed;
    int32_t mk = 1;

    seedArray[55] = mj;

    #pragma unroll
    for (int i = 1; i < 55; i++) {
        int ii = (21 * i) % 55; 

        seedArray[ii] = mk;
        mj = SubAndNorm(mj, mk);
        
        int32_t temp = mk;
        mk = mj;
        mj = temp;
    }
}

__device__ void __forceinline__ RandomInitFull(int32_t* seedArray, int32_t seed) {
    RandomInitBase(seedArray, seed);

    for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int i = 1; i <= 24; i++) {
            seedArray[i] = SubAndNorm(seedArray[i], seedArray[i + 31]);
        }
        #pragma unroll
        for (int i = 25; i <= 55; i++) {
            seedArray[i] = SubAndNorm(seedArray[i], seedArray[i - 24]);
        }
    }
}

__device__ __forceinline__ double RandomSample(int32_t* seedArray, uint32_t& inext) {
    return static_cast<double>(RandomInternalSample(seedArray, inext)) * 4.656612875245797E-10;
}

__device__ __forceinline__ int32_t RandomNext(int32_t* seedArray, uint32_t& inext) {
    return RandomInternalSample(seedArray, inext);
}

__device__ __forceinline__ int32_t RandomNextMax(int32_t* seedArray, uint32_t& inext, int32_t maxValue) {
    return static_cast<int32_t>((RandomSample(seedArray, inext) * static_cast<double>(maxValue)));
}

__device__ __forceinline__ int32_t RandomNextRange(int32_t* seedArray, uint32_t& inext, int32_t minValue, int32_t maxValue) {
    int64_t range = static_cast<int64_t>(maxValue) - static_cast<int64_t>(minValue);
    return static_cast<int32_t>(RandomSample(seedArray, inext) * static_cast<double>(range)) + minValue;
}

__device__ __forceinline__ bool CheckSetInclusion(const int* generated, int genCount, const int* config, int configCount) {
    for (int c = 0; c < configCount; c++) {
        int required = config[c];
        if (required == -1) {
            continue;
        }

        bool found = false;
        for (int g = 0; g < genCount; g++) {
            if (generated[g] == required) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

template <EWorldSize SIZE>
__device__ __forceinline__ bool RandomizeTreeStyle(int32_t* seedArray, uint32_t& inext) {
    RandomInternalSample(seedArray, inext);
    if constexpr (SIZE == EWorldSize::Medium || SIZE == EWorldSize::Large) RandomInternalSample(seedArray, inext);
    if constexpr (SIZE == EWorldSize::Large) RandomInternalSample(seedArray, inext);

    int32_t treeStyle[4];
    const int maxValue = 6;

    treeStyle[0] = RandomNextMax(seedArray, inext, maxValue);
    treeStyle[1] = RandomNextMax(seedArray, inext, maxValue);

    if constexpr (SIZE == EWorldSize::Medium || SIZE == EWorldSize::Large) { treeStyle[2] = RandomNextMax(seedArray, inext, maxValue); }
    if constexpr (SIZE == EWorldSize::Large) { treeStyle[3] = RandomNextMax(seedArray, inext, maxValue); }

    while (treeStyle[1] == treeStyle[0]) { treeStyle[1] = RandomNextMax(seedArray, inext, maxValue); }

    if constexpr (SIZE == EWorldSize::Medium || SIZE == EWorldSize::Large) {
        while (treeStyle[2] == treeStyle[0] || treeStyle[2] == treeStyle[1]) { treeStyle[2] = RandomNextMax(seedArray, inext, maxValue); }
    }
    else {
        treeStyle[2] = -2;
    }

    if constexpr (SIZE == EWorldSize::Large) {
        while (treeStyle[3] == treeStyle[0] || treeStyle[3] == treeStyle[1] || treeStyle[3] == treeStyle[2]) { treeStyle[3] = RandomNextMax(seedArray, inext, maxValue); }
    }
    else {
        treeStyle[3] = -2;
    }

    constexpr int count = (SIZE == EWorldSize::Small) ? 2 : ((SIZE == EWorldSize::Medium) ? 3 : 4);

    for (int i = 0; i < count; i++) {
        if (treeStyle[i] == 0 && RandomNextMax(seedArray, inext, 3) != 0) {
            treeStyle[i] = 4;
        }
    }

    return CheckSetInclusion(treeStyle, count, cConfig.treeStyle, count);
}

template <EWorldSize SIZE>
__device__ __forceinline__ bool RandomizeCaveBackgrounds(int32_t* seedArray, uint32_t& inext) {
    RandomInternalSample(seedArray, inext);
    if constexpr (SIZE == EWorldSize::Medium || SIZE == EWorldSize::Large) RandomInternalSample(seedArray, inext);
    if constexpr (SIZE == EWorldSize::Large) RandomInternalSample(seedArray, inext);

    int32_t caveStyle[4];
    const int maxValue = 8;

    caveStyle[0] = RandomNextMax(seedArray, inext, maxValue);
    caveStyle[1] = RandomNextMax(seedArray, inext, maxValue);

    if constexpr (SIZE == EWorldSize::Medium || SIZE == EWorldSize::Large) { caveStyle[2] = RandomNextMax(seedArray, inext, maxValue); }
    if constexpr (SIZE == EWorldSize::Large) { caveStyle[3] = RandomNextMax(seedArray, inext, maxValue); }

    while (caveStyle[1] == caveStyle[0]) {
        caveStyle[1] = RandomNextMax(seedArray, inext, maxValue);
    }

    if constexpr (SIZE == EWorldSize::Medium || SIZE == EWorldSize::Large) {
        while (caveStyle[2] == caveStyle[0] || caveStyle[2] == caveStyle[1]) { caveStyle[2] = RandomNextMax(seedArray, inext, maxValue); }
    }
    else {
        caveStyle[2] = -2;
    }

    if constexpr (SIZE == EWorldSize::Large) {
        while (caveStyle[3] == caveStyle[0] || caveStyle[3] == caveStyle[1] || caveStyle[3] == caveStyle[2]) { caveStyle[3] = RandomNextMax(seedArray, inext, maxValue); }
    }
    else {
        caveStyle[3] = -2;
    }

    int32_t iceBackStyle = RandomNextMax(seedArray, inext, 4);
    int32_t hellBackStyle = RandomNextMax(seedArray, inext, 3);
    int32_t jungleBackStyle = RandomNextMax(seedArray, inext, 2);

    if (cConfig.iceBackStyle != -1 && iceBackStyle != cConfig.iceBackStyle) { return false; }
    if (cConfig.hellBackStyle != -1 && hellBackStyle != cConfig.hellBackStyle) { return false; }
    if (cConfig.jungleBackStyle != -1 && jungleBackStyle != cConfig.jungleBackStyle) { return false; }

    constexpr int count = (SIZE == EWorldSize::Small) ? 2 : ((SIZE == EWorldSize::Medium) ? 3 : 4);
    return CheckSetInclusion(caveStyle, count, cConfig.caveStyle, count);
}

__device__ __forceinline__ int RollRandomForestBGStyle(int32_t* seedArray, uint32_t& inext) {
    int num = RandomNextMax(seedArray, inext, 14);
    if ((num == 1 || num == 2) && RandomNextMax(seedArray, inext, 2) == 0) { num = RandomNextMax(seedArray, inext, 14); }
    if (num == 0) { num = RandomNextMax(seedArray, inext, 14); }
    if (num == 3 && RandomNextMax(seedArray, inext, 3) == 0) { num = 31; } // mountain1: 90, tree0: 91, tree1: -1, tree2: 11
    if (num == 5 && RandomNextMax(seedArray, inext, 2) == 0) { num = 51; } // mountain1: 93, mountain2: 94, tree0: -1, tree1: -1, tree2: 11
    if (num == 7 && RandomNextMax(seedArray, inext, 4) == 0) { num = RandomNextRange(seedArray, inext, 71, 74); } // mountain1: 176, mountain2: 177, tree0: 178, tree1: -1, tree2: 11/52/55
    return num;
}

__device__ __forceinline__ bool RandomizeBackgrounds(int32_t* seedArray, uint32_t& inext) {
    int32_t treeBG[4];

    treeBG[0] = RollRandomForestBGStyle(seedArray, inext);

    do { treeBG[1] = RollRandomForestBGStyle(seedArray, inext); } while (treeBG[1] == treeBG[0]);

    do { treeBG[2] = RollRandomForestBGStyle(seedArray, inext); } while (treeBG[2] == treeBG[0] || treeBG[2] == treeBG[1]);

    do { treeBG[3] = RollRandomForestBGStyle(seedArray, inext); } while (treeBG[3] == treeBG[0] || treeBG[3] == treeBG[1] || treeBG[3] == treeBG[2]);

    int corruptBG = RandomNextMax(seedArray, inext, 6);
    if (corruptBG == 5) {
        corruptBG = 51 + RandomNextMax(seedArray, inext, 2); // 51 or 52
    }

    int jungleBG = RandomNextMax(seedArray, inext, 7);

    int snowBG = RandomNextMax(seedArray, inext, 9);
    if (snowBG == 2 && RandomNextMax(seedArray, inext, 2) == 0) {
        snowBG = 21 + RandomNextMax(seedArray, inext, 2); // 21 or 22
    }
    else if (snowBG == 3 && RandomNextMax(seedArray, inext, 2) == 0) {
        snowBG = 31 + RandomNextMax(seedArray, inext, 2); // 31 or 32
    }
    else if (snowBG == 4 && RandomNextMax(seedArray, inext, 2) == 0) {
        snowBG = 41 + RandomNextMax(seedArray, inext, 2); // 41 or 42
    }
    // Snow can be 0-8, 21-22, 31-32, 41-42

    
    int hallowBG = RandomNextMax(seedArray, inext, 6);
    int crimsonBG = RandomNextMax(seedArray, inext, 7);

    int desertBG = RandomNextMax(seedArray, inext, 6);
    if (desertBG == 5) {
        desertBG = 51 + (RandomNextMax(seedArray, inext, 5) / 2);
    }
    // Desert can be 0-4 and 51-53
    
    int oceanBG = RandomNextMax(seedArray, inext, 8);
    int mushroomBG = RandomNextMax(seedArray, inext, 5);
    int underworldBG = RandomNextMax(seedArray, inext, 3);


    if (cConfig.corruptBG != -1 && corruptBG != cConfig.corruptBG) { return false; }
    if (cConfig.jungleBG != -1 && jungleBG != cConfig.jungleBG) { return false; }
    if (cConfig.snowBG != -1 && snowBG != cConfig.snowBG) { return false; }
    if (cConfig.hallowBG != -1 && hallowBG != cConfig.hallowBG) { return false; }
    if (cConfig.crimsonBG != -1 && crimsonBG != cConfig.crimsonBG) { return false; }
    if (cConfig.desertBG != -1 && desertBG != cConfig.desertBG) { return false; }
    if (cConfig.oceanBG != -1 && oceanBG != cConfig.oceanBG) { return false; }
    if (cConfig.mushroomBG != -1 && mushroomBG != cConfig.mushroomBG) { return false; }
    if (cConfig.underworldBG != -1 && underworldBG != cConfig.underworldBG) { return false; }

    return CheckSetInclusion(treeBG, 4, cConfig.treeBG, 4);
}

__device__ __forceinline__ void RandomizeWeather(int32_t* seedArray, uint32_t& inext) {
    RandomInternalSample(seedArray, inext); // Number of clouds
    // In 1.4.5, only a single seed will reroll. It's fine to keep as is for general purpose things
    while (true) {
        const float val = (float)RandomSample(seedArray, inext) * 0.35f * (float)(RandomNextMax(seedArray, inext, 2) * 2 - 1);
        if (val != 0.0f) { break; }
    }
}

template <EWorldSize SIZE>
__device__ __forceinline__ bool ResetGenerationPass(int32_t* seedArray, uint32_t& inext) {
    RandomInternalSample(seedArray, inext); // jungleHut Next(5)
    RandomInternalSample(seedArray, inext); // crimson side Next(2)
    RandomizeWeather(seedArray, inext);
    RandomInternalSample(seedArray, inext); // list
    RandomInternalSample(seedArray, inext); // list
    RandomInternalSample(seedArray, inext); // list
    RandomInternalSample(seedArray, inext); // list
    RandomInternalSample(seedArray, inext); // list
    RandomInternalSample(seedArray, inext); // slimeRainTime
    RandomInternalSample(seedArray, inext); // cloudBGActive
    int oreTier1 = RandomNextMax(seedArray, inext, 2); // Copper, Tin
    int oreTier2 = RandomNextMax(seedArray, inext, 2); // Iron, Lead
    int oreTier3 = RandomNextMax(seedArray, inext, 2); // Silver, Tungsten
    int oreTier4 = RandomNextMax(seedArray, inext, 2); // Gold, Platinum
    int evil = RandomNextMax(seedArray, inext, 2); // Corruption, Crimson
    RandomInternalSample(seedArray, inext); // World ID
    bool treeMatch = RandomizeTreeStyle<SIZE>(seedArray, inext);
    bool caveMatch = RandomizeCaveBackgrounds<SIZE>(seedArray, inext);
    bool backgroundMatch = RandomizeBackgrounds(seedArray, inext);
    int moonStyle = RandomNextMax(seedArray, inext, 9);

    int dungeonSide = (RandomNextMax(seedArray, inext, 2) == 0) ? (int)EDungeonSide::Left : (int)EDungeonSide::Right;

    if (!treeMatch || !caveMatch || !backgroundMatch) {
        return false;
    }

        if (cConfig.dungeonSide != -1 && dungeonSide != cConfig.dungeonSide) { return false; }
        if (cConfig.oreTier1 != -1 && oreTier1 != cConfig.oreTier1) { return false; }
        if (cConfig.oreTier2 != -1 && oreTier2 != cConfig.oreTier2) { return false; }
        if (cConfig.oreTier3 != -1 && oreTier3 != cConfig.oreTier3) { return false; }
        if (cConfig.oreTier4 != -1 && oreTier4 != cConfig.oreTier4) { return false; }
        if (cConfig.evil != -1 && evil != cConfig.evil) { return false; }
        if (cConfig.moonStyle != -1 && moonStyle != cConfig.moonStyle) { return false; }

        return true;
}


template <EWorldSize SIZE>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
CheckSeedKernel(int64_t baseSeedOffset, int32_t* dFoundSeeds, int32_t* dFoundCount) {
    const int64_t idx = baseSeedOffset + (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx > TOTAL_SEEDS) {
        return;
    }

    const int32_t currentSeed = (int32_t)idx;
    int32_t seedArray[56]; 
    uint32_t inext = 0;

    RandomInitFull(seedArray, currentSeed);

    if(ResetGenerationPass<SIZE>(seedArray, inext)) {
        int idx = atomicAdd(dFoundCount, 1);
        if (idx < MAX_HITS_PER_BATCH) {
            dFoundSeeds[idx] = currentSeed;
        }
    }
}

void LoadConfig(const std::string& filename, SearchConfig& cfg) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("WorldConfig.txt not found, exiting now. Make sure you download it from the repository and place it next to the executable!\n");
        std::exit(1);
    }

    std::string line;
    std::map<std::string, int*> intMap;
    intMap["maxSeeds"] = &cfg.maxSeeds;
    intMap["worldSize"] = &cfg.worldSize;
    intMap["dungeonSide"] = &cfg.dungeonSide;
    intMap["evil"] = &cfg.evil;
    intMap["moonStyle"] = &cfg.moonStyle;
    intMap["oreTier1"] = &cfg.oreTier1;
    intMap["oreTier2"] = &cfg.oreTier2;
    intMap["oreTier3"] = &cfg.oreTier3;
    intMap["oreTier4"] = &cfg.oreTier4;

    intMap["treeStyle0"] = &cfg.treeStyle[0];
    intMap["treeStyle1"] = &cfg.treeStyle[1]; 
    intMap["treeStyle2"] = &cfg.treeStyle[2];
    intMap["treeStyle3"] = &cfg.treeStyle[3];

    intMap["caveStyle0"] = &cfg.caveStyle[0];
    intMap["caveStyle1"] = &cfg.caveStyle[1]; 
    intMap["caveStyle2"] = &cfg.caveStyle[2];
    intMap["caveStyle3"] = &cfg.caveStyle[3];

    intMap["treeBG0"] = &cfg.treeBG[0];
    intMap["treeBG1"] = &cfg.treeBG[1]; 
    intMap["treeBG2"] = &cfg.treeBG[2];
    intMap["treeBG3"] = &cfg.treeBG[3];

    intMap["corruptBG"] = &cfg.corruptBG;
    intMap["jungleBG"] = &cfg.jungleBG;
    intMap["snowBG"] = &cfg.snowBG;
    intMap["hallowBG"] = &cfg.hallowBG;
    intMap["crimsonBG"] = &cfg.crimsonBG;
    intMap["desertBG"] = &cfg.desertBG;
    intMap["oceanBG"] = &cfg.oceanBG;
    intMap["mushroomBG"] = &cfg.mushroomBG;
    intMap["underworldBG"] = &cfg.underworldBG;
    intMap["iceBackStyle"] = &cfg.iceBackStyle;
    intMap["hellBackStyle"] = &cfg.hellBackStyle;
    intMap["jungleBackStyle"] = &cfg.jungleBackStyle;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        size_t commentPos = line.find('#');
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string key;
        int val;
        if (ss >> key >> val) {
            if (intMap.count(key)) {
                *(intMap[key]) = val;
            }
        }
    }
}

int main(int argc, char **argv) {
    SearchConfig hostConfig;
    LoadConfig("WorldConfig.txt", hostConfig);

    CHECK_CUDA(cudaMemcpyToSymbol(cConfig, &hostConfig, sizeof(SearchConfig)));

    int32_t* dFoundSeeds;
    int32_t* dFoundCount;

    CHECK_CUDA(cudaMalloc(&dFoundSeeds, MAX_HITS_PER_BATCH * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dFoundCount, sizeof(int32_t)));

    int32_t* hFoundSeeds = new int32_t[MAX_HITS_PER_BATCH];
    int32_t hFoundCount = 0;
    
    int64_t totalSaved = 0;

    std::ofstream outfile("found_seeds.txt", std::ios_base::app);
    const auto tStart = std::chrono::high_resolution_clock::now();

    for (int64_t offset = 0; offset < TOTAL_SEEDS; offset += BATCH_SIZE) {
        CHECK_CUDA(cudaMemset(dFoundCount, 0, sizeof(int32_t)));
        
        switch (hostConfig.worldSize) { // Could remove template for this...
            case 0: CheckSeedKernel<EWorldSize::Small><<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(offset, dFoundSeeds, dFoundCount); break;
            case 1: CheckSeedKernel<EWorldSize::Medium><<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(offset, dFoundSeeds, dFoundCount); break;
            case 2: CheckSeedKernel<EWorldSize::Large><<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(offset, dFoundSeeds, dFoundCount); break;
            default: break;
        }
        
        CHECK_CUDA(cudaMemcpy(&hFoundCount, dFoundCount, sizeof(int32_t), cudaMemcpyDeviceToHost));

        if (hFoundCount > 0) {
            const int copyCount = std::min(hFoundCount, MAX_HITS_PER_BATCH);
            CHECK_CUDA(cudaMemcpy(hFoundSeeds, dFoundSeeds, copyCount * sizeof(int32_t), cudaMemcpyDeviceToHost));

            for (int i = 0; i < copyCount; i++) {
                if(totalSaved >= hostConfig.maxSeeds){
                    goto endSearch;
                }

                if (outfile.is_open()) {
                    outfile << hFoundSeeds[i] << "\n";
                    totalSaved++;
                }
            }
            
            if (hFoundCount > MAX_HITS_PER_BATCH) {
                printf("WARNING: Batch overflow! Found %d seeds but buffer only holds %d.\n", hFoundCount, MAX_HITS_PER_BATCH);
            }
        }

        if ((offset % (BATCH_SIZE * 100)) == 0) {
            double pct = (double)offset / TOTAL_SEEDS * 100.0;
            printf("  Progress: %.1f%%\r", pct);
            fflush(stdout);
        }
    }

endSearch:
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto tEnd = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = tEnd - tStart;
    
    printf("\n\nFinished! Total Time: %.2fs\n", duration.count());
    
    cudaFree(dFoundSeeds);
    cudaFree(dFoundCount);
    delete[] hFoundSeeds;
    outfile.close();

    return 0;
}