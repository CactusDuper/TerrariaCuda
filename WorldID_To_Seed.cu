// This is for when (316080314LL * (long long)worldId + 75192552LL) % 2147483647LL does NOT result in the correct seed (OR you want to search for collisions)
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstdio>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Precomputed Bounds for (x * 801 / INT_MAX) == 400
constexpr int32_t WIND_400_MIN_RAW = 1072401322;
constexpr int32_t WIND_400_MAX_RAW = 1075082325;
constexpr uint32_t WIND_RANGE_LEN = WIND_400_MAX_RAW - WIND_400_MIN_RAW;

constexpr int32_t MSEED = 161803398;
constexpr int32_t INT_MAX_VAL = 2147483647;


// IF YOUR GPU DOES NOT RUN/THIS PROGRAM CRASHES, LOWER THREADS_PER_BLOCK TO 256.
// IF THAT DOES NOT WORK, LOWER BLOCKS_PER_GRID TO 2048 AS WELL.
constexpr int THREADS_PER_BLOCK = 512; // TODO: Tweak for older GPUs
constexpr int BLOCKS_PER_GRID = 4096;  // TODO: Also tweak
constexpr int MAX_HITS_PER_BATCH = 4;  // Max matches expected per batch (Really only 1 in the entire seedspace but w/e)

__device__ __forceinline__ int32_t SubAndNorm(int32_t a, int32_t b) {
    int32_t ret = a - b;
    // if (ret == INT_MAX_VAL) { ret--; } // Can be ignored based on tests
    if (ret < 0) { ret += INT_MAX_VAL; } 
    return ret; // return ret + ((ret >> 31) & INT_MAX_VAL); is slower???
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

// Full (1 / 800)
__device__ void RandomInitFull(int32_t* seedArray, int32_t seed) {
    RandomInitBase(seedArray, seed);

    for (int k = 0; k < 4; k++) {
        // Range 1: i=1 to 24.  idx = i + 30 + 1 = i + 31
        #pragma unroll
        for (int i = 1; i <= 24; i++) {
            seedArray[i] = SubAndNorm(seedArray[i], seedArray[i + 31]);
        }
        // Range 2: i=25 to 55. idx = (i + 30) - 55 + 1 = i - 24
        #pragma unroll
        for (int i = 25; i <= 55; i++) {
            seedArray[i] = SubAndNorm(seedArray[i], seedArray[i - 24]);
        }
    }
}

// Sparse (799 / 800)
__device__ __forceinline__ void RandomInitSparse(int32_t* seedArray) {
    // PASS 1
    seedArray[3] = SubAndNorm(seedArray[3], seedArray[34]);
    seedArray[6] = SubAndNorm(seedArray[6], seedArray[37]);
    seedArray[7] = SubAndNorm(seedArray[7], seedArray[38]);
    seedArray[10] = SubAndNorm(seedArray[10], seedArray[41]);
    seedArray[13] = SubAndNorm(seedArray[13], seedArray[44]);
    seedArray[14] = SubAndNorm(seedArray[14], seedArray[45]);
    seedArray[16] = SubAndNorm(seedArray[16], seedArray[47]);
    seedArray[17] = SubAndNorm(seedArray[17], seedArray[48]);
    seedArray[20] = SubAndNorm(seedArray[20], seedArray[51]);
    seedArray[21] = SubAndNorm(seedArray[21], seedArray[52]);
    seedArray[23] = SubAndNorm(seedArray[23], seedArray[54]);
    seedArray[24] = SubAndNorm(seedArray[24], seedArray[55]);
    seedArray[27] = SubAndNorm(seedArray[27], seedArray[3]);
    seedArray[30] = SubAndNorm(seedArray[30], seedArray[6]);
    seedArray[31] = SubAndNorm(seedArray[31], seedArray[7]);
    seedArray[34] = SubAndNorm(seedArray[34], seedArray[10]);
    seedArray[37] = SubAndNorm(seedArray[37], seedArray[13]);
    seedArray[38] = SubAndNorm(seedArray[38], seedArray[14]);
    seedArray[41] = SubAndNorm(seedArray[41], seedArray[17]);
    seedArray[44] = SubAndNorm(seedArray[44], seedArray[20]);
    seedArray[45] = SubAndNorm(seedArray[45], seedArray[21]);
    seedArray[47] = SubAndNorm(seedArray[47], seedArray[23]);
    seedArray[48] = SubAndNorm(seedArray[48], seedArray[24]);
    seedArray[51] = SubAndNorm(seedArray[51], seedArray[27]);
    seedArray[54] = SubAndNorm(seedArray[54], seedArray[30]);
    seedArray[55] = SubAndNorm(seedArray[55], seedArray[31]);

    // PASS 2
    seedArray[3] = SubAndNorm(seedArray[3], seedArray[34]);
    seedArray[6] = SubAndNorm(seedArray[6], seedArray[37]);
    seedArray[7] = SubAndNorm(seedArray[7], seedArray[38]);
    seedArray[10] = SubAndNorm(seedArray[10], seedArray[41]);
    seedArray[13] = SubAndNorm(seedArray[13], seedArray[44]);
    seedArray[14] = SubAndNorm(seedArray[14], seedArray[45]);
    seedArray[16] = SubAndNorm(seedArray[16], seedArray[47]);
    seedArray[17] = SubAndNorm(seedArray[17], seedArray[48]);
    seedArray[20] = SubAndNorm(seedArray[20], seedArray[51]);
    seedArray[23] = SubAndNorm(seedArray[23], seedArray[54]);
    seedArray[24] = SubAndNorm(seedArray[24], seedArray[55]);
    seedArray[27] = SubAndNorm(seedArray[27], seedArray[3]);
    seedArray[30] = SubAndNorm(seedArray[30], seedArray[6]);
    seedArray[31] = SubAndNorm(seedArray[31], seedArray[7]);
    seedArray[34] = SubAndNorm(seedArray[34], seedArray[10]);
    seedArray[37] = SubAndNorm(seedArray[37], seedArray[13]);
    seedArray[38] = SubAndNorm(seedArray[38], seedArray[14]);
    seedArray[41] = SubAndNorm(seedArray[41], seedArray[17]);
    seedArray[44] = SubAndNorm(seedArray[44], seedArray[20]);
    seedArray[47] = SubAndNorm(seedArray[47], seedArray[23]);
    seedArray[51] = SubAndNorm(seedArray[51], seedArray[27]);
    seedArray[54] = SubAndNorm(seedArray[54], seedArray[30]);
    seedArray[55] = SubAndNorm(seedArray[55], seedArray[31]);

    // PASS 3
    seedArray[3] = SubAndNorm(seedArray[3], seedArray[34]);
    seedArray[7] = SubAndNorm(seedArray[7], seedArray[38]);
    seedArray[10] = SubAndNorm(seedArray[10], seedArray[41]);
    seedArray[13] = SubAndNorm(seedArray[13], seedArray[44]);
    seedArray[16] = SubAndNorm(seedArray[16], seedArray[47]);
    seedArray[20] = SubAndNorm(seedArray[20], seedArray[51]);
    seedArray[23] = SubAndNorm(seedArray[23], seedArray[54]);
    seedArray[24] = SubAndNorm(seedArray[24], seedArray[55]);
    seedArray[31] = SubAndNorm(seedArray[31], seedArray[7]);
    seedArray[34] = SubAndNorm(seedArray[34], seedArray[10]);
    seedArray[37] = SubAndNorm(seedArray[37], seedArray[13]);
    seedArray[44] = SubAndNorm(seedArray[44], seedArray[20]);
    seedArray[47] = SubAndNorm(seedArray[47], seedArray[23]);
    seedArray[55] = SubAndNorm(seedArray[55], seedArray[31]);

    // PASS 4
    seedArray[3] = SubAndNorm(seedArray[3], seedArray[34]);
    seedArray[13] = SubAndNorm(seedArray[13], seedArray[44]);
    seedArray[16] = SubAndNorm(seedArray[16], seedArray[47]);
    seedArray[24] = SubAndNorm(seedArray[24], seedArray[55]);
    seedArray[37] = SubAndNorm(seedArray[37], seedArray[13]);
}

__device__ __noinline__ int32_t GetWorldIDSlowPath(int32_t seed) {
    int32_t seedArray[56];

    RandomInitFull(seedArray, seed);

    // Wind Loop
    uint32_t inext = 3; 
    while (true) {
        inext++;
        const uint32_t inextp = inext + 21;             
        const int32_t val = SubAndNorm(seedArray[inext], seedArray[inextp]);
        // Check if still 400
        if ((uint32_t)(val - WIND_400_MIN_RAW) > WIND_RANGE_LEN) { break; }
    }

    inext += 13;
    return SubAndNorm(seedArray[inext], seedArray[inext + 21]);
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2)
CheckKernel(long long baseSeedOffset, int32_t targetWorldId, int32_t* dFoundSeeds, int32_t* dFoundCount, long long maxSeedLimit) {
    const long long idx = baseSeedOffset + (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx > maxSeedLimit) { // GE?
        return;
    }
    
    const int32_t currentSeed = (int32_t)idx;
    int32_t seedArray[56];
    int32_t result;

    // Start with sparse
    RandomInitBase(seedArray, currentSeed);
    RandomInitSparse(seedArray);

    // Wind uses the 3rd call    
    const int32_t windVal = SubAndNorm(seedArray[3], seedArray[24]);
    const bool isWind400 = ((uint32_t)(windVal - WIND_400_MIN_RAW) <= WIND_RANGE_LEN); // Optimized check

    if (!isWind400) { // 799 / 800
        // Skip 12 calls
        // [16] and [37] are good from RandomInitSparse        
        result = SubAndNorm(seedArray[16], seedArray[37]);
    } 
    else { // 1 / 800
        result = GetWorldIDSlowPath(currentSeed);
    }

    // Check if match
    if (result == targetWorldId) {
        int idx = atomicAdd(dFoundCount, 1);
        if (idx < MAX_HITS_PER_BATCH) {
            dFoundSeeds[idx] = currentSeed;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <world_id>\n", argv[0]);
        return 1;
    }

    int32_t worldId = static_cast<int32_t>(std::atoi(argv[1]));
    printf("Target World ID: %d\n", worldId);

    // Device
    int32_t* dFoundSeeds;
    int32_t* dFoundCount;
    CHECK_CUDA(cudaMalloc(&dFoundSeeds, MAX_HITS_PER_BATCH * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dFoundCount, sizeof(int32_t)));

    // Host
    int32_t* hFoundSeeds = new int32_t[MAX_HITS_PER_BATCH];
    int32_t hFoundCount = 0;

    constexpr long long totalSeeds = static_cast<long long>(INT_MAX_VAL) + 1; // 0 to INT_MAX
    constexpr long long batchSize = (long long)BLOCKS_PER_GRID * THREADS_PER_BLOCK; 

    printf("Search Space: %lld seeds\n", totalSeeds);
    printf("Batch Size:   %lld\n", batchSize);
    printf("Searching...\n");

    const auto t1 = std::chrono::high_resolution_clock::now();

    for (long long offset = 0; offset < totalSeeds; offset += batchSize) {
        CHECK_CUDA(cudaMemset(dFoundCount, 0, sizeof(int32_t)));

        // Launch Kernel
        CheckKernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
            offset, 
            worldId, 
            dFoundSeeds, 
            dFoundCount, 
            totalSeeds
        );

        // Copy count back to CPU
        CHECK_CUDA(cudaMemcpy(&hFoundCount, dFoundCount, sizeof(int32_t), cudaMemcpyDeviceToHost));

        if (hFoundCount > 0) {
            const int copyCount = std::min(hFoundCount, MAX_HITS_PER_BATCH);
            CHECK_CUDA(cudaMemcpy(hFoundSeeds, dFoundSeeds, copyCount * sizeof(int32_t), cudaMemcpyDeviceToHost));

            for (int i = 0; i < copyCount; i++) {
                printf("Found Seed: %d\n", hFoundSeeds[i]);
            }
        }
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    const auto t2 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> diff = t2 - t1;

    printf("\nDone.\nTime: %.2f s\n", diff.count());
    printf("Speed: %.2f Million seeds/sec\n", (totalSeeds / 1e6) / diff.count());

    // Cleanup
    cudaFree(dFoundSeeds);
    cudaFree(dFoundCount);
    delete[] hFoundSeeds;

    return 0;
}


