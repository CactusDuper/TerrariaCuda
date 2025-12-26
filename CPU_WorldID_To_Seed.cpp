#include <array>
#include <chrono>
#include <iostream>
#include <cmath>

constexpr int32_t WIND_400_MIN_RAW = 1072401322;
constexpr int32_t WIND_400_MAX_RAW = 1075082325;
constexpr uint32_t WIND_RANGE_LEN = WIND_400_MAX_RAW - WIND_400_MIN_RAW;

constexpr int32_t INT_MAX_VAL = 2147483647;
constexpr int32_t MSEED = 161803398;

constexpr int32_t SubAndNorm(int32_t a, int32_t b) {
    int32_t ret = a - b;
    return ret + ((ret >> 31) & INT_MAX_VAL);
}

constexpr void RandomInitFull(int32_t* seedArray, int32_t seed) {
    int32_t mj = MSEED - seed;
    int32_t mk = 1;
    seedArray[55] = mj;

    for (int i = 1; i < 55; i++) {
        int ii = (21 * i) % 55;
        seedArray[ii] = mk;
        mj = SubAndNorm(mj, mk);
        const int32_t temp = mk;
        mk = mj;
        mj = temp;
    }

    for (int k = 0; k < 4; k++) {
        for (int i = 1; i <= 24; i++) {
            seedArray[i] = SubAndNorm(seedArray[i], seedArray[i + 31]);
        }
        for (int i = 25; i <= 55; i++) {
            seedArray[i] = SubAndNorm(seedArray[i], seedArray[i - 24]);
        }
    }
}

constexpr int32_t GetCallAt(int32_t seed, int callIndex) {
    int32_t seedArray[56]{};
    RandomInitFull(seedArray, seed);
    return SubAndNorm(seedArray[callIndex], seedArray[callIndex + 21]); 
}

// Inverse(a) = a^(M-2) mod M
int64_t ModPow(int64_t base, int64_t exp, int64_t mod) {
    int64_t res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) {
            res = (res * base) % mod;
        }
        base = (base * base) % mod;
        exp /= 2;
    }
    return res;
}

int64_t ModInverse(int64_t n, int64_t mod) {
    return ModPow(n, mod - 2, mod);
}

int32_t GetWorldIDFull(int32_t seed) {
    int32_t seedArray[56];
    RandomInitFull(seedArray, seed);

    // Wind uses the 3rd call    
    const int32_t windVal = SubAndNorm(seedArray[3], seedArray[24]);
    const bool isWind400 = ((uint32_t)(windVal - WIND_400_MIN_RAW) <= WIND_RANGE_LEN);

    if (!isWind400) [[likely]] { // 799 / 800
        // Skip 12 calls
        return SubAndNorm(seedArray[16], seedArray[37]);
    } 
    else { // 1 / 800
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
}

// Overkill but w/e :)
constexpr std::array<int64_t, 5> B_ARR = {
    GetCallAt(0, 16),
    GetCallAt(0, 17),
    GetCallAt(0, 18),
    GetCallAt(0, 19),
    GetCallAt(0, 20)
};

constexpr int64_t CalcA(int idx){
    int64_t diff = (int64_t)GetCallAt(1, idx) - (int64_t)GetCallAt(0, idx);
    if (diff < 0) { diff += INT_MAX_VAL; }
    return diff;
}

constexpr std::array<int64_t, 5> A_ARR = {
    CalcA(16),
    CalcA(17),
    CalcA(18),
    CalcA(19),
    CalcA(20)
};

int main() {
    int32_t target;
    std::cout << "Enter World ID: ";
    if (!(std::cin >> target)) { return 0; }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Check 16-20
    for (int callIdx = 0; callIdx <= 4; callIdx++) {        
        const int64_t B = B_ARR[callIdx];
        const int64_t A = A_ARR[callIdx];
        
        int64_t diff = (int64_t)target - B;
        if (diff < 0) { diff += INT_MAX_VAL; }

        // Seed = (Target - B) * Inverse(A) % M
        const int64_t invA = ModInverse(A, INT_MAX_VAL);
        const int64_t foundSeed = (diff * invA) % INT_MAX_VAL;
        
        // Verify seed generates the target world id
        const int32_t realWorldID = GetWorldIDFull((int32_t)foundSeed);
        if (realWorldID == target) {
            std::cout << "Seed: " << foundSeed << "\n";
        }
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    auto delt = t2 - t1;
    std::cout << "Time taken: " << delt.count() << "ns\n";
    return 0;
}
