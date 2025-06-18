#include <stdio.h>
#include <hip/hip_runtime.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>

#ifndef ITER
#endif


template<typename T, int iter>
__global__ void vectorAdd(T *buf, const uint64_t n) {
    const uint32_t gid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    const uint32_t nThreads  = gridDim.x * blockDim.x;
    const int nEntriesPerThread = n / nThreads;
    const uint64_t maxOffset = nEntriesPerThread * nThreads;

    T *ptr;
    const T y = (T) 1.0;

    ptr = &buf[gid];
    T x = (T) 2.0;

    // For every vector element, its doing one read
    // For every vector element, its doing 2 * iter flops
    // For every thread, its doing one write
    
    for (uint64_t offset = 0; offset < maxOffset; offset += nThreads) {
        for (int j = 0; j < iter; j++) {
            x = ptr[offset] * x + y;
        }
    }
    ptr[0] = -x;
}

float getPerf(const uint64_t n, const int numExperiments, int blockSize, int gridSize, int numThreads, void *mem_a, float time) {
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);
    for (int run = 0; run < numExperiments; run++) {
        
        hipLaunchKernelGGL((vectorAdd<double, ITER>), dim3(gridSize), dim3(blockSize), 0, 0, (double *)mem_a, n);

    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&time, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return time;
}


char* get_parameter(const std::string & option, char ** begin, char ** end){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end)
  {
    return *itr;
  }
  return 0;
}

bool parameter_exists(const std::string& option, char** begin, char** end ){
  return std::find(begin, end, option) != end;
}


int main(int argc, char* argv[]) {

    uint64_t n = 134217728;
    uint64_t n_experiments = 10000;

    if (parameter_exists("--vector_size", argv, argv+argc)) n = std::stod(get_parameter("--vector_size", argv, argv+argc));
    if (parameter_exists("--n_experiments", argv, argv+argc)) n_experiments = std::stod(get_parameter("--n_experiments", argv, argv+argc));
    

    std::cout << "Vector size: " << n << std::endl;
    std::cout << "N experiments: " << n_experiments << std::endl;

    void *mem_a;
    hipMalloc(&mem_a, n * sizeof(double));

    int factor = n / 134217728;
    int blockSize = 256;
    int gridSize = 228 * 128 * factor;
    int numThreads = gridSize * blockSize;
    uint64_t flops = n * ITER * 2;
    uint64_t data_moved = (n + numThreads) * sizeof(double);

    std::cout << "Number of iterations: " << ITER << std::endl;
    std::cout << "Grid size: " << gridSize << std::endl;
    std::cout << "Block size: " << blockSize << std::endl;
    std::cout << "Number of threads: " << gridSize * blockSize << std::endl;
    std::cout << "Number of elements per thread: " << n / (gridSize * blockSize) << std::endl;
    std::cout << "Expected number of FP64 Flops: " << flops << std::endl << std::endl;

    float *runtimes;
    runtimes = (float *)malloc(n_experiments * sizeof(float));
    float time;
    hipLaunchKernelGGL((vectorAdd<double, ITER>), dim3(gridSize), dim3(blockSize), 0, 0, (double *)mem_a, n);
    hipDeviceSynchronize();
    time = getPerf(n, n_experiments, blockSize, gridSize, numThreads, mem_a, time);
    
    std::cout << std::endl;

    float avg_runtime = time / n_experiments;
    std::cout << "Average runtime: " << avg_runtime << " ms" << std::endl;
    
    std::string filename;
    float avg_runtime_seconds = avg_runtime / 1000;
    double tflops = static_cast<double>(flops) / 1e12;


    std::cout << "Arithmetic Intensity: " << static_cast<float>(flops) / data_moved << std::endl;
    std::cout << "Average runtime: " << avg_runtime << " ms" << std::endl;
    std::cout << "Average runtime: " << avg_runtime_seconds << " secs" << std::endl;
    std::cout << "TFLOPS/sec: " << tflops / avg_runtime_seconds << std::endl;
    std::cout << "BW [GB/sec]: " << data_moved / avg_runtime_seconds / 1e9 << std::endl;

    hipFree(mem_a);

    return 0;
}