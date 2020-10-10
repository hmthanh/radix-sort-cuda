/*
Baseline 4.2 : Song song 2 bước tính hist và scan*/
#include <stdio.h>
#include <stdint.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// Sequential Radix Sort
void sortByHost(const uint32_t * in, int n,
                uint32_t * out)
{

    int nBits = 4; // Assume: nBits in {1, 2, 4, 8, 16}
    int nBins = 1 << nBits; // 2^nBits

    int * hist = (int *)malloc(nBins * sizeof(int));
    int * histScan = (int *)malloc(nBins * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
    // (Each digit consists of nBits bit)
    // In each loop, sort elements according to the current digit from src to dst 
    // (using STABLE counting sort)
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
        // TODO: Compute histogram
        memset(hist, 0, nBins * sizeof(int));
        for (int i = 0; i < n; i++)
        {
            int bin = (src[i] >> bit) & (nBins - 1);
            hist[bin]++;
        }

        // TODO: Scan histogram (exclusively)
        histScan[0] = 0;
        for (int bin = 1; bin < nBins; bin++)
            histScan[bin] = histScan[bin - 1] + hist[bin - 1];

        // TODO: Scatter elements to correct locations
        for (int i = 0; i < n; i++)
        {
            int bin = (src[i] >> bit) & (nBins - 1);
            dst[histScan[bin]] = src[i];
            histScan[bin]++;
        }
        
        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    // Copy result to out
   memcpy(out, src, n * sizeof(uint32_t)); 
}

// #########################################################
// Baseline
void sortBaseline(const uint32_t * in, int n, uint32_t * out, int bklSize)
{
    // TODO
    int nBits = 1; // Assume: nBits in {1, 2, 4, 8, 16}
    int nBins = 1 << nBits; // 2^nBits

    dim3 blockSize(bklSize); // block size
    dim3 gridSize((n - 1) / blockSize.x + 1); // grid size

    int * hist = (int *)malloc(nBins * gridSize.x * sizeof(int));
    int *histScan = (int * )malloc(nBins * gridSize.x * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
        memset(hist, 0, nBins * gridSize.x * sizeof(int));
        // compute historgram
        for (int i = 0; i < gridSize.x; i++)
        {
            for (int j = 0; j < blockSize.x; j++)
            if (i * blockSize.x + j < n)
            {
                int bin = (src[i * blockSize.x + j] >> bit) & (nBins - 1);
                hist[i * nBins + bin]++;
            }
        }

        // compute scan
        int previous = 0;
        for (int j = 0; j < nBins; j++){
            for (int i = 0; i < gridSize.x; i++)
            {
                histScan[i * nBins + j] = previous;
                previous = previous + hist[i * nBins + j];
            }
        }

        // scatter
        for (int i = 0; i < gridSize.x; i++)
        {
            for (int j = 0; j < blockSize.x; j++)
            {
                int id = i * blockSize.x + j;
                if (id < n)
                {
                    int bin = i * nBins + ((src[id] >> bit) & (nBins - 1));
                    dst[histScan[bin]] = src[id];
                    histScan[bin]++;
                }
            }
        }
        uint32_t * temp = src;
        src = dst;
        dst = temp; 
    }

    memcpy(out, src, n * sizeof(uint32_t));
    free(hist);
    free(histScan);
}

// #########################################################
// Radix sort by device
// #########################################################
// Histogram kernel
__global__ void computeHistogram(uint32_t * in, int n, int * hist, int nBins, int bit)
{
    // TODO
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int bin = (in[i] >> bit) & (nBins - 1);
        atomicAdd(&hist[bin * gridDim.x + blockIdx.x], 1);
    }
}

// scan kernel
__global__ void scanExclusiveBlk(int * in, int n, int * out, int * blkSums)
{   
    // TODO
    extern __shared__ int s_data[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < n){
        s_data[threadIdx.x] = in[i - 1];
    }
    else{
        s_data[threadIdx.x] = 0;
    }
    __syncthreads();
    
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int val = 0;
        if (threadIdx.x >= stride){
            val = s_data[threadIdx.x - stride];
        }
        __syncthreads();

        s_data[threadIdx.x] += val;
        __syncthreads();
    }
    
    if (i < n){
        out[i] = s_data[threadIdx.x];
    }
    if (blkSums != NULL){
        blkSums[blockIdx.x] = s_data[blockDim.x - 1];
    }
}

__global__ void computeHistScan(int * in, int n, int* blkSums)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && blockIdx.x > 0)
        in[i] += blkSums[blockIdx.x - 1];
}

// Parallel Radix Sort
void sortByDevice(const uint32_t * in, int n, uint32_t * out, int bklSize)
{
    // TODO
    int nBits = 1; // Assume: nBits in {1, 2, 4, 8, 16}
    int nBins = 1 << nBits; // 2^nBits

    dim3 blockSize(bklSize);
    dim3 gridHistSize((n - 1) / blockSize.x + 1);
    dim3 gridScanSize((nBins * gridHistSize.x - 1) / blockSize.x + 1);
    
    int * scan = (int * )malloc(nBins * gridHistSize.x * sizeof(int));
    int * blkSums = (int *)malloc(gridScanSize.x * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    uint32_t * d_src;
    int *d_hist, *d_scan, *d_blkSums;

    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_hist, nBins * gridHistSize.x * sizeof(int)));
    CHECK(cudaMalloc(&d_scan, nBins * gridHistSize.x * sizeof(int)));
    CHECK(cudaMalloc(&d_blkSums, gridScanSize.x * sizeof(int)));

    size_t smemHistBytes = nBins * sizeof(int); 
    size_t smemScanBytes = blockSize.x * sizeof(int);
    
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
        // compute historgram
        CHECK(cudaMemcpy(d_src, src, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemset(d_hist, 0, nBins * gridHistSize.x * sizeof(int)));
        computeHistogram<<<gridHistSize, blockSize, smemHistBytes>>>(d_src, n, d_hist, nBins, bit);
        cudaDeviceSynchronize();

        // compute scan
        scanExclusiveBlk<<<gridScanSize, blockSize, smemScanBytes>>>(d_hist, nBins * gridHistSize.x, d_scan, d_blkSums);
        cudaDeviceSynchronize();
        
        CHECK(cudaMemcpy(blkSums, d_blkSums, gridScanSize.x * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 1; i < gridScanSize.x; i++){
            blkSums[i] += blkSums[i - 1];
        }
        CHECK(cudaMemcpy(d_blkSums, blkSums, gridScanSize.x * sizeof(int), cudaMemcpyHostToDevice));
        computeHistScan<<<gridScanSize, blockSize>>>(d_scan, nBins * gridHistSize.x, d_blkSums);
        cudaDeviceSynchronize();
        CHECK(cudaMemcpy(scan, d_scan, nBins * gridHistSize.x * sizeof(int), cudaMemcpyDeviceToHost)); 
        
        // Scatter
        for (int i = 0; i < n ; i++)
        {
            int bin = i / blockSize.x + ((src[i] >> bit) & (nBins - 1)) * gridHistSize.x;
            dst[scan[bin]] = src[i];
            scan[bin]++;
        }
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }
    // Copy result to "out"
    memcpy(out, src, n * sizeof(uint32_t));

    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_scan));
    CHECK(cudaFree(d_blkSums));
    
    free(blkSums);
    free(scan);
}

// Radix Sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        bool useDevice=false, int blockSize=1, int type=0)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
        printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
    }
    else if (type == 1){ // Baseline
        printf("\nBaseline Radix Sort (highlight)\n");
        sortBaseline(in, n, out, blockSize);
    }
    else // use device
    {
        printf("\nRadix Sort by device\n");
        sortByDevice(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    int n = (1 << 24) + 1; // For test by eye
    //int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out_baseline = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
    {
        in[i] = rand() % 255; // For test by eye
        //in[i] = rand();
    }
    // printArray(in, n); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY HOST
    sort(in, n, correctOut);
    // printArray(correctOut, n);

    // SORT BY BASELINE
    sort(in, n, out_baseline, true, blockSize, 1);
    checkCorrectness(out_baseline, correctOut, n);
    
    // SORT BY DEVICE
    sort(in, n, out, true, blockSize, 2);
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
