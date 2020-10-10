%%cu
/*
Radix sort final
*/
#include <stdio.h>
#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

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
void sortBaseline(const uint32_t * in, int n, uint32_t * out, int blkSize)
{
    // TODO
    int nBits = 1; // Assume: nBits in {1, 2, 4, 8, 16}
    int nBins = 1 << nBits; // 2^nBits

    dim3 blockSize(blkSize); // block size
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

__global__ void scatter(uint32_t * in, int n, int nBits, int bit, int nBins, int *histScan, uint32_t * out)
{
    extern __shared__ int s_data[];
    int * s_in = s_data;
    int * s_hist = (int *)&s_in[blockDim.x];
    int * dst = (int *)&s_hist[blockDim.x];
    int * dst_ori = (int *)&dst[blockDim.x];
    int * startIndex = (int *)&dst_ori[blockDim.x];
    int * hist = (int *)&startIndex[blockDim.x];
    int * scan = (int *)&hist[blockDim.x];

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (id < n){
        s_in[threadIdx.x] = in[id];
        s_hist[threadIdx.x] = (s_in[threadIdx.x] >> bit) & (nBins - 1); // get bit
    }
    else {
        s_hist[threadIdx.x] = nBins - 1;
    }
    // Step 1 : sort radix with k = 1
    for (int b = 0; b < nBits; b++){
        // compute hist
        hist[threadIdx.x] = (s_hist[threadIdx.x] >> b) & 1;
        __syncthreads();

        // scan
        if (threadIdx.x == 0){
            scan[0] = 0;
        }
        else {
            scan[threadIdx.x] = hist[threadIdx.x - 1];
        }
        __syncthreads();

        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int val = 0;
            if (threadIdx.x >= stride){
                val = scan[threadIdx.x - stride];
            }
            __syncthreads();

            scan[threadIdx.x] += val;
            __syncthreads();
        }
        __syncthreads();

        // scatter
        int nZeros = blockDim.x - scan[blockDim.x - 1] - hist[blockDim.x - 1];
        int rank = 0;
        if (hist[threadIdx.x] == 0){
            rank = threadIdx.x - scan[threadIdx.x];
        }
        else{
            rank = nZeros + scan[threadIdx.x];
        }
        dst[rank] = s_hist[threadIdx.x];
        dst_ori[rank] = s_in[threadIdx.x];
        __syncthreads();

        // copy or swap
        s_hist[threadIdx.x] = dst[threadIdx.x];
        s_in[threadIdx.x] = dst_ori[threadIdx.x];
    }
    __syncthreads();

    // Step 2
    if (threadIdx.x == 0){
        startIndex[s_hist[0]] = 0;
    }
    else
    {
        if (s_hist[threadIdx.x] != s_hist[threadIdx.x - 1]){
            startIndex[s_hist[threadIdx.x]] = threadIdx.x;
        }
    }
    __syncthreads();

    // Step 3
    if (id < n)
    {
        int preRank = threadIdx.x - startIndex[s_hist[threadIdx.x]];
        int bin = ((s_in[threadIdx.x] >> bit) & (nBins - 1));
        int scan = histScan[bin * gridDim.x + blockIdx.x];
        int rank = scan + preRank;
        out[rank] = s_in[threadIdx.x];
    }
}

// Parallel Radix Sort
void sortByDevice(const uint32_t * in, int n, uint32_t * out, int blkSize)
{
    // TODO
    int nBits = 8; // Assume: nBits in {1, 2, 4, 8, 16}
    int nBins = 1 << nBits; // 2^nBits

    dim3 blockSize(blkSize);
    dim3 gridHistSize((n - 1) / blockSize.x + 1);
    dim3 gridScanSize((nBins * gridHistSize.x - 1) / blockSize.x + 1);
    dim3 gridScatterSize((n - 1) / blockSize.x + 1);
    
    int * blkSums = (int *)malloc(gridScanSize.x * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));

    int *d_hist, *d_scan, *d_blkSums;
    CHECK(cudaMalloc(&d_hist, nBins * gridHistSize.x * sizeof(int)));
    CHECK(cudaMalloc(&d_scan, nBins * gridHistSize.x * sizeof(int)));
    CHECK(cudaMalloc(&d_blkSums, gridScanSize.x * sizeof(int)));

    uint32_t * d_src, *d_dst;
    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_src, src, n * sizeof(uint32_t), cudaMemcpyHostToDevice)); // copy to device
    CHECK(cudaMalloc(&d_dst, n * sizeof(uint32_t)));

    size_t smemBytes = blockSize.x * sizeof(int);
    size_t smemScatterBytes = blockSize.x * 7 * sizeof(int);
    
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
        // Compute "hist" of the current digit
        CHECK(cudaMemset(d_scan, 0, nBins * gridHistSize.x * sizeof(int)));
        computeHistogram<<<gridHistSize, blockSize, smemBytes>>>(d_src, n, d_scan, nBins, bit);
        cudaDeviceSynchronize();
        
        // Scan
        scanExclusiveBlk<<<gridScanSize, blockSize, smemBytes>>>(d_scan, nBins * gridHistSize.x, d_scan, d_blkSums);
        cudaDeviceSynchronize();
        CHECK(cudaMemcpy(blkSums, d_blkSums, gridScanSize.x * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 1; i < gridScanSize.x; i++){
            blkSums[i] += blkSums[i - 1];
        }
        CHECK(cudaMemcpy(d_blkSums, blkSums, gridScanSize.x * sizeof(int), cudaMemcpyHostToDevice));
        computeHistScan<<<gridScanSize, blockSize>>>(d_scan, nBins * gridHistSize.x, d_blkSums);
        cudaDeviceSynchronize();

        // Scatter
        scatter<<<gridScatterSize, blockSize, smemScatterBytes>>>(d_src, n, nBits, bit, nBins, d_scan, d_dst);
        cudaDeviceSynchronize();
        
        // Swap "src" and "dst"
        uint32_t * temp = d_src;
        d_src = d_dst;
        d_dst = temp;
    }
    // Copy "d_src" to "out"
    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_scan));
    CHECK(cudaFree(d_blkSums));
    
    free(blkSums);
}

// #########################################################
// Radix sort by thrust
// #########################################################
void sortWithThrust(const uint32_t * in, int n, uint32_t * out)
{
    thrust::device_vector<uint32_t> dv_out(in, in + n);
    thrust::sort(dv_out.begin(), dv_out.end());
    thrust::copy(dv_out.begin(), dv_out.end(), out);
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
    else if (type == 2) // use device
    {
        printf("\nRadix Sort by device\n");
        sortByDevice(in, n, out, blockSize);
    }else {
        printf("\nRadix Sort with thrust\n");
        sortWithThrust(in, n, out);
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
    uint32_t * out_thrust = (uint32_t *)malloc(bytes);
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
    // Calcute avg
    GpuTimer timer; 
    
    float avg_time = 0;
    int loop = 16;
    printf("\nRadix sort by device avg\n");
    for (int i = 0; i < loop; i++){
        timer.Start();
        sort(in, n, out, true, blockSize, 2);
        timer.Stop();
        avg_time += timer.Elapsed();
    }
    printf("AVG TIME: %.f ms\n", avg_time / loop);
    checkCorrectness(out, correctOut, n);

    // SORT BY THRUST
    sort(in, n, out_thrust, true, blockSize, 3);
    checkCorrectness(out_thrust, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
