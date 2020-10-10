/*
Baseline 4 : Thuật toán Radix Sort tuần tự theo hướng dẫn đồ án
Cài đặt mảng lưu bin như file hướng dẫn. (hàng là các block, cột là các bin)
*/
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

// Parallel Radix Sort
void sortByDevice(const uint32_t * in, int n, uint32_t * out, int bklSize)
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
        for (int i = 0; i < gridSize.x; i++)
        {
            for (int j = 0; j < blockSize.x; j++)
            if (i * blockSize.x + j < n)
            {
                int bin = (src[i * blockSize.x + j] >> bit) & (nBins - 1);
                hist[i * nBins + bin]++;
            }
        }

        int previous = 0;
        for (int j = 0; j < nBins; j++){
            for (int i = 0; i < gridSize.x; i++)
            {
                histScan[i * nBins + j] = previous;
                previous = previous + hist[i * nBins + j];
            }
        }

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
    // Free memories
    free(hist);
    free(histScan);
}

// Radix Sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        bool useDevice=false, int blockSize=1)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
        printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
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
    
    // SORT BY DEVICE
    sort(in, n, out, true, blockSize);
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
