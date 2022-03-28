#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <chrono>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


namespace kernels{
//pretty trivial kernel that has as many threads in the x-dimension of a block as output channels
//it is used as fallback!
    template<typename scalar_t>
    __global__ void cond_mul(
            const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
            const torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits, size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> weights,
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> bias,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output) {
        //extern __shared__ uint8_t shared[];
        const int ind = blockIdx.x * blockDim.y + threadIdx.y;
        const int m = weights.size(1);//m input channels
        const int n = weights.size(2);//n output channels
        const int in = threadIdx.x;//index for the output is the thread index

        //Note that storing weights + accumulators in shared memory does not necessarily yield better results.
        //especially if the indices are very random
        if (ind >= input.size(0)) {
            return;
        }

        const int ind_w = inds[ind];
#ifdef FORCE_DOUBLE
        double result = bias[ind_w][0][in];
#else
        scalar_t result = bias[ind_w][0][in];
#endif
        for (int im = 0; im < m; im++) {
            result += input[ind][im] * weights[ind_w][im][in];
        }
        output[ind][in] = result;
    }

// (UNUSED)
// Buffering input and output in smem. Demonstration how inefficient this is! (3 times as slow)
// It is meant to run on inputs of m and n being multiple of 32
// Due to inefficient memory access it is 3 times slower than the other optimized paths working on
// per-warp versions.
// const dim3 threads3(256); //256 threads per block
// size_t smem_size = 16384;
// size_t pixel_per_block = smem_size / (sizeof(scalar_t) * (m + n) + sizeof(scalar_t));
    template<typename scalar_t>
    __global__ void cond_mul_buffer_io_kernel(
            const int cache_count, // how many inputs/outputs and indices are cached in shared memory
            const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
            const torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits, size_t> indices,
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> weights,
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> bias,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output) {

        const int overall_pix_count = input.size(0);
        const int tid = threadIdx.x; //index within warp!
        const int n = output.size(1);
        const int m = input.size(1);
        const int base_ind = blockIdx.x * cache_count;
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            //printf("m %d, n %d \n", m, n);
        }
        extern __shared__ uint8_t shared[];
        scalar_t *in = (scalar_t *) shared;
        scalar_t *out = &in[cache_count * m];
        int *inds = (int *) (&out[cache_count * n]);

        //load input:
        int i = tid;
        while (i < m * cache_count) {
            int ch = i % m;
            int ind = i / m; // pixel index(local
            if (ind + base_ind >= overall_pix_count) {
                break;
            }
            in[ind * m + ch] = input[ind + base_ind][ch];
            i += blockDim.x;
        }
        //load inds:
        i = tid;
        while (i < cache_count) {
            if (base_ind + i >= overall_pix_count) {
                break;
            }
            inds[i] = indices[base_ind + i];
            i += blockDim.x;
        }
        //setup output buffer
        i = tid;
        while (i < n * cache_count) {
            int ch = i % n;
            int ind = i / n; // pixel index(local
            if (ind + base_ind >= overall_pix_count) {
                break;
            }
            out[ind * n + ch] = 0;
            i += blockDim.x;
        }

        __syncthreads();
        //do processing
        //this part eats up a lot of compute (actually it is 3 times as slow as the m=32 to n=32 path)
        int ind_w = -1;
        float w;
        i = tid;
        while (i < m * n) {
            int ind = 0;
            //for a transposed weight tensor:
            // (just transposing the weight tensor via
            // torch::transpose(weights, 1, 2) and weightsT = weightsT.contiguous();
            // takes 1.5 times as long as a good implementation)
            //int ind_m = i % m;
            //int ind_n = i / m;
            int ind_m = i / n;
            int ind_n = i % n;

            while (ind < cache_count &&
                   (ind + base_ind) < overall_pix_count) {
                if (ind_w != inds[ind]) {
                    ind_w = inds[ind];
                    w = weights[ind_w][ind_m][ind_n];
                }
                scalar_t update = w * in[ind * m + ind_m];
                //out[ind * n + ind_n] += update; (not using the atomic is wrong and only 1/6th faster)
                atomicAdd(&out[ind * n + ind_n], update);
                ind++;
            }
            i += blockDim.x;
        }

        //write out!
        __syncthreads();
        i = tid;
        while (i < n * cache_count) {
            int ch = i % n;
            int ind = i / n; // pixel index(local
            if (ind + base_ind >= overall_pix_count) {
                break;
            }
            int ind_w = inds[ind];
            output[ind + base_ind][ch] = out[ind * n + ch] + bias[ind_w][0][ch];
            i += blockDim.x;
        }
    }


/*
* Memory bandwith is utilized between 75%(4 consecutive shared weights) and 90% (random weights) at n = 32
* 55 registers for n = 16 56 registers for n = 32
* with shared memory of 4224 bytes for one block there is only a occupancy of 23%
* to improve this one would need to reduce the use of shared memory by a lot
* but even when not adding more register usage the occupancy would not go higher than 50%
* thats due to the 32 threads per block. Warp level synchronization is free though...
* to get full utilization we need to get down to 32 registers per thread & have at least 64 threads per block (two warps)
* most GPUs allow for 64 warps but only 32 blocks to be managed simultaneously
*
* 100% occupancy can only be reached with 64 threads per block or more
* 2048 bytes shared memory ... or less
* and lass than 32 registers utilized
*
* for n==1 we only have 22 registers per thread but the same 23% occupancy and around 50% (not fully random) percent
* of utilized memory bandwith
* TODO:
* Idea to reach better occupancy (50% with 32 threads for one block) for fewer output channels (lets talk about n=1):
* no shared memory usage: reduction after every multiplication (i know this takes 6 cycles with 32, 16, 8, 4, 2, 1
* active threads each). The results would be accumulated up in one result register and wouldn't need any shared memory.
* For n=2 its 2 output registers but the accumulation would only take 5 cycles with 32,16,8,4,2 active threads.
* for n=4 its 4 output registers with only 4 accumulation cycles after each round of multiplication
* For n=8 its 8 output registers (remember its also 8 registers for weights so this might pay off but its already quite
* big
* For n=16 the register count probably would go towards 64 so it could still make sense (50% utilization max) but we have high memory bandwith
* with that already
*
* The issue with n=1 is that we will probably only have 32 inputs and therefore 64 threads per block is hard to achieve
* for n=2 and higher it already is easier to utilize 64 threads even with 32 inputs but still pretty hard to achieve
* the easiest way would be to have multiple of 32 as input and each warp handle one of them.
* The other way to achieve 64 threads would be to have each set of 32 threads work on one pixel... that might be way more efficient
*
* how to use nvidia profiler: (profiling needs sudo rights but python doesn't find modules with sudo)
source venv/bin/activate
sudo env PATH=$PATH nvprof --analysis-metrics -f -o prof.nvvp venv/bin/python test_cuda_cond_mul.py
nvvp prof.nvvp
*/

//Kernel for m multiple of 32 and n being one of 1, 2, 4, 8, 16, 32
//weights are not reloaded at every new pixel
    template<typename scalar_t, int m_per_warp, int n_per_set> //TODO: this one is in use
    __global__ void
    cond_mul_cache_output(//cond_mul_cuda_forward_deep_reuse32_kernel(//cond_mul_cache_output_kernel
            const int parts_in, //sets as in the template and parts as in this parameter are the same TODO: rename either set or part!
            const int parts_out,//since the accumulator (shared memory) stores results for the
            // n_per_set outputs (+ groups) for the next 32 pixel. Only with parts > 1 more than 32 outputs
            // n>32 can be achieved. This also means that inputs are being read multiple times.
            // Actually parts_out is bullshit and anyway set to 1! TODO: remove if not needed at all
            const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input,
            const torch::PackedTensorAccessor<int32_t, 1, torch::RestrictPtrTraits, size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> weights,
            const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> bias,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output) {
        extern __shared__ uint8_t shared[];

        //TODO: rethink the meaning of threadIdx.x! It should be n

        const int base_ind = 32 * blockIdx.x; // the starting pixel for this block
        const int overall_samples = input.size(0);
        //const int m = weights.size(1); //m... how many input channels alltogether
        //const int n = blockDim.x;//weights.size(2); // should be same asblockDim.x
        //const int in = threadIdx.x;
        const int threads = 32; // threads in one block/warp (always 32)
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int simultaneous_pix = m_per_warp; //threads / n // same as blockDim.y
        //const int columns_per_warp = m_per_warp; //threads / n
        const int warps_per_weight_set = n_per_set; // n

        //buffer for the weights of the n outputs (for big n this will use too many registers)
        scalar_t w[n_per_set];

        scalar_t *acc = (scalar_t *) &shared[0];
        //load indices
        int weight_index;
        if ((base_ind + tid) < overall_samples) { //also check if we are overstepping the boundaries
            //load inds for the next 32 pixel
            weight_index = inds[base_ind + tid];
        }
        //if n <=32 (n = n_per_set) we can load enough weights / have enough shared memory for the results of the next 32
        // outputs. If n is multiple of 32 we could just do it in multiple blocks/parts.
        for (int l = 0; l < parts_out; l++) {
            //clear the accumulator
            for (int i = 0; i < threads + 1; i++) {
                acc[i * threads + tid] = 0;
            }

            //int im = tid;
            //the input/weights of one pixel need to be split into parts
            for (int j = 0; j < parts_in; j++) {
                // load the next 32 values for 32 pixel:
                int last_ind = -1;
                for (int k = 0; k < 32; k++) {
                    int pix = base_ind + k;
                    if (pix >= overall_samples) {
                        break;
                    }
                    scalar_t v = input[pix][32 * j + tid];
                    int ind_w = __shfl_sync(0xffffffff, weight_index, k);

                    // in case of a new index for loading, we reload new weights
                    if (ind_w != last_ind) {

                        //we load a set of weights for n_per_set outputs
                        for (int i = 0; i < n_per_set; i++) {
                            int im = j * 32 + i * blockDim.y + threadIdx.y; //index along m direction of weight / input
                            w[i] = weights[ind_w][im][threadIdx.x + l * 32];
                        }
                        last_ind = ind_w;
                    }
                    scalar_t result = 0;

                    //v holds 32 values of the input features of this block.
                    for (int i = 0; i < n_per_set; i++) {
                        result += w[i] *
                                  __shfl_sync(0xffffffff, v, i * blockDim.y + threadIdx.y);
                    }
                    //the complex indexing scheme with + n_per_set should prevent page conflicts
                    acc[n_per_set * k +
                        threadIdx.y * (32 * n_per_set + n_per_set) +
                        threadIdx.x] += result;
                    //TODO: can we use more than one warp if we go atomic here?

                }
            }
            // the warp should be in sync anyway (except for turing gpus and newer there it might differ!!!)
            __syncwarp();
            // n_per_set also means that one set has the size 32/n_per_set.
            // With 32 threads this means we need n_per_set iterations.
            for (int i = 0; i < n_per_set; i++) {
                int pix_local = i * blockDim.y + threadIdx.y;
                int pix = base_ind + pix_local;
                if (pix >= overall_samples) {
                    return;
                }
                int ind_w = __shfl_sync(0xffffffff, weight_index, pix_local);
                scalar_t accu = bias[ind_w][0][threadIdx.x];

                //iterate over all the accumulators for this set of values
                for (int j = 0; j < simultaneous_pix; j++) {
                    accu += acc[j * (n_per_set * 32 + n_per_set) +
                                n_per_set * (threadIdx.y + i * blockDim.y) +
                                threadIdx.x];

                }
                output[pix][threadIdx.x + l * 32] = accu;

                //TODO: find out if this one is necessary!!
                __syncwarp(); // the warp should be in sync anyway (except for turing gpus... there it might differ!!!)
            }
        }

    }


/*
* all what has been written in the comments of the function above is applied here...
* it improves performance for n = 1, 2 and 4! For bigger n, it uses too many registers.
* to reach 100% occupancy it would be ideal to have 64 threads per block not more than 2048 bytes per block.
* the limit for shared memory is reached for n = 8 due to the additional index we use to prevent bank conflicts
* (otherwise it could be at n=16).
*/
    template <typename scalar_t,int m_per_warp,int n> //TODO: THIS IS USED
    __global__ void cond_mul_cache_output_high_occupancy( //cond_mul_cuda_forward_deep_reuse32_high_occupancy_kernel // cond_mul_cache_output_high_occupancy
            const int parts, //the parts in which the input/weights get loaded. for each part all outputs are calculated simultanously
            const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
            const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
            torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
        extern __shared__ uint8_t shared[];

        //each block has 64 threads in this version, each warp of 32 is processing 32 "pixel"
        const int base_ind = 64 * blockIdx.x + 32*threadIdx.z; // the starting pixel for this block
        const int overall_samples = input.size(0);
        const int m = weights.size(1);
        const int threads = 32; // threads in one block/warp (always 32)
        const int tid = threadIdx.y * blockDim.x + threadIdx.x; //thread index within a warp
        const int simultaneous_pix = m_per_warp; //threads / n // same as blockDim.y
        //const int columns_per_warp = m_per_warp; //threads / n
        const int warps_per_weight_set = n; // n

        //lets store weights for the processing in a register vairable
        scalar_t w[n];
        //the accumulator should be exactly n*32 long
#ifdef FORCE_DOUBLE
        double *acc = (double*)&shared[sizeof(double) * 32 * n * threadIdx.z];
#else
        scalar_t *acc = (scalar_t*)&shared[sizeof(scalar_t) * 32 * n * threadIdx.z]; //TODO: find out if threadIdx.z really is used!!!
#endif
        //load indices
        int weight_index;
        if( (base_ind + tid) < overall_samples){ //also check if we are overstepping the boundaries
            //load inds for the next 32 pixel
            weight_index = inds[base_ind + tid];
        }
        //scalar_t v[32];
        //clear the accumulator
        for(int i = 0;i < n; i++){
            acc[i * 32 + tid] = 0;
        }


        //the input/weights of each pixel need to be split into parts (of size 32)
        //TODO: think why we are buffering the weights and not input values + output accumulators?
        for(int j = 0; j < parts;j++){
            // load the next 32 values for 32 pixel:
            int last_ind = -1;
            for(int k = 0; k < 32; k++){
                int pix = base_ind + k;

                //stop when we are out of pixel!
                if( pix >= overall_samples){
                    break;
                }
                //load the value we need
                scalar_t v = input[pix][32 * j + tid];
                //get the current index we are at from other threads within the warp.(loaded in the beginning)
                int ind_w = __shfl_sync(0xffffffff, weight_index, k);
                if(ind_w != last_ind){
                    for(int i=0;i<n;i++){
                        //im consists of a few things
                        //the threadIdx.y... the "lane" of weights/values since only blockDim.y input channels can be processed at once
                        //i... as blockDim.y "lanes" are processed at once a block of 32 needs to be split into n (blockDim.x) parts
                        //j... the part we are at.
                        int im = j * 32 + i * blockDim.y + threadIdx.y; //index along m direction of weight / input
                        w[i] = weights[ind_w][im][threadIdx.x];
                    }
                    last_ind = ind_w;
                }
                scalar_t result = 0;
                for(int i=0;i<n;i++){
                    result += w[i] * __shfl_sync(0xffffffff, v, i * blockDim.y + threadIdx.y);
                }

                //proper reduction, we don't need to close down threads since they all are synced
                // (the unnecessary additions are not too bad)
                for (int offset = 16; offset >= n; offset /= 2)
                    result += __shfl_down_sync(0xffffffff, result, offset);

                if(tid < n){
                    //store result in accumulator (shared memory
                    acc[tid + k * n] += result;
                }

            }
        }
        __syncwarp(); // the warp should be in sync anyway (except for turing gpus... there it might differ!!!)

        for(int i=0;i<n;i++){
            int pix_local = i * blockDim.y + threadIdx.y;
            int pix = base_ind + pix_local;
            if(pix >= overall_samples){
                return;
            }
            int ind_w = __shfl_sync(0xffffffff, weight_index, pix_local);
            output[pix][threadIdx.x] = bias[ind_w][0][threadIdx.x] + acc[pix_local * n + threadIdx.x];
        }

    }

//TODO: fix the function names beginning with here!

//again we want to have maximum occupancy here:
//maximum 2048 bytes of shared memory per block
//maximum 32registers per thread
// 64 threads per block
// To always keep the shared memory at its full potential one could store more than 32 pixel simultaneously
// calculation for shared memory:
// 1 channel input: 512 vals buffer -> 2048 bytes (weight index array would take 16 registers)
// 2 channel input: 256 vals buffer -> 2048 bytes (weight index array would take 8 registers)
// 4 channel input: 128 vals buffer -> 2048 bytes (weight index array would take 4 registers)
// 8 channel input: 64 vals buffer -> 2048 bytes (weight index array would take 2 registers)
// 16 channel input: 32 vals buffer -> 2048 bytes (weight index takes 1 register)
// 32 channel input: 16 vals buffer -> 2048 bytes (weight index array takes 1 register for half of the warps
// for the applications this was designed, we don't at best have 608 pixel per line with 16 different classes
// this gives 38 consecutive pixel with same class. Probably the improvements of implementing an index array are minor.
//TODO: maybe also make this work with more than a 32 pixel buffer. (it doesn't pay
//TODO: for the very common case of m=16 and n=1 it would be nice to actually calculate multiple
// pixel outputs at the same time! (rename pixel_per_warp to something more fitting since we are looping trough pixel)
// also, the neighbouring pixel should be looped as in working on pixel in the order (0,16) (1,17) (2,18)

//TODO: m_per_warp and n_per_warp is the same as blockDim.x and blockDim.y
    template <typename scalar_t, int pixel_per_warp> //TODO:THIS IS IN USE
    __global__ void cond_mul_cache_input_few_in(//cond_mul_cuda_forward_deep_reuse32_few_in_many_out(//cond_mul_cache_input_few_in
            const int loop_pixel, //in the normal case we would loop over pixel_per_warp pixel since we go 1 pixel at a time TODO: rename loop_out_pixel
            const int simultaneous_out_pixel, //if we have very few output channels we can handle multiple pixel at once
            const int simultaneous_out_channels, //TODO: simultaneous out_channels can be derived from blockDim.y (blockDim.y = simultaneous_out_pixel * simultaneous_out_channels)
            const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
            const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
            torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
        //return;
        //shared memory needs to hold pixel_per_warp * input_channels pixel.
        extern __shared__ uint8_t shared[];
        const int base_ind = pixel_per_warp * (blockIdx.x * blockDim.z + threadIdx.z); // the starting pixel for this block
        const int overall_samples = input.size(0);
        const int m_per_warp = blockDim.x; // same as m

        const int n = output.size(1);
        //const int m = input.size(1);//todo:would m_per_warp be the same minus the cost of reading it?

        //split up the available output channels (threadIdx.y) into the simultaneous pixel into the simultaneously
        // worked channels (ind_y_sub) and the simultaneously worked pixel (ind_y_main)
        // e.g. m=2 -> blockDim = (2,16,2) and 32 pixel_per_warp -> simultaneous_out_channels = 16
        // ind_y goes 0,1,2,3,4,5,6,7, 08,09,10,11,12,13,14,15
        // ind_y_main 0,0,0,0,0,0,0,0, 01,01,01,01,01,01,01,01 // main is used for the input channels m
        // ind_y_sub  0,1,2,3,4,5,6,7, 00,01,02,03,04,05,06,07 // sub is used for the input dimension n
        int ind_y_main = threadIdx.y / simultaneous_out_channels;
        int ind_y_sub = threadIdx.y % simultaneous_out_channels;
        //TODO: is this the same as


        //buffer the input
        scalar_t* buf_in = (scalar_t*)&shared[sizeof(scalar_t) * threadIdx.z * pixel_per_warp * m_per_warp];
        //printf("offset shared: %d \n", (threadIdx.z * pixel_per_warp * m_per_warp));
        const int tid = threadIdx.y * blockDim.x + threadIdx.x; //thread index within a warp

        //TODO: if we want to buffer more than 32 pixel (could make sense) we want weight index to be an array.
        //get the current index
        int weight_index;
        if( (base_ind + tid) < overall_samples){ //also check if we are overstepping the boundaries
            //load inds for the next 32 pixel
            weight_index = inds[base_ind + tid];
        }

        //load the the input of the next pixel_per_warp pixel into shared memory
        //parts_in has to be selected accordingly when calling this kernel
        for(int i=0;i<pixel_per_warp;i+=blockDim.y){
            //blockDim.y tells me how many pixel we load simultaneously
            //blockDim.x how many weights are loaded for each pixel (should be the size of m)
            const int ind = base_ind + threadIdx.y + i;

            if(ind >= overall_samples){
                break;
            }
            //we don't need to trick around here since it doesn't seem like we are provoking bank conflicts
            //tid goes from 0:32t warps, i goes for e.g. m=16 ->blockDim = (16,2,2) and 32 pixel_per_warp
            // 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, ...30 multiplied by blockDim.y this makes 0, 32, 64, 96 etc. till
            // 480 (512-32) Note that 32 pixel_per_warp are using too much shared memory for m=16... in reality it would be
            // 16 for m=16 and 8 for m=32
            buf_in[tid + i * blockDim.x] = input[ind][threadIdx.x];
        }
        //return;
        //accumulate over all the elements and create outputs:
        //TODO: parts_out could be skipped if we go for something like this:
        // for(int i=0; i < n; i+=simultaneous_out_channels)
        //for(int i=0; i < parts_out; i++){
        for(int i=0; i < n; i+=simultaneous_out_channels){
            //the index for the output channel
            //int ind_n = ind_y_sub + i * simultaneous_out_channels;//threadIdx.y + i * n_per_warp;
            int ind_n = ind_y_sub + i;
            int ind_m = threadIdx.x;
            //iterate over the next pixel_per_warp (32 in most cases) pixel!
            int ind_w_last = -1;
            scalar_t b = 0;
            scalar_t w = 0;
            //TODO: if we want ot have a buffer for more than 32 pixel we need to take care of this!
            for(int j=0; j<loop_pixel;j++){ //iterate over the 16/32 pixel
                //int pix_ind = base_ind + j; //TODO: fix this for working multiple pixel simultaneously:
                int pix_ind = base_ind + j + loop_pixel * ind_y_main;
                //pix_ind = base_ind + j + simultaneous_out_pixel * ind_y_main; //TODO: find out if loop_pixel / simultaneous out_pixel does the same here?
                //if(pix_ind >= overall_samples){
                //TODO: now we have one issue: not all pixel of this group work on one pixel...
                // we can't just break here and ignore all the potential shfl_syncs that will never be achieved!!!!!!
                //    break; // obviously we don't go further than the amount of samples we need
                //}
                int ind_w = __shfl_sync(0xffffffff, weight_index, j + ind_y_main * loop_pixel);
                if( ind_w_last != ind_w &&
                    (pix_ind < overall_samples)){ //prevent readout at invalid pixel!
                    ind_w_last = ind_w;
                    if(ind_n < n){ //in cases we have fewer outputs than we need threads for (e.g. 1 input channel and 16 output channels)
                        //printf("weight index, m, n%d, %d, %d\n", ind_w, ind_m, ind_n);
                        b = bias[ind_w][0][ind_n]; // even though only one thread (the one writing out really needs this)
                        w = weights[ind_w][ind_m][ind_n]; // this access is not the most efficient
                    }
                }
                //TODO: fix readout or right buffer!!! (index is definitely not right) (2022: is it still?)
                scalar_t acc = w * buf_in[(j + loop_pixel * ind_y_main) * m_per_warp + threadIdx.x];
                //accumulate via shuffeling (e.g. when we have 16 input channels and 2 outputs: the offsets should be
                // 8, 4, 2 and 1. the results of thread 0 and 16 should then be stored in the two outputs!!!)
                //during this phase only half of the threads are really active (so we have to make it up with occupancy)
                for (int offset = m_per_warp / 2; offset >= 1; offset /= 2)
                    acc += __shfl_down_sync(0xffffffff, acc, offset);
                if((threadIdx.x == 0) && (ind_n < n) && // only the one of the threads is supposed to writeout results
                   (pix_ind < overall_samples)) {// prevent invalid writeouts
                    output[pix_ind][ind_n] = acc + b;
                    //printf("%d, %d, = %f\n", pix_ind, ind_n, (float)(acc + b));
                }
            }


        }

    }

//The opposite of the version from before, this is meant to double buffer the input
// or n=32 by splitting up the group via blockDim.z and adaption of shared memory and removal(templated if) of
// the syncthreads. More outputs n via more threads in anycase
// ACTUALLY: A SPECIALIZED VERSION WITHOUT SHARED MEMORY AND SHUFFELING INSTEAD WHEN N <= 32
// interesting case m=16 -> n=1 .... does not pay. It probably only makes sense for n being multiples of m!!!
    template <typename scalar_t, int m>//TODO: this is in use!
    __global__ void cond_mul_cache_input_double_buffer_few_in(//cond_mul_cuda_few_in_many_out( //cond_mul_cache_input_double_buffer_few_in
            const int loop_outer,
            const int loop_inner,
            const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
            const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
            torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
        const int tid = threadIdx.x + threadIdx.y * blockDim.x; //also this is ind_n
        const int ind_n = tid;
        const int base_ind = blockIdx.x * loop_outer * loop_inner;
        const int overall_samples = input.size(0);
        //shared memory needs to hold pixel_per_warp * input_channels pixel.
        extern __shared__ uint8_t shared[];
        scalar_t* buffers[2];
        buffers[0] = (scalar_t*)(&shared[0]);
        buffers[1] = (scalar_t*)(&shared[sizeof(scalar_t) * blockDim.x * blockDim.y]);
        int current_buffer = 1;

        int ind_w = -1;
        scalar_t w[m];
        scalar_t b;
        for(int i=0;i<loop_outer;i++){
            current_buffer = (current_buffer + 1) % 2;
            int pix_ind = base_ind + i * loop_inner + threadIdx.y;
            if( pix_ind < overall_samples){
                buffers[current_buffer][tid] = input[pix_ind][threadIdx.x];
            }
            __syncthreads();


            for(int j=0;j<loop_inner; j++){
                pix_ind = base_ind + i * loop_inner + j;

                if(pix_ind >= overall_samples){
                    return;
                }
                int ind_w_new = inds[pix_ind];
                if(ind_w_new != ind_w){
                    ind_w = ind_w_new;
                    b = bias[ind_w][0][ind_n];
                    scalar_t acc = b;
                    for(int k=0;k<m;k++){
                        //this actually is somewhat efficient when loading
                        w[k] = weights[ind_w][k][ind_n];
                        acc += w[k] * buffers[current_buffer][m * j + k];
                    }
                    output[pix_ind][ind_n]= acc;
                }else{
                    scalar_t acc = b;
                    for(int k=0;k<m;k++){
                        //reading from shared memory might not be the most efficient here
                        // there are 32 banks. we store 64 values in them. within one warp only 2 banks are used.
                        // each by 16 threads accessing the same value!
                        //printf("buffer_index: %d \n", m * j + k);
                        acc += w[k] * buffers[current_buffer][m * j + k];
                    }
                    //also this access is very efficient
                    output[pix_ind][ind_n]= acc;
                }
                //We can omit the synchronization due to double buffering
                //__syncthreads();

            }
        }
    }



//(unused, as it doesn't have speed benefits)
// in case we have fewer than 32 outputs we go by without any shared memory
//only worth if m=4,8,16 && n=1,2,4,8,16,32 &&  n >= m
// const int pixel_per_warp = 32;
//
//          const dim3 threads3(m, 32/m, 3); //96 threads (3 warps)
//          const int pixel_per_block = pixel_per_warp * threads3.z;
//          dim3 blocks((overall_samples + pixel_per_block - 1) / (pixel_per_block));
    template <typename scalar_t, int m>
    __global__ void cond_mul_reg_cache_weights_few_in(//cond_mul_cuda_few_in_many_out_no_shared( //cond_mul_reg_cache_weights_few_in
            const int loop_outer,// loop_outer * loop_inner must be 32
            const int loop_inner,// see ind_buffer! TODO: delete the remainders of ind_buffer
            const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
            const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
            const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
            torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {

        const int tid = threadIdx.x + threadIdx.y * blockDim.x; //index within warp!
        const int n = output.size(1);
        const int simultaneous_pixel_out = 32 / n; //how many pixel can one warp put out
        const int ind_n_out = tid % n; //basically reshaping block
        const int ind_pix_sub = tid/n;
        const unsigned mask = 0xffffffff;

        const int base_ind = (blockIdx.x * blockDim.z + threadIdx.z) * loop_outer * loop_inner;
        if(tid == 0){
            //printf("base_ind %d, outer_loop %d, inner_loop %d, simultaneous_pixel_out %d\n",base_ind,loop_outer, loop_inner, simultaneous_pixel_out);
        }
        const int overall_samples = input.size(0);
        scalar_t buffer;
        int ind_w = -1;
        int pix_ind = base_ind + tid;
        scalar_t w[m];
        scalar_t b;
        for(int i=0;i<loop_outer;i++){

            if(base_ind + i * loop_inner >= overall_samples){
                //if even the base index for this loop iteration is out of bounds
                // we want to exit since none of the threads in this warp will have work to do
                return;
            }
            pix_ind = base_ind + i * loop_inner + threadIdx.y;
            if( pix_ind < overall_samples){
                buffer = input[pix_ind][threadIdx.x]; //threadIdx = ind_m
            }
            //printf("pix_ind_read %d, ind_m %d, tid %d, warp_nr %d, blockIdx %d\n",pix_ind, threadIdx.x, tid, threadIdx.z, blockIdx.x);


            for(int j=0;j<loop_inner; j+=simultaneous_pixel_out){
                pix_ind = base_ind + i * loop_inner + j + ind_pix_sub;
                //printf("pixel_ind: %d of %d, j %d, loop_inner %d, simultaneous_pixel_out %d, local_pixel index %d \n", pix_ind, overall_samples,j,loop_inner,simultaneous_pixel_out,i * loop_inner + j + ind_pix_sub);

                int ind_w_new = ind_w;
                if(pix_ind < overall_samples){
                    ind_w_new = inds[pix_ind];
                }

                //it is somewhat unfortunate that the loop for loading the weights needs to
                //be separate from the one calculating the result
                if(ind_w_new != ind_w){// &&
                    //pix_ind < overall_samples){ //don't load new weights for non-existing pixel
                    ind_w = ind_w_new;
                    b = bias[ind_w][0][ind_n_out];
                    for(int k=0;k<m;k++) {
                        w[k] = weights[ind_w][k][ind_n_out];
                    }
                }
                scalar_t acc = b;
                for(int k=0;k<m;k++){
                    //it is important that none of the threads has exited for loading the things with shuffle
                    //printf("pixel_ind: %d of %d, locally: %d \n", pix_ind, overall_samples, k + (j + ind_pix_sub) * m);
                    scalar_t value = __shfl_sync(mask, buffer, k + (j + ind_pix_sub) * m);
                    acc += w[k] * value;
                }
                if(pix_ind < overall_samples) {
                    output[pix_ind][ind_n_out] = acc;
                }


            }
        }
    }

}//namespace kernels