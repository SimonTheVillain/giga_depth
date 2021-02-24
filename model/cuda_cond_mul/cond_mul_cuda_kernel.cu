
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <chrono>
//#define FORCE_DOUBLE

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
//gpuErrchk( cudaPeekAtLastError() );
//gpuErrchk( cudaDeviceSynchronize() );
namespace {



//extremely slow kernel... TODO: delete!!!
/*
template <typename scalar_t>
__global__ void cond_mul_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
    //printf("i hate you\n");

    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = weights.size(1);
    const int n = weights.size(2);
    if(ind >= input.size(0)){
        return;
    }
    int ind_w = inds[ind];
    for(size_t i = 0; i<n ; i++){
        scalar_t accu = bias[ind_w][0][i];

        for(size_t j = 0; j<m; j++){
            //printf("weights %f input %f \n",weights[ind_w][j][i], input[ind][j]);
            accu += weights[ind_w][j][i] * input[ind][j];
        }

        output[ind][i] = accu;

    }
}
*/


//pretty trivial kernel that has as many threads in the x-dimension of a block as output channels
template <typename scalar_t>
__global__ void cond_mul_cuda_forward_wide_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
    //extern __shared__ uint8_t shared[];
    const int ind = blockIdx.x * blockDim.y + threadIdx.y;
    const int m = weights.size(1);//m input channels
    const int n = weights.size(2);//n output channels
    const int in = threadIdx.x;//index for the output is the thread index

    //Note that storing weights + accumulators in shared memory does not necessarily yield better results.
    //especially if the indices aver very random

    if(ind >= input.size(0)){
        return;
    }

    const int ind_w = inds[ind];
#ifdef FORCE_DOUBLE
	double result = bias[ind_w][0][in];
#else
	scalar_t result = bias[ind_w][0][in];
#endif
    for(int im=0;im<m;im++){
        result += input[ind][im] * weights[ind_w][im][in];
    }
    output[ind][in] = result;
}



/* TODO: optimize once more?
* Memory bandwith is used between 75%(4 consecutive shared weights) and 90% (random weights) at n = 32
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

//TODO: extend this for more than 32 outputs (multiple of 32!) (and template it so we don't loose any  performance    
//Kernel for m multiple of 32 and n being one of 1, 2, 4, 8, 16, 32 TODO: n multiple of 32
//reuse means it is trying not to reload weights at every pixel.
template <typename scalar_t,int m_per_warp,int n_per_set>
__global__ void cond_mul_cuda_forward_deep_reuse32_kernel(
                        const int parts_in, //sets as in the template and parts as in this parameter are the same TODO: rename either set or part!
                        const int parts_out,//since the accumulator (shared memory) stores results for the
                        // n_per_set outputs (+ groups) for the next 32 pixel. Only with parts > 1 more than 32 outputs
                        // n>32 can be achieved. This also means that inputs are being read multiple times.
                        // Actually parts_out is bullshit and anyway set to 1! TODO: remove if not needed at all
                        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
                        const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
                        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
                        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
                        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
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
    const int colums_per_warp = m_per_warp; //threads / n
    const int warps_per_weight_set = n_per_set; // n

    //buffer for the weights of the n outputs (for big n this will use too many registers)
    scalar_t w[n_per_set];

    scalar_t *acc = (scalar_t*)&shared[0];
    //load indices
    int weight_index;
    if( (base_ind + tid) < overall_samples){ //also check if we are overstepping the boundaries
        //load inds for the next 32 pixel
        weight_index = inds[base_ind + tid];
    }
    //if n <=32 (n = n_per_set) we can load enough weights / have enough shared memory for the results of the next 32
    // outputs. If n is multiple of 32 we just do it in multiple blocks/parts.
    for(int l = 0; l < parts_out;l++) {
        //scalar_t v[32];
        //clear the accumulator
        for (int i = 0; i < threads + 1; i++) {
            acc[i * threads + tid] = 0;
        }
        //return;
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

                //TODO: document this part

                for (int i = 0; i < n_per_set; i++) {
                    result += w[i] *
                              __shfl_sync(0xffffffff, v, i * blockDim.y + threadIdx.y);
                }
                acc[n_per_set * k + threadIdx.y * (32 * n_per_set + n_per_set) + threadIdx.x] += result;

            }
        }
        __syncwarp(); // the warp should be in sync anyway (except for turing gpus... there it might differ!!!)
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
                            threadIdx.x]; //the current thread
                //accu +=1;
                /*
                printf("i %d, j %d, thdy %d, thdx %d, accu %f \n",
                            i, j,
                            threadIdx.y, threadIdx.y,
                            acc[ j * (n*32 + n) + n * (threadIdx.y + i * blockDim.y) + threadIdx.x]);
                            */

            }
            output[pix][threadIdx.x + l*32] = accu;

            //TODO: find out if this one is necessary!!
            __syncwarp(); // the warp should be in sync anyway (except for turing gpus... there it might differ!!!)
        }
    }

}

/*
* all what has been written in the comments of the function above is applied here...
* it improves performance for n = 1, 2 and 4 so it's actually pretty useless! For bigger n, it uses too many registers.
* to reach 100% occupancy it would be ideal to have 64 threads per not more than 2048 bytes per block.
* the limit for shared memory is reached for n = 8 due to the additional index we use to prevent bank conflicts
* (otherwise it could be at n=16).
*/

template <typename scalar_t,int m_per_warp,int n>
__global__ void cond_mul_cuda_forward_deep_reuse32_high_occupancy_kernel(
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
    const int colums_per_warp = m_per_warp; //threads / n
    const int warps_per_weight_set = n; // n
    //const int parts = (m + threads - 1) / threads; //TODO: make this a parameter

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
                	//TODO: is 32 really right?
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
                result += w[i] *
                                __shfl_sync(0xffffffff, v, i * blockDim.y + threadIdx.y);
            }

            //printf("j %d, tid %d, result = %f\n",j,tid, result);
            //now do reduction: I know. for a few clocks this will underutilize the SM
			/*
            if(n <= 16){
                result += __shfl_down_sync(0x0000ffff, result, 16);
                //printf("shfl 16 tid %d, result = %f\n",tid, __shfl_down_sync(0xffffffff, result, 16));
            }
            if(n <= 8 && tid < 16){
                result += __shfl_down_sync(0x000000ff, result, 8);
            }
            if(n <= 4 && tid < 8){
                result += __shfl_down_sync(0x0000000f, result, 4);
            }
            if(n <= 2 && tid < 4){
                result += __shfl_down_sync(0x00000004, result, 2);
            }
            if(n <= 1 && tid < 2){
                result += __shfl_down_sync(0x00000001, result, 1);
            }
            */

			//proper reduction, we don't need to close down threads since they all are synced (the inneccessary additions are not too bad)
			for (int offset = 16; offset >= n; offset /= 2)
				result += __shfl_down_sync(0xffffffff, result, offset);

            if(tid < n){
                //store result in accumulator (shared memory
                acc[tid + k * n] += result;
            }

        }
    }
    __syncwarp(); // the warp should be in sync anyway (except for turing gpus... there it might differ!!!)
	//return;
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


//again we want to have maximum occupancy here:
//maximum 2048 bytes of shared memory per block
//maximum 32registers per thread
// 64 threads per block
// To always keep the shared memory at its full potential one could store more than 32 pixel simultaneously
// calculation for shared memory:
// 1 channel input: 512 pixel buffer -> 2048 bytes (weight index array would take 16 registers)
// 2 channel input: 256 pixel buffer -> 2048 bytes (weight index array would take 8 registers)
// 4 channel input: 128 pixel buffer -> 2048 bytes (weight index array would take 4 registers)
// 8 channel input: 64 pixel buffer -> 2048 bytes (weight index array would take 2 registers)
// 16 channel input: 32 pixel buffer -> 2048 bytes (weight index takes 1 register)
// 32 channel input: 16 pixel buffer -> 2048 bytes (weight index array takes 1 register for half of the warps
// for the applications this was designed, we don't at best have 608 pixel per line with 16 different classes
// this gives 38 consecutive pixel with same class. Probably the improvements of implementing an index array are minor.
//TODO: maybe also make this work with more than a 32 pixel buffer. (it doesn't pay
//TODO: for the very common case of m=16 and n=1 it would be nice to actually calculate multiple
// pixel outputs at the same time! (rename pixel_per_warp to something more fitting since we are looping trough pixel)
// also, the neighbouring pixel should be looped as in working on pixel in the order (0,16) (1,17) (2,18)

//TODO: m_per_warp and n_per_warp is the same as blockDim.x and blockDim.y
template <typename scalar_t, int pixel_per_warp>
__global__ void cond_mul_cuda_forward_deep_reuse32_few_in_many_out(
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


    //TODO:CHECK!! properly calculate starting index here! (according to threadIdx.z!!!!!!!)
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
            //TODO: fix readout or right buffer!!! (index is definitely not right
            scalar_t acc = w * buf_in[(j + loop_pixel * ind_y_main) * m_per_warp + threadIdx.x]; //bias[ind_w];
            /*
            printf("index shared_buffer %d, m %d, pixel %d, j %d, loop_pixel %d, ind_y_main %d \n",
                    (j + loop_pixel * ind_y_main) * m_per_warp + threadIdx.x,
                    m_per_warp,
                    (j + loop_pixel * ind_y_main),
                    j,
                    loop_pixel,
                    ind_y_main); //max value for the index should be 256 / 16 for pixel index
            scalar_t acc = 0;
             */
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

//TODO: this can be extended to different cases quite easily. Fewer inputs m,
// or n=32 by splitting up the group via blockDim.z and adaption of shared memory and removal(templated if) of
// the syncthreads. More outputs n via more threads in anycase
// ACTUALLY: A SPECIALIZED VERSION WITHOUT SHARED MEMORY AND SHUFFELING INSTEAD WHEN N <= 32
// interesting case m=16 -> n=1 .... does not pay. It probably only makes sense for n being multiples of m!!!
template <typename scalar_t, int m>
__global__ void cond_mul_cuda_few_in_many_out(
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
        if(threadIdx.x == 0){
            //printf("pix_ind %d\n", pix_ind);
        }
        __syncthreads();


        for(int j=0;j<loop_inner; j++){
            pix_ind = base_ind + i * loop_inner + j;

            if(tid == 0){
                //printf("pix_ind %d of %d\n", pix_ind, overall_samples);
            }

            //printf("pixel_ind: %d of %d \n", pix_ind, overall_samples);
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
            //this seems necessary except you would utilize some kind of double buffering
            //or another finer grained method of securing resources between threads
            //__syncthreads();

        }
    }
}

// in case we have fewer than 32 outputs we go by without any shared memory
template <typename scalar_t, int m>
__global__ void cond_mul_cuda_few_in_many_out_no_shared(
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


__global__ void count_classes(
                const size_t class_count,
                const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds,
                int32_t *counters){
    const int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if(ind >= inds.size(0)){
        return;
    }
    int ind_w = inds[ind];
    if(ind_w >= class_count || ind_w < 0){
        printf("[count_classes]something is seriously off here ind_w %d, class_count %d \n",ind_w, class_count);
    }
    atomicAdd(&counters[ind_w], 1);
}

__global__ void setup_indices(
                const size_t class_count,
                const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds,
                const int32_t *sizes, // the amount of elements on each class
                const int32_t *start_inds, //the staring indices for each class in the lookup buffer
                int32_t *lookup_buffer,
                int32_t *counters
                ){
    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if(ind >= inds.size(0)){
        return;
    }
    int ind_w = inds[ind];
    if(ind_w > class_count){
        printf("[setup_indices]something is seriously off here ind_w %d, class_count %d \n",ind_w, class_count);
    }
    //TODO: utilize warp aggregated atomics here!!!
    int count_old = atomicAdd(&counters[ind_w], 1);
    int start_ind = start_inds[ind_w];
    lookup_buffer[start_ind + count_old] = ind;

}

template <typename scalar_t>
__global__ void cond_mul_cuda_backward_b_kernel(
                        const int32_t *sample_count,
                        const int32_t *starting_inds,
                        const int32_t *lookup_buffer,
                        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_output,
                        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_b){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int im = tid % grad_b.size(2);
    const int ind_w = tid / grad_b.size(2);
    if(ind_w >= grad_b.size(0)){
        return;
    }
#ifdef FORCE_DOUBLE
	double accu = 0;//TODO: get rid of the need for double precision operations here!
#else
	scalar_t accu = 0;
#endif
    const int start_ind = starting_inds[ind_w];
    const int count = sample_count[ind_w];
    for(int i=0; i < count; i++){
        int ind = lookup_buffer[start_ind + i];
        accu += grad_output[ind][im];
    }
    grad_b[ind_w][0][im] = accu;
}


template <typename scalar_t>
__global__ void cond_mul_cuda_backward_w_kernel(
                        const int32_t *sample_count,
                        const int32_t *starting_inds,
                        const int32_t *lookup_buffer,
                        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_output,
                        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
                        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_w){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = grad_w.size(1);
    const int n = grad_w.size(2);
    const int imn = tid % (m * n); // this is not fast or beautiful
    const int ind_w = tid / (m * n);
    const int im = imn / n;
    const int in = imn % n;//this is not beautiful
    if(ind_w >= grad_w.size(0)){
        return;
    }
#ifdef FORCE_DOUBLE
    double accu = 0;//TODO: get rid of the need for double precision operations here!
#else
	scalar_t accu = 0;
#endif
    const int start_ind = starting_inds[ind_w];
    const int count = sample_count[ind_w];
    for(int i=0; i < count; i++){
        int ind = lookup_buffer[start_ind + i];
        // while grad_output will be read relatively efficiently (neighbouring threads read neighbouring values)
        // input is read less efficient. Alltogether it probably is not a superterrible approach, even when
        // the lookup_buffer introduces some pointer chasing.
        // What easily could be improved: Aligning the indices so that workgroups would be working on the same block
        // and have the same count of values to accumulate.
        accu += grad_output[ind][in] * input[ind][im];
    }
    grad_w[ind_w][im][in] = accu;
}

//TODO: why is this commented out? was it because it was too slow?
 /*
template <typename scalar_t>
__global__ void cond_mul_cuda_backward_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_input,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> d_weights,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> d_bias,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_output) {

  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < d_gates.size(2)){
    const auto d_output_gate = tanh(new_cell[n][c]) * grad_h[n][c];
    const auto d_tanh_new_cell = output_gate[n][c] * grad_h[n][c];
    const auto d_new_cell =
        d_tanh(new_cell[n][c]) * d_tanh_new_cell + grad_cell[n][c];


    d_old_cell[n][c] = d_new_cell;
    const auto d_candidate_cell = input_gate[n][c] * d_new_cell;
    const auto d_input_gate = candidate_cell[n][c] * d_new_cell;

    d_gates[n][0][c] =
        d_input_gate * d_sigmoid(gate_weights[n][0][c]);
    d_gates[n][1][c] =
        d_output_gate * d_sigmoid(gate_weights[n][1][c]);
    d_gates[n][2][c] =
        d_candidate_cell * d_elu(gate_weights[n][2][c]);

  }

}
*/
} // namespace

std::vector<torch::Tensor> cond_mul_cuda_forward(
    torch::Tensor input,
    torch::Tensor inds,
    torch::Tensor weights,
    torch::Tensor bias) {
  auto options = torch::TensorOptions()
    .dtype(weights.dtype())
    .layout(torch::kStrided)
    .device(weights.device());
    //.requires_grad(true);
  auto output = torch::zeros({input.size(0), weights.size(2)}, options);

  const int overall_samples = input.size(0);
  const int m = weights.size(1);
  const int n = weights.size(2);
  //std::cout << "i shit you not!!! this is forward" << std::endl;




  AT_DISPATCH_FLOATING_TYPES(input.type(), "cond_mul_forward_cuda", ([&] {
      int threads = m;
      threads = 32;
      if(threads>1024){
        threads = 128;
      }


      int simultaneous_pix = threads/n;
      //memory used for weights, bias, variables and accumulator
      size_t shared_size = 0;

      //TODO: a few issues still reside: for 1 its not better than just having 128 results and then picking the right one
      //also, for 8 its not better than the version without the shared memory
      if((((n == 1) || (n == 2) || (n == 4)) && m%32 == 0) || false){

      	//std::cout << "DEBUG: Running the high occupancy kernel(cond_mul_cuda_forward_deep_reuse32_high_occupancy_kernel)" << std::endl;
            //TODO: reevaluate this implementation!!!!
            //neither is it good for n == 32 nor for n == 16 and for n == 1 its for sure not any better!
#ifdef FORCE_DOUBLE
		  shared_size = 2 * sizeof(double) * 32 * n; // for the accumulator
#else
		  shared_size = 2 * sizeof(scalar_t) * 32 * n; // for the accumulator
#endif

            const int per_group = 32/n;
            const dim3 threads3(n, per_group, 2); //lets have 64 threads per group (doubles the use of shared memory)
            const dim3 blocks((overall_samples + 64 - 1) / 64);
            //std::cout << threads3.x << ", " << threads3.y << ", " << threads3.z << std::endl;

            //std::cout << blocks.x << ", " << blocks.y << ", " << blocks.z << std::endl;
            const int parts = (m + 32 - 1) / 32;

            switch(n){
                case 1:
                    cond_mul_cuda_forward_deep_reuse32_high_occupancy_kernel<scalar_t, 32, 1><<<blocks, threads3, shared_size>>>(
                        parts,
                        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                        inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                        weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                    break;
                case 2:
                    cond_mul_cuda_forward_deep_reuse32_high_occupancy_kernel<scalar_t, 16, 2><<<blocks, threads3, shared_size>>>(
                        parts,
                        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                        inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                        weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                    break;
                case 4:
                    cond_mul_cuda_forward_deep_reuse32_high_occupancy_kernel<scalar_t, 8, 4><<<blocks, threads3, shared_size>>>(
                        parts,
                        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                        inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                        weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                    break;
                case 8:
                    cond_mul_cuda_forward_deep_reuse32_high_occupancy_kernel<scalar_t, 4, 8><<<blocks, threads3, shared_size>>>(
                        parts,
                        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                        inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                        weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                    break;
                case 16:
                    cond_mul_cuda_forward_deep_reuse32_high_occupancy_kernel<scalar_t, 2, 16><<<blocks, threads3, shared_size>>>(
                        parts,
                        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                        inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                        weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                    break;
                case 32:
                    cond_mul_cuda_forward_deep_reuse32_high_occupancy_kernel<scalar_t, 1, 32><<<blocks, threads3, shared_size>>>(
                        parts,
                        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                        inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                        weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                    break;

            }
      //}else if((((n == 8) || (n == 16) || (n == 32) || (n%32 == 0)) && m%32 == 0) && true){
      }else if((((n == 8) || (n == 16) || (n == 32)) && m%32 == 0) && true) {
          //std::cout << "DEBUG:Running the deep_reuse kernel(cond_mul_cuda_forward_deep_reuse32_kernel) n,m: " << n << ", " << m << std::endl;
          //TODO: reevaluate this implementation!!!!


          //shared memory (approx 4kb used) is used for the accumulator of the weights.
          //
          shared_size = sizeof(scalar_t) * threads * (threads + 1); // for the accumulator
          //std::cout << "parts: " << parts << ", n: " << n << std::endl;

          const int per_group = 32 / n;
          const dim3 threads3(n, per_group);
          const dim3 blocks((overall_samples + 32 - 1) / 32);
          const int parts_in = (m + 32 - 1) / 32;
          const int parts_out = (n - 1) / 32 + 1; //in this branch, parts_out should always be 1 (it is meant for n>32)
          //std::cout << "shared_size: " << shared_size << std::endl;
          switch (n) {
              case 1:
                  cond_mul_cuda_forward_deep_reuse32_kernel<scalar_t, 32, 1> << < blocks, threads3, shared_size >> > (
                          parts_in,
                                  parts_out,
                                  input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                  inds.packed_accessor<int32_t, 1, torch::RestrictPtrTraits, size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  bias.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
                  break;
              case 2:
                  cond_mul_cuda_forward_deep_reuse32_kernel<scalar_t, 16, 2> << < blocks, threads3, shared_size >> > (
                          parts_in,
                                  parts_out,
                                  input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                  inds.packed_accessor<int32_t, 1, torch::RestrictPtrTraits, size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  bias.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
                  break;
              case 4:
                  cond_mul_cuda_forward_deep_reuse32_kernel<scalar_t, 8, 4> << < blocks, threads3, shared_size >> > (
                          parts_in,
                                  parts_out,
                                  input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                  inds.packed_accessor<int32_t, 1, torch::RestrictPtrTraits, size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  bias.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
                  break;
              case 8:
                  cond_mul_cuda_forward_deep_reuse32_kernel<scalar_t, 4, 8> << < blocks, threads3, shared_size >> > (
                          parts_in,
                                  parts_out,
                                  input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                  inds.packed_accessor<int32_t, 1, torch::RestrictPtrTraits, size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  bias.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
                  break;
              case 16:
                  cond_mul_cuda_forward_deep_reuse32_kernel<scalar_t, 2, 16> << < blocks, threads3, shared_size >> > (
                          parts_in,
                                  parts_out,
                                  input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                  inds.packed_accessor<int32_t, 1, torch::RestrictPtrTraits, size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  bias.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
                  break;
              case 32:
                  cond_mul_cuda_forward_deep_reuse32_kernel<scalar_t, 1, 32> << < blocks, threads3, shared_size >> > (
                          parts_in,
                                  parts_out,
                                  input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                  inds.packed_accessor<int32_t, 1, torch::RestrictPtrTraits, size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  bias.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                  output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
                  break;
              default:
                  //n is multiple of 32, the solution to that is to simply loop this kernel a bit!
                  std::cout << "argh! this is a trap" << std::endl;
                  /*
                  cond_mul_cuda_forward_deep_reuse32_kernel<scalar_t, 1, 32><<<blocks, threads3, shared_size>>>(
                          parts_in,
                          parts_out,
                          input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                          inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                          weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                          bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                          output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                      break;
                   */

          }
      }else if(((m==4 || m==8 || m==16)  && // theoretically also for m==1, m==2 but performs worse than trivial implementation
                (n==1 || n==2 || n==4 || n==8 || n==16 || n==32) &&
                n >= m)
               && true){
          //std::cout << "Hyperspecialized kernel part 2" << std::endl;

          //the kernel is optimized for 32 pixel_per_warp. Don't use with anything else!
          // background is the
          const int pixel_per_warp = 32;

          const dim3 threads3(m, 32/m, 3); //96 threads (3 warps)
          const int pixel_per_block = pixel_per_warp * threads3.z;
          dim3 blocks((overall_samples + pixel_per_block - 1) / (pixel_per_block)); // most of the blocks take the next 32 pixel for each active warp (64)

          switch(m){
              case 1:
                  cond_mul_cuda_few_in_many_out_no_shared<scalar_t, 1><<<blocks, threads3>>>(
                          pixel_per_warp /threads3.y, //how often do we need to read channels/pixels to fill all pixel
                                  threads3.y,//pixel of the inner loop (can take more than 1 pixel at a time)
                                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  break;
              case 2:
                  cond_mul_cuda_few_in_many_out_no_shared<scalar_t, 2><<<blocks, threads3>>>(
                                    pixel_per_warp /threads3.y, //how often do we need to read channels/pixels to fill all pixel
                                  threads3.y,//pixel of the inner loop (can take more than 1 pixel at a time)
                                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  break;
              case 4:
                  cond_mul_cuda_few_in_many_out_no_shared<scalar_t, 4><<<blocks, threads3>>>(
                                    pixel_per_warp /threads3.y, //how often do we need to read channels/pixels to fill all pixel
                                  threads3.y,//pixel of the inner loop (can take more than 1 pixel at a time)
                                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  break;
              case 8:
                  cond_mul_cuda_few_in_many_out_no_shared<scalar_t, 8><<<blocks, threads3>>>(
                                    pixel_per_warp /threads3.y, //how often do we need to read channels/pixels to fill all pixel
                                  threads3.y,//pixel of the inner loop (can take more than 1 pixel at a time)
                                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  break;
              case 16:
                  cond_mul_cuda_few_in_many_out_no_shared<scalar_t, 16><<<blocks, threads3>>>(
                                    pixel_per_warp /threads3.y, //how often do we need to read channels/pixels to fill all pixel
                                  threads3.y,//pixel of the inner loop (can take more than 1 pixel at a time)
                                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  break;
          };

      }else if(((m==1 || m==2 || m==4 || m==8 || m==16)  &&
                (n==64 || n==96 || n==128 || n==160 || n==192 || n==224 || n==256))
                && true){
          //TODO: this is supposed to be the most specialized kernel of them all! ( the next kernel)
          // but the principle could work for different applications as well. e.g.
          // 16->16 or 16->32 or 16->128, 4->16 etc.
          // fewer than 64 outputs should probably not be done by reducing thread count, but by adding to threadDim.z
          // e.g. 16->16 could still have 64 threads if we just have (16, 1, 4) blocks.
          // actually for 32 and fewer outputs we shouldn't use shared memory at all. SHUFFELING
          //std::cout << " the hyperspecialized kernel, m: " << m << " n: " << n << std::endl;
          const int pixel_per_block = std::max(64, n/m);
          //std::cout << "pixel_per_block" << pixel_per_block << std::endl;
          dim3 blocks((overall_samples + pixel_per_block - 1) / (pixel_per_block)); // most of the blocks take the next 32 pixel for each active warp (64)
          const dim3 threads3(m, n/m); //alltogether we want n threads.
          shared_size = sizeof(scalar_t) * threads3.x * threads3.y * threads3.z * 2; //two times since we double buffer!!!

          //std::cout << "loop_inner " << threads3.y << std::endl;
          //std::cout << "loop_outer " << pixel_per_block/threads3.y << std::endl;
          switch(m){
              case 1:
                  cond_mul_cuda_few_in_many_out<scalar_t, 1><<<blocks, threads3, shared_size>>>(
                            pixel_per_block /threads3.y, //how often do we need to read channels/pixels to fill all pixel
                                  threads3.y,//loop_inner (or how many pixel are read simultaneously by all the threads)
                                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  break;
              case 2:
                  cond_mul_cuda_few_in_many_out<scalar_t, 2><<<blocks, threads3, shared_size>>>(
                          pixel_per_block /threads3.y, //how often do we need to read channels/pixels to fill all pixel
                                  threads3.y,//loop_inner (or how many pixel are read simultaneously by all the threads)
                                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  break;
              case 4:
                  cond_mul_cuda_few_in_many_out<scalar_t, 4><<<blocks, threads3, shared_size>>>(
                          pixel_per_block /threads3.y, //loop outer
                                  threads3.y,//loop_inner (or how many pixel are read simultaneously by all the threads)
                                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  break;
              case 8:
                  cond_mul_cuda_few_in_many_out<scalar_t, 8><<<blocks, threads3, shared_size>>>(
                                    pixel_per_block / threads3.y, //loop outer
                                  threads3.y,//loop_inner (or how many pixel are read simultaneously by all the threads)
                                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  break;
              case 16:
                  cond_mul_cuda_few_in_many_out<scalar_t, 16><<<blocks, threads3, shared_size>>>(
                                  pixel_per_block /threads3.y, //loop outer
                                  threads3.y,//loop_inner (or how many pixel are read simultaneously by all the threads)
                                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  break;
          };

          //std::cout << "blocks: " << blocks.x << std::endl;
        /*
          cond_mul_cuda_few_in_many_out<scalar_t, 16><<<blocks, threads3, shared_size>>>(
                  outer_iterations, //loop outer
                  inner_iterations,//loop_inner
                  input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                  inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                  weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                  bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                  output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                  */

      }else if(     ((m == 1) && ((n==4) || (n == 8) || (n==16) || (n%32 == 0) )) || //n==1,2 would also be an inefficient option
                    ((m == 2) && ((n==4)|| (n==8) || (n%16 == 0))) || //n==1,2 would probably be very inefficient
                    ((m == 4) && ((n==2) || (n==4) || (n%16 == 0))) ||
                    ((m==8) && ((n==1) || (n==2) || (n%4 == 0)) ||
                    ((m==16) && ((n==1) || (n%2==0)) ) ||
                    (m==32)) && false){
          //TODO: remove!
          // this actually is slow as fuck!!!!! ABYSMAL. It might have been better with more shared memory but as it is,
          // it is useless!!!
          //TODO:
          //This branch cares for cases in which the input is smaller than 32. Ideally the output has enough output channels
          //to sufficiently saturate the warps.
          //e.g. 16 input channels and only 1 output channel would only use only half of a given warp in the output
          //so for saturation the ideal combinations would be
          // m==1 n%32==0
          // m==2 n%16==0
          // m==4 n%8==0 etc.
          //TODO: find a way to better handle those cases?
          //e.g. for m==16 / n==1 one warp could handle 2 pixel at once (by changing a pixel_per_warp
          // parameter/template, as this is currently describing how many pixels we loop trough. Not how many we work
          // simultaneously)
          // Or for m==1 / n==1 one warp could handle 32 pixel simultaneously, which would probably be worse than
          // the unbuffered version. To better utilize this weights would need to be reused for more than 32 pixel,
          // data & implementation wise.
          size_t per_group = 32 / m;
          const dim3 threads3(m, per_group, 2); //64 threads 2 active warps per block

          ////TODO: make it possible to have more than 32 pixel for a block!
          //dim3 blocks((overall_samples + 64 - 1) / 64); // most of the blocks take the next 32 pixel for each active warp (64)
          //shared_size = m * sizeof(scalar_t) * 32; //TODO: if we go for more than 32 pixel per warp we definitely want to reuse this
          //int parts_in;
          //int parts_out;
          int loop_pixel;

          if( m <= 8){
              //std::cout << "TODO: debug this new kernel (few_in_many_out) for m<=16 " << std::endl;
              //m == 1,2,4,8
              //we buffer 32 pixel in advance:

              //parts_out = (n + 32 -1) / 32; //TODO: i don't get it! fix this
              //the valid n is 1,2,4,8,16,32, multiples of 32
              const int pixel_per_warp = 32;

              //how many pixel should be worked on simultaneously:
              // m=1    n=1 -> 32 pixel simultaneously
              //        n=2 -> 16 pix
              //        n=4 -> 8 pix
              //        n=8 -> 4 pix
              //        n=16-> 2 pix
              //        n>=32-> 1 pix
              //m=2     n=1 -> 16 pixel simultaneously
              //        n=2,4,8 -> 8,4,2
              //        n>=16 -> 1pix
              //etc.
              int simultaneous_output_pixel = std::max(1, static_cast<int>(threads3.y/n));
              int simultaneous_output_channels = threads3.y / simultaneous_output_pixel; //TODO: difference to n_per_warp!? (difference to n_per_warp!?)
              int loop_pixel = pixel_per_warp / simultaneous_output_pixel;
              dim3 blocks((overall_samples + 2*pixel_per_warp - 1) / (2*pixel_per_warp)); // most of the blocks take the next 32 pixel for each active warp (64)
              shared_size = sizeof(scalar_t) * m * pixel_per_warp * 2; // twice since we have 2 warps working on stuff!!!
              //std::cout << "shared_size" << shared_size << std::endl;
              //std::cout << "simultaneous_output_pixel "  << simultaneous_output_pixel << " simultaneous_output_channels " << simultaneous_output_channels << std::endl;
              //std::cout << "loop_pixel " << loop_pixel << std::endl;

              cond_mul_cuda_forward_deep_reuse32_few_in_many_out<scalar_t, 32><<<blocks, threads3, shared_size>>>(
                      loop_pixel, //in the normal case we would loop over pixel_per_warp pixel since we go 1 pixel at a time TODO: rename loop_out_pixel
                      simultaneous_output_pixel, //if we have very few output channels we can handle multiple pixel at once
                      simultaneous_output_channels, //TODO: simultaneous out_channels can be derived from blockDim.y (blockDim.y = simultaneous_out_pixel * simultaneous_out_channels)
                      input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                      inds.packed_accessor<int32_t, 1, torch::RestrictPtrTraits, size_t>(),//indices are in cheaper datatype
                      weights.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                      bias.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                      output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
              );
            /*
template <typename scalar_t, int pixel_per_warp>
__global__ void cond_mul_cuda_forward_deep_reuse32_few_in_many_out(
        //const int parts_in, // the input is split up in parts. Needs to be set so it works with pixel_per_warp TODO: Template? rename loop_in
        //const int parts_out, //the output is split up in parts. Each part/warp can have n_per_warp outputs TODO: rename loop_out_channels
        //const int n_simultaneous,
        const int loop_pixel, //in the normal case we would loop over pixel_per_warp pixel since we go 1 pixel at a time TODO: rename loop_out_pixel
        const int simultaneous_out_pixel, //if we have very few output channels we can handle multiple pixel at once
        const int simultaneous_out_channels, //TODO: simultaneous out_channels can be derived from blockDim.y (blockDim.y = simultaneous_out_pixel * simultaneous_out_channels)
        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
        const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
        const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
        torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
             */

          }else if(m == 16){
              //m == 16
              //we buffer only 16 pixel in advance

              //std::cout << "TODO: debug this new kernel (few_in_many_out) for m==16 m, n:" << m << ", " << n << std::endl;
              //we buffer 16 pixel in advance:
              const int pixel_per_warp = 16;

              //how many pixel should be worked on simultaneously:
              // m=32 -> simultaneous_output_pixel = 1 (threads3.y = 1),
              // simultaneous_output_channels = 1
              // loop_pixel = pixel_per_warp
              int simultaneous_output_pixel = std::max(1, static_cast<int>(threads3.y/n));
              int simultaneous_output_channels = threads3.y / simultaneous_output_pixel; //TODO: difference to n_per_warp!? (difference to n_per_warp!?)
              int loop_pixel = pixel_per_warp / simultaneous_output_pixel;
              dim3 blocks((overall_samples + 2*pixel_per_warp - 1) / (2*pixel_per_warp)); // most of the blocks take the next 32 pixel for each active warp (64)
              shared_size = sizeof(scalar_t) * m * pixel_per_warp * 2; // twice since we have 2 warps working on stuff!!!
              //std::cout << "shared_size" << shared_size << std::endl;
              //std::cout << "simultaneous_output_pixel "  << simultaneous_output_pixel << " simultaneous_output_channels " << simultaneous_output_channels << std::endl;
              //std::cout << "loop_pixel " << loop_pixel << std::endl;

              //TODO: n_per_warp!?
              //TODO: reduce parameters. eg. parts_in should be utterly useless and derived from nr of pixel
              cond_mul_cuda_forward_deep_reuse32_few_in_many_out<scalar_t, 16><<<blocks, threads3, shared_size>>>(
                              loop_pixel, //in the normal case we would loop over pixel_per_warp pixel since we go 1 pixel at a time TODO: rename loop_out_pixel
                              simultaneous_output_pixel, //if we have very few output channels we can handle multiple pixel at once
                              simultaneous_output_channels, //TODO: simultaneous out_channels can be derived from blockDim.y (blockDim.y = simultaneous_out_pixel * simultaneous_out_channels)
                              input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                              inds.packed_accessor<int32_t, 1, torch::RestrictPtrTraits, size_t>(),//indices are in cheaper datatype
                              weights.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                              bias.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                              output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
              );

          }else{
              //m == 32
              //we buffer only 16 pixel in advance

              //std::cout << "TODO: debug this new kernel (few_in_many_out) for m==32 m, n:" << m << ", " << n << std::endl;
              //we buffer 16 pixel in advance:
              const int pixel_per_warp = 8;

              //how many pixel should be worked on simultaneously:
              // m=32 -> simultaneous_output_pixel = 1 (threads3.y = 1),
              // simultaneous_output_channels = 1
              // loop_pixel = pixel_per_warp
              int simultaneous_output_pixel = std::max(1, static_cast<int>(threads3.y/n));
              int simultaneous_output_channels = threads3.y / simultaneous_output_pixel; //TODO: difference to n_per_warp!? (difference to n_per_warp!?)
              int loop_pixel = pixel_per_warp / simultaneous_output_pixel;
              dim3 blocks((overall_samples + 2*pixel_per_warp - 1) / (2*pixel_per_warp)); // most of the blocks take the next 32 pixel for each active warp (64)
              shared_size = sizeof(scalar_t) * m * pixel_per_warp * 2; // twice since we have 2 warps working on stuff!!!
              //std::cout << "shared_size" << shared_size << std::endl;
              //std::cout << "simultaneous_output_pixel "  << simultaneous_output_pixel << " simultaneous_output_channels " << simultaneous_output_channels << std::endl;
              //std::cout << "loop_pixel " << loop_pixel << std::endl;

              //TODO: n_per_warp!?
              //TODO: reduce parameters. eg. parts_in should be utterly useless and derived from nr of pixel
              cond_mul_cuda_forward_deep_reuse32_few_in_many_out<scalar_t, 8><<<blocks, threads3, shared_size>>>(
                      loop_pixel, //in the normal case we would loop over pixel_per_warp pixel since we go 1 pixel at a time TODO: rename loop_out_pixel
                              simultaneous_output_pixel, //if we have very few output channels we can handle multiple pixel at once
                              simultaneous_output_channels, //TODO: simultaneous out_channels can be derived from blockDim.y (blockDim.y = simultaneous_out_pixel * simultaneous_out_channels)
                              input.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                              inds.packed_accessor<int32_t, 1, torch::RestrictPtrTraits, size_t>(),//indices are in cheaper datatype
                              weights.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                              bias.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                              output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
              );
          }
      }else{
         //version without shared memory
         //TODO: fix the cases in which this fails!!!
         //std::cout << "wide branch" << std::endl;

         size_t per_group = 256/n;// it actually doesn't matter if this were 32 threads. works just the same
         assert(n * per_group == 256);
         const dim3 threads3(n, per_group);
         const dim3 blocks((overall_samples + per_group - 1) / per_group);


         cond_mul_cuda_forward_wide_kernel<scalar_t><<<blocks, threads3>>>(
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
            weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
      }
  }));
  //TODO: remove this device synchronize!
  //gpuErrchk( cudaPeekAtLastError() );
  //gpuErrchk( cudaDeviceSynchronize() );
  return {output};
}


//#define MEASURE_TIME
std::vector<torch::Tensor> cond_mul_cuda_backward(
        torch::Tensor grad_output,//gradient of output
        torch::Tensor input,
        torch::Tensor inds,
        torch::Tensor weights) {
    //std::cout << "cond_mul_cuda_backward (start)" << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    //i shit you not its not even trying to compile this!!!!
    auto device = weights.device();
    auto grad_weights = torch::zeros_like(weights);
    auto grad_input = torch::zeros_like(input);
    auto weights_t = torch::transpose(weights, 1, 2);
    auto options =
    torch::TensorOptions()
        .dtype(weights.dtype())
        .layout(torch::kStrided)
        .device(weights.device());
    //.requires_grad(true); //what if the tensor is supposed to be the gradient itself

    auto grad_bias = torch::zeros({weights.size(0), 1, weights.size(2)}, options);//TODO: device of correct type would be nice!!!

    auto bias_back_zero = torch::zeros({weights.size(0), 1, weights.size(1)}, options);
    size_t overall_samples = input.size(0);
    /*
    std::cout << "creating tensors of size = " <<  weights.size(0) * weights.size(2) * sizeof(float) << " and " <<
                weights.size(0) * weights.size(1) * sizeof(float) << " bytes" << std::endl;
    */

    if(grad_output.size(0) == 0){
        //this is not supposed to happen but it happened once so lets keep it here
        assert(0);
    }

    int32_t *sizes_gpu;
    int32_t *starting_inds_gpu;
    int32_t *counters_gpu;
    int32_t *ind_lookup_gpu;
    /*
    std::cout << "allocating temporary " <<
                               sizeof(int32_t) * weights.size(0)*3 +
                               sizeof(int32_t) * grad_output.size(0) << " bytes" << std::endl;*/
    cudaMalloc(&sizes_gpu, sizeof(int32_t) * weights.size(0));
    cudaMalloc(&starting_inds_gpu, sizeof(int32_t) * weights.size(0));
    cudaMalloc(&counters_gpu, sizeof(int32_t) * weights.size(0));
    cudaMalloc(&ind_lookup_gpu, sizeof(int32_t) * grad_output.size(0));

    cudaMemset(sizes_gpu, 0, sizeof(int32_t) * weights.size(0));
    cudaMemset(counters_gpu, 0, sizeof(int32_t) * weights.size(0));
    { //DEBUG: TODO: REMOVE
        //download to cpu
        /*
        std::vector<int32_t> sizes_cpu(weights.size(0));
        cudaMemcpy(&sizes_cpu[0], sizes_gpu, sizeof(int32_t) * weights.size(0), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        //accumulate the sizes to get the starting positions (on CPU)
        std::vector<int32_t> starting_inds_cpu(weights.size(0));
        int count = 0;
        for(int i=0;i<weights.size(0);i++){
            //std::cout << "sizes_cpu " << sizes_cpu[i] << std::endl;
        }
         */
    }

    //count occurence of each class
    int threads = 256;
    dim3 blocks((overall_samples + threads - 1) / threads);


    count_classes<<<blocks, threads>>>(weights.size(0),//nr of different classes //grad_output.size(0),
                                    inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),
                                    sizes_gpu); //the counts for each class
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		const char* errstr = cudaGetErrorString(error);
		std::cout << errstr << std::endl;
	}
	//TODO: Is there a sensible way of doing this on the GPU? Not synchronizing here!
	//cudaDeviceSynchronize(); // the synchronization happens inherently with memcpy since it is on the default stream
    //download to cpu
    std::vector<int32_t> sizes_cpu(weights.size(0));
    cudaMemcpy(&sizes_cpu[0], sizes_gpu, sizeof(int32_t) * weights.size(0), cudaMemcpyDeviceToHost);

    //cudaDeviceSynchronize(); // the synchronization happens inherently with memcpy since it is on the default stream
    //accumulate the sizes to get the starting positions (on CPU)
    std::vector<int32_t> starting_inds_cpu(weights.size(0));
    int count = 0;


    //std::cout << "calculating the starting positions of " << weights.size(0) << "weights" << std::endl;
    for(int i=0;i<weights.size(0);i++){
        starting_inds_cpu[i] = count;
        //std::cout << "starting_ind " << starting_inds_cpu[i] << std::endl;
        count += sizes_cpu[i];
    }

    if(count != grad_output.size(0)){
		//std::cout << "accumulating weight gradients on gpu: " << gpu << " " << prop.name << std::endl;
        // a serious issue, that needs to be fixed!!!!!
        std::cout << "counted samples " << count << " vs overall samples " << grad_output.size(0) << std::endl;
    }
    assert(count == grad_output.size(0));

    //upload the starting indices for the individual weights
    cudaMemcpy(starting_inds_gpu, &starting_inds_cpu[0], sizeof(int32_t) * weights.size(0), cudaMemcpyHostToDevice);

    //setup lookup buffer
    setup_indices<<<blocks, threads>>>( weights.size(0),//nr of different classes//grad_output.size(0),
                                    inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),
                                    sizes_gpu,
                                    starting_inds_gpu,
                                    ind_lookup_gpu,
                                    counters_gpu); // the counters for each individiual class

    //Reuse the forward code for the backward pass!!!!
    //TODO: validate this result Otherwise

#ifdef MEASURE_TIME
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time for preparation: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    begin = end;
#endif

    std::vector<torch::Tensor>  result =  cond_mul_cuda_forward(
            grad_output,
            inds,
            weights_t,
            bias_back_zero);
    grad_input = result[0];
#ifdef MEASURE_TIME
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    std::cout << "Time for backward: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    begin = end;
#endif


    //calc gradients for input, w and b
    AT_DISPATCH_FLOATING_TYPES(weights.type(), "cond_mul_forward_cuda", ([&] {
        //gradient for input: (basically the opposite of the forward path with transposed weights and zeroed bias
        const int m = weights_t.size(1);
        const int n = weights_t.size(2);

        //TODO: remove if the validation from up there is fruitful!!! this with the call of the forward pass
        /*
        size_t per_group = 256/std::min(n, 256);//prevent division by zero
        const dim3 threads3(n, per_group);
        dim3 blocks((overall_samples + per_group - 1) / per_group);
        cond_mul_cuda_forward_wide_kernel<scalar_t><<<blocks, threads3>>>(
                    grad_output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(), // input
                    inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                    weights_t.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), // transposed weights
                    bias_back_zero.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), // no bias (zero bias)
                    grad_input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()); //output
        */


        // gradient for b

        threads = 256;
        dim3 blocks((grad_bias.size(0) * grad_bias.size(2) + threads - 1) / threads);
        //blocks.x = (grad_bias.size(0) * grad_bias.size(2) + threads - 1) / threads;
        cond_mul_cuda_backward_b_kernel<scalar_t><<<blocks, threads>>>(
                    sizes_gpu,
                    starting_inds_gpu,
                    ind_lookup_gpu,
                    grad_output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    grad_bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());


#ifdef MEASURE_TIME
        cudaDeviceSynchronize();
        end = std::chrono::steady_clock::now();
        std::cout << "Time for accumulating bias grads: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
        begin = end;
#endif

        threads = 256;
        blocks.x = (grad_weights.size(0) * grad_weights.size(1) * grad_weights.size(2) + threads - 1) / threads;
        cond_mul_cuda_backward_w_kernel<scalar_t><<<blocks, threads>>>(
                    sizes_gpu,
                    starting_inds_gpu,
                    ind_lookup_gpu,
                    grad_output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    grad_weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());

    }));

#ifdef MEASURE_TIME
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    std::cout << "Time for accumulating weight grads: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
#endif

    //free all the buffers we created
    cudaFree(sizes_gpu);
    cudaFree(starting_inds_gpu);
    cudaFree(ind_lookup_gpu);
    cudaFree(counters_gpu);
    //std::cout << "freeing temporary memory" << std::endl;

  //auto d_gate_weights = d_gates.flatten(1, 2);
  //auto d_weights = d_gate_weights.t().mm(X);
  //auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

  //auto d_X = d_gate_weights.mm(weights);
  //auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  //auto d_input = d_X.slice(/*dim=*/1, state_size);
  /*
	std::cout << "right before the end of backward cond_mul" << std::endl;
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
	std::cout << "after backward cond_mul" << std::endl;
	*/
  return {grad_input, grad_weights, grad_bias};
}
