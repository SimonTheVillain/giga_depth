
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

namespace {
/*
template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmaxf(0.0, z) + fminf(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}
*/
template <typename scalar_t>
__global__ void test_kernel(const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
                            const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds){
    //printf("this is not working\n");
};

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


template <typename scalar_t>
__global__ void cond_mul_cuda_forward_wide_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
    //extern __shared__ uint8_t shared[];
    const int ind = blockIdx.x * blockDim.y + threadIdx.y;
    const int m = weights.size(1);
    const int n = weights.size(2);
    const int in = threadIdx.x;

    //scalar_t *v = (scalar_t*)&shared[0 * sizeof(scalar_t)];//TODO: this! if it is necessary
    //int32_t *is = (int32_t*)&shared[n * blockDim.y * sizeof(scalar_t)];

    if(ind >= input.size(0)){
        return;
    }

    const int ind_w = inds[ind];
    scalar_t result = bias[ind_w][0][in];
    for(int im=0;im<m;im++){
        result += input[ind][im] * weights[ind_w][im][in];
    }
    output[ind][in] = result;
}

template <typename scalar_t, bool m_mult_32>
__global__ void cond_mul_cuda_forward_deep_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
    extern __shared__ uint8_t shared[];

    //TODO: rethink the meaning of threadIdx.x! It should be n

    const int ind = blockIdx.x * blockDim.y;
    const int overall_samples = input.size(0);
    const int m = weights.size(1);
    const int n = blockDim.x;//weights.size(2); // should be same asblockDim.x
    //const int in = threadIdx.x;
    const int threads = blockDim.x*blockDim.y; // threads in one block
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int simultaneous_pix = threads/n; //TODO: template this (should be the same as blockDim.y)
    const int colums_per_warp = threads/n; //TODO: at least replace it with blockDim.y!!!!
    const int warps_per_weight_set = threads / colums_per_warp;
    const int parts = (m + threads -1) / threads; //TODO: make this a parameter


    //int32_t *is = (int32_t*)&shared[0 * sizeof(scalar_t)]; //stored indices (nr of simultaneous points) blockDim.y
    //stored weights for one block of threads n * (threads + 1) // + 1 to fix bank conflict
    scalar_t *w = (scalar_t*)&shared[0];
    //accumulator: (nr of threads + 1 * nr of threads) // + 1 to fix bank conflicts
    scalar_t *acc = (scalar_t*)&shared[(n * (threads + 1) ) * sizeof(scalar_t)];
    //load indices
    int weight_index;
    if( tid < simultaneous_pix &&
        (ind + tid) < overall_samples){ //also check if we are overstepping the boundaries
        //is[tid] = inds[ind + tid];
        weight_index = inds[ind + tid];
    }

    //clear the accumulator
    for(int i = 0;i < threads + 1; i++){
        acc[i * threads + tid] = 0;
    }

    //int im = tid;
    int last_ind = -1; //TODO: incorporate this
    //go trough the pixel that is treated by one warp:
    for(int i = 0;i < simultaneous_pix;i++){
        int pix = ind + i; //pixel index
        //we are hitting the end here:
        if(pix >= overall_samples){
            break;
        }
        //int ind_w = is[i]; //shared memory way of doing things
        int ind_w = __shfl_sync(0xffffffff, weight_index, i);
        //go trough dimension m in the according amount of steps:
        for(int j = 0; j < parts;j++){ //TODO: flip this with the outer loop so we can reuse weights in neighbouring pixel
            //check if we are out of bound
            //load the input vector
            scalar_t v;
            if(threads * j + tid < m || m_mult_32){ // check for bounds (disabled by m_mult_32)
                v = input[pix][threads * j + tid];
            }
            //loading the weights (again in pieces)
            for(int k=0; k < warps_per_weight_set; k++){
                if(j*threads + k * blockDim.y + threadIdx.y < m  || m_mult_32){ //check bounds (disabled by m_mult_32)

                    //n + 1 to fix bank conflict
                     w[k * (threads + 1) + tid] =
                                        weights[ind_w][j * threads + k * blockDim.y + threadIdx.y][threadIdx.x];
                }
            }

            //multiply and store in the according accumulators
            if(threads * j + tid < m  || m_mult_32){//check for bounds (disabled by m_mult_32)
                for(int k=0; k < n;k++){
                    int l = tid * n + k;
                    //bank conflict by reading w fixed by l/threads it makes the access threads + 1 possible
                    acc[(i * n + k) * (threads + 1) + tid] += v * w[l + l / threads];
                }
            }

        }
    }
    {
        //drop out if we are out of bounds:
        int pix = ind + threadIdx.y;
        if(pix >= overall_samples){
            return;
        }

        //int ind_w = is[threadIdx.y]; // will this cause bank conflicts?
        int ind_w = __shfl_sync(0xffffffff, weight_index, threadIdx.y);
        float accu = bias[ind_w][0][threadIdx.x];
        //beautifully sum up the accumulators:
        for(int i = 0; i < threads; i++){
            accu += acc[tid * (threads + 1) + i]; // + 1 for deleting bank conflicts
        }
        //write out the result
        output[pix][threadIdx.x] = accu;

    }

}


template <typename scalar_t, bool m_mult_32>
__global__ void cond_mul_cuda_forward_deep_reuse_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
    extern __shared__ uint8_t shared[];

    //TODO: rethink the meaning of threadIdx.x! It should be n

    const int ind = blockIdx.x * blockDim.y;
    const int overall_samples = input.size(0);
    const int m = weights.size(1);
    const int n = blockDim.x;//weights.size(2); // should be same asblockDim.x
    //const int in = threadIdx.x;
    const int threads = 32; // threads in one block/warp (always 32)
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int simultaneous_pix = blockDim.y; //threads / n
    const int colums_per_warp = blockDim.y; //threads / n
    const int warps_per_weight_set = n; // n
    const int parts = (m + threads - 1) / threads; //TODO: make this a parameter


    //int32_t *is = (int32_t*)&shared[0 * sizeof(scalar_t)]; //stored indices (nr of simultaneous points) blockDim.y
    //stored weights for one block of threads n * (threads + 1) // + 1 to fix bank conflict
    scalar_t *w = (scalar_t*)&shared[0];
    //accumulator: (nr of threads + 1 * nr of threads) // + 1 to fix bank conflicts
    scalar_t *acc = (scalar_t*)&shared[(n * (threads + 1) ) * sizeof(scalar_t)];
    //load indices
    int weight_index;
    if( tid < simultaneous_pix &&
        (ind + tid) < overall_samples){ //also check if we are overstepping the boundaries
        //is[tid] = inds[ind + tid];
        weight_index = inds[ind + tid];
    }

    //clear the accumulator
    for(int i = 0;i < threads + 1; i++){
        acc[i * threads + tid] = 0;
    }

    //int im = tid;
    //go trough the pixel that is treated by one warp:
    for(int j = 0; j < parts;j++){
        int last_ind = -1;
        for(int i=0;i<simultaneous_pix;i++){
            int pix = ind + i; // pixel index

            if(pix >= overall_samples){
                break;
            }
            int ind_w = __shfl_sync(0xffffffff, weight_index, i);
            scalar_t v;
            if(threads * j + tid < m || m_mult_32){ // check for bounds (disabled by m_mult_32)
                v = input[pix][threads * j + tid];
            }
            if(last_ind != ind_w){ // check if we encountered a new weight
                 //loading the weights (again in pieces)
                for(int k=0; k < warps_per_weight_set; k++){
                    if(j*threads + k * blockDim.y + threadIdx.y < m  || m_mult_32){ //check bounds (disabled by m_mult_32)
                        //n + 1 to fix bank conflict
                         w[k * (threads + 1) + tid] =
                                            weights[ind_w][j * threads + k * blockDim.y + threadIdx.y][threadIdx.x];
                    }
                }
                last_ind = ind_w;
            }

            //multiply and store in the according accumulators
            if(threads * j + tid < m  || m_mult_32){//check for bounds (disabled by m_mult_32)
                for(int k=0; k < n;k++){
                    int l = tid * n + k;
                    //bank conflict by reading w fixed by l/threads it makes the access threads + 1 possible
                    acc[(i * n + k) * (threads + 1) + tid] += v * w[l + l / threads];
                }
            }
        }

    }
    {
        //drop out if we are out of bounds:
        int pix = ind + threadIdx.y;
        if(pix >= overall_samples){
            return;
        }

        //int ind_w = is[threadIdx.y]; // will this cause bank conflicts?
        int ind_w = __shfl_sync(0xffffffff, weight_index, threadIdx.y);
        scalar_t accu = bias[ind_w][0][threadIdx.x];
        //beautifully sum up the accumulators:
        for(int i = 0; i < threads; i++){
            accu += acc[tid * (threads + 1) + i]; // + 1 for deleting bank conflicts
        }
        //write out the result
        output[pix][threadIdx.x] = accu;

    }

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


* how to use nvidia profiler: (profiling needs sudo rights but python doesn't find modules with sudo)
sudo env PATH=$PATH nvprof --analysis-metrics -f -o prof.nvvp venv/bin/python test_cuda_cond_mul.py
nvvp prof.nvvp
*/
template <typename scalar_t, bool m_mult_32,int m_per_warp,int n>
__global__ void cond_mul_cuda_forward_deep_reuse32_kernel(
    const int parts,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
    extern __shared__ uint8_t shared[];

    //TODO: rethink the meaning of threadIdx.x! It should be n

    const int base_ind = 32 * blockIdx.x; // the starting pixel for this block
    const int overall_samples = input.size(0);
    const int m = weights.size(1);
    //const int n = blockDim.x;//weights.size(2); // should be same asblockDim.x
    //const int in = threadIdx.x;
    const int threads = 32; // threads in one block/warp (always 32)
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int simultaneous_pix = m_per_warp; //threads / n // same as blockDim.y
    const int colums_per_warp = m_per_warp; //threads / n
    const int warps_per_weight_set = n; // n
    //const int parts = (m + threads - 1) / threads; //TODO: make this a parameter
    scalar_t w[n];

    scalar_t *acc = (scalar_t*)&shared[0];
    //load indices
    int weight_index;
    if( (base_ind + tid) < overall_samples){ //also check if we are overstepping the boundaries
        //load inds for the next 32 pixel
        weight_index = inds[base_ind + tid];
    }
    //scalar_t v[32];
    //clear the accumulator
    for(int i = 0;i < threads + 1; i++){
        acc[i * threads + tid] = 0;
    }
    //return;
    //int im = tid;
    //the input/weights of one pixel need to be split into parts
    for(int j = 0; j < parts;j++){
        // load the next 32 values for 32 pixel:
        int last_ind = -1;
        for(int k = 0; k < 32; k++){
            int pix = base_ind + k;
            if( pix >= overall_samples){
                break;
            }
            scalar_t v = input[pix][32 * j + tid];
            int ind_w = __shfl_sync(0xffffffff, weight_index, k);
            if(ind_w != last_ind){
                //scalar_t result = 0;//TODO: see if this is a good idea
                for(int i=0;i<n;i++){
                    int im = j * 32 + i * blockDim.y + threadIdx.y; //index along m direction of weight / input
                    w[i] = weights[ind_w][im][threadIdx.x];

                    //TODO: this is not ideal actually!!!! for proper Instruction Level Paralellism we want to have all
                    // the load instructions here and the operations on it later
                    //result += w[i] *//TODO: see if this is a good idea
                    //                __shfl_sync(0xffffffff, v, i * blockDim.y + threadIdx.y);//TODO: see if this is a good idea

                }
                //acc[n * k + threadIdx.y * (32*n + n) + threadIdx.x] += result;//TODO: see if this is a good idea
                last_ind = ind_w;
            }//else{//TODO: see if this is a good idea
                scalar_t result = 0;
                for(int i=0;i<n;i++){
                    //TODO: one could interleave loading w and multiplying
                    result += w[i] *
                                    __shfl_sync(0xffffffff, v, i * blockDim.y + threadIdx.y);
                }
                acc[n * k + threadIdx.y * (32*n + n) + threadIdx.x] += result;
            //}//TODO: see if this is a good idea

        }
        /*
        for(int i = 0;i < threads; i++){
            int pix = base_ind + i;
            if( pix >= overall_samples){
                break;
            }
            v[i] = input[pix][threads * j + tid];
        }
        //each iteration 32/n lines in the mxn matrix are worked on so we need to work n lines to get trough the whole warp:
        for(int i = 0; i < n; i++){
            int last_ind = -1;
            int im = j * 32 + i * blockDim.y + threadIdx.y; //index along m direction of weight / input
            scalar_t w;
            //run trough the 32 consecutive pixel
            for(int k = 0; k < 32;k++){
                if(base_ind + k >= overall_samples){
                    break;
                }
                int ind_w = __shfl_sync(0xffffffff, weight_index, k);
                if(ind_w != last_ind){

                    //printf("im %d, \n",im);
                    w = weights[ind_w][im][threadIdx.x];
                    //w=1;
                    last_ind = ind_w;
                }
                scalar_t result = w *
                            __shfl_sync(0xffffffff, v[k], i * blockDim.y + threadIdx.y);
                //scalar_t vau = __shfl_sync(0xffffffff, v[k], i * blockDim.y + threadIdx.y);
                //fill the accumulator
                //printf("i %d, k %d, thdy %d, thdx %d, result %f, w %f, v %f\n",i, k, threadIdx.y, threadIdx.x, result, w, vau);
                //result = 1;
                acc[n * k + threadIdx.y * (32*n + n) + threadIdx.x] += result;
            }

        }*/
    }
    __syncwarp(); // the warp should be in sync anyway (except for turing gpus... there it might differ!!!)
    for(int i=0;i<n;i++){
        int pix_local = i * blockDim.y + threadIdx.y;
        int pix = base_ind + pix_local;
        if(pix >= overall_samples){
            return;
        }
        int ind_w = __shfl_sync(0xffffffff, weight_index, pix_local);
        scalar_t accu = bias[ind_w][0][threadIdx.x];

        //iterate over all the accumulators for this set of values
        for(int j = 0; j < simultaneous_pix;j++){
            //TODO: why isn't i in this
            accu += acc[ j * (n*32 + n) + n * (threadIdx.y + i * blockDim.y) + threadIdx.x];
            //accu +=1;
            /*
            printf("i %d, j %d, thdy %d, thdx %d, accu %f \n",
                        i, j,
                        threadIdx.y, threadIdx.y,
                        acc[ j * (n*32 + n) + n * (threadIdx.y + i * blockDim.y) + threadIdx.x]);
                        */

        }
        output[pix][threadIdx.x] = accu;
    }

}

template <typename scalar_t, bool m_mult_32>
__global__ void cond_mul_cuda_forward_deep_small_shared_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
    extern __shared__ uint8_t shared[];

    //TODO: rethink the meaning of threadIdx.x! It should be n

    const int ind = blockIdx.x * blockDim.y;
    const int overall_samples = input.size(0);
    const int m = weights.size(1);
    const int n = blockDim.x;//weights.size(2); // should be same asblockDim.x
    //const int in = threadIdx.x;
    const int threads = blockDim.x*blockDim.y; // threads in one block
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int simultaneous_pix = blockDim.y; //TODO: template this (should be the same as blockDim.y)
    const int colums_per_warp = blockDim.y;
    const int warps_per_weight_set = threads / colums_per_warp;
    const int parts = (m + threads -1) / threads; //TODO: make this a parameter


    scalar_t *acc = (scalar_t*)&shared[0];
    //load indices
    int weight_index;
    if( tid < simultaneous_pix &&
        (ind + tid) < overall_samples){ //also check if we are overstepping the boundaries
        weight_index = inds[ind + tid];
    }
    //clear the accumulator
    for(int i = 0;i < blockDim.y + 1 ; i++){
        acc[i * threads + tid] = 0;
    }
    //int im = tid;
    int last_ind = -1; //TODO: incorporate this
    //go trough the pixel that is treated by one warp:
    for(int i = 0;i < simultaneous_pix;i++){
        int pix = ind + i; //pixel index
        //we are hitting the end here:
        if(pix >= overall_samples){
            break;
        }
        //int ind_w = is[i]; //shared memory way of doing things
        int ind_w = __shfl_sync(0xffffffff, weight_index, i);
        //go trough dimension m in the according amount of steps:
        //scalar_t accu = 0;
        for(int j = 0; j < parts;j++){ //TODO: flip this with the outer loop so we can reuse weights in neighbouring pixel
            //check if we are out of bound
            //load the input vector
            scalar_t v;
            if(threads * j + tid < m || m_mult_32){ // check for bounds (disabled by m_mult_32)
                //v[tid] = input[pix][threads * j + tid]; //TODO: looks like v could be shfl_synced in (if it were not for it being undefined type)
                v = input[pix][threads * j + tid];
            }

            for(int k=0;k< warps_per_weight_set;k++){
                //load weight
                scalar_t w = weights[ind_w][j*threads + k * blockDim.y + threadIdx.y][threadIdx.x];
                //+ blockDim.x to remove bank conflicts
                acc[threadIdx.y*(threads + blockDim.x) + i * blockDim.x + threadIdx.x] +=
                        w *
                         __shfl_sync(0xffffffff, v, k * blockDim.y + threadIdx.y);

            }
            //acc[threadIdx.y*threads + i * blockDim.x + threadIdx.x] += accu;

        }
        //acc[i * threads + tid] = accu; // for each pixel we have a separate accumulator
    }

    //return;
    {
        //drop out if we are out of bounds:
        int pix = ind + threadIdx.y;
        if(pix >= overall_samples){
            return;
        }

        //int ind_w = is[threadIdx.y]; // will this cause bank conflicts?
        int ind_w = __shfl_sync(0xffffffff, weight_index, threadIdx.y);
        float accu = bias[ind_w][0][threadIdx.x];
        //beautifully sum up the accumulators:
        for(int i = 0; i < blockDim.y; i++){
            //+ blockDim.x to remove bank conflicts
            accu += acc[tid + i * (threads + blockDim.x)]; // TODO: no page conflict here but when filling the accumulators
        }
        //write out the result
        output[pix][threadIdx.x] = accu;

    }

}

//this kernel actually is not really faster in many circumstances
/*
template <typename scalar_t>
__global__ void cond_mul_cuda_forward_fast_kernel(
    const size_t group_len,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds, //indices are in int32 datatype
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> weights,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> bias,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {
    extern __shared__ uint8_t shared[];

    int last_ind = -1;
    const int overall_samples = input.size(0);
    const int m = weights.size(1);
    const int n = weights.size(2);
    const int tid = threadIdx.x;
    scalar_t *w = (scalar_t*)&shared[0 * sizeof(scalar_t)]; // size = m * n
    scalar_t *b = (scalar_t*)&shared[(m * n) * sizeof(scalar_t)]; // size = n
    scalar_t *accu = (scalar_t*)&shared[((m * n) + n) * sizeof(scalar_t)]; // size = n * blockDim.x
    scalar_t *values = (scalar_t*)&shared[((m * n) + n + (n * blockDim.x)) * sizeof(scalar_t)]; // size = m

    for(size_t ind = blockIdx.x * group_len; ind < (blockIdx.x + 1) * group_len; ind++){
        if(ind >= overall_samples){
            //end of the dataset is reached
            return;
        }
        int ind_w = inds[ind];
        if(last_ind != ind_w){
            //reloading weights
            int lid = tid; // load index
            while(lid < m*n){
                int im = lid / n; //these two are not efficient either
                int in = lid % n;
                w[in*m + im] = weights[ind_w][im][in]; // this is not super efficient
                lid += blockDim.x;
            }
            //reloading bias:
            lid = tid;
            while(lid < n){
                b[lid] = bias[ind_w][0][lid];
                lid += blockDim.x;
            }
            last_ind = ind_w;
        }

        //loading the input

        int lid = tid;
        while(lid < m){
            values[lid] = input[ind][lid];
            lid += blockDim.x;
        }
        //clear the accumulators:
        lid = tid;
        while(lid < n * blockDim.x){
            accu[lid] = 0.0f;
            lid += blockDim.x;
        }
        __syncthreads();

        //multiply
        //TODO: this is not efficient in case there is more threads than input channels
        lid = tid;
        while(lid < m){
            scalar_t v = values[lid]; //if we read in values here, we actually don't need to use shared memory input[ind][lid];
            for(int i=0 ; i < n; i++){
                accu[i*blockDim.x + tid] += v * w[i*m + lid];
            }
            lid += blockDim.x;
        }

        __syncthreads();
        // reduce
        //TODO: this is not efficient in case there is less output channels than threads (especially if its less than 32)
        lid = tid;
        while(lid < n){
            scalar_t result = b[lid];

            for(int i=0;i<blockDim.x;i++){
                result += accu[lid*blockDim.x + i];
            }
            output[ind][lid] = result;
            lid += blockDim.x;
        }


    }
}
*/

__global__ void count_classes(
                const size_t class_count,
                const torch::PackedTensorAccessor<int32_t,1,torch::RestrictPtrTraits,size_t> inds,
                int32_t *counters){
    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if(ind >= inds.size(0)){
        return;
    }
    int ind_w = inds[ind];
    if(ind_w > class_count){
        printf("something is seriously off here ind_w %d, class_count%d \n",ind_w, class_count);
    }
    //printf("result for: %d \n", ind_w);
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
        printf("something is seriously off here \n");
    }
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
    scalar_t accu = 0;
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
    scalar_t accu = 0;
    const int start_ind = starting_inds[ind_w];
    const int count = sample_count[ind_w];
    for(int i=0; i < count; i++){
        int ind = lookup_buffer[start_ind + i];
        accu += grad_output[ind][in] * input[ind][im];
    }
    grad_w[ind_w][im][in] = accu;
}
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
      size_t shared_size =  //sizeof(scalar_t) * (threads) + //store values for each thread TODO: remove
                            //sizeof(int32_t) * simultaneous_pix + //store indices for simultaneous pixel TODO: remove
                            //sizeof(scalar_t) * (n + 1) * threads + //weights for each group of threads TODO: remove
                            sizeof(scalar_t) * n * (threads + 1) + //weights for each group of threads
                            sizeof(scalar_t) * (threads + 1) * threads; // accumulator for the result

      //TODO: a few issues still reside: for 1 its not better than just having 128 results and then picking the right one
      //also, for 8 its not better than the version without the shared memory
      if(((n == 1) || (n == 2) || (n == 4)  || (n == 8) || (n == 16) || (n == 32)) && // maybe templating would work
         m%32 == 0){
            //TODO: reevaluate this implementation!!!!
            //neither is it good for n == 32 nor for n == 16 and for n == 1 its for sure not any better!
            shared_size = sizeof(scalar_t) * threads * (threads + 1); // for the accumulator

            const int per_group = 32/n;
            const dim3 threads3(n, per_group);
            const dim3 blocks((overall_samples + 32 - 1) / 32);
            const int parts = (m + threads - 1) / threads;

            switch(n){
                case 1:
                    cond_mul_cuda_forward_deep_reuse32_kernel<scalar_t, true, 32, 1><<<blocks, threads3, shared_size>>>(
                        parts,
                        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                        inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                        weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                    break;
                case 2:
                    cond_mul_cuda_forward_deep_reuse32_kernel<scalar_t, true, 16, 2><<<blocks, threads3, shared_size>>>(
                        parts,
                        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                        inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                        weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                    break;
                case 4:
                    cond_mul_cuda_forward_deep_reuse32_kernel<scalar_t, true, 8, 4><<<blocks, threads3, shared_size>>>(
                        parts,
                        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                        inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                        weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                    break;
                case 8:
                    cond_mul_cuda_forward_deep_reuse32_kernel<scalar_t, true, 4, 8><<<blocks, threads3, shared_size>>>(
                        parts,
                        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                        inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                        weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                    break;
                case 16:
                    cond_mul_cuda_forward_deep_reuse32_kernel<scalar_t, true, 2, 16><<<blocks, threads3, shared_size>>>(
                        parts,
                        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                        inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                        weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                    break;
                case 32:
                    cond_mul_cuda_forward_deep_reuse32_kernel<scalar_t, true, 1, 32><<<blocks, threads3, shared_size>>>(
                        parts,
                        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                        inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                        weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
                        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
                    break;

            }


      }else if(((n == 1) || (n == 2) || (n == 4)  || (n == 8) || (n == 16)) && // maybe templating would work
                m%32 == 0 &&
                shared_size < 16000){
        //terrible performance at n = 32 (next one is 2 times faster)
        //better performance for n = 16 if the weights are shared for consecutive frames
        //same if they are not shared. Alltogether, it is same / better than
        // cond_mul_cuda_forward_deep_small_shared_kernel which uses shared memory only to a small extend

        //TODO: try kernel with shared weights

        //even though this implementation has quite some bank conflicts it gets by with way less shared memory.
        shared_size = //sizeof(scalar_t) * (threads) + //store values for each thread
                      sizeof(scalar_t) * n * (threads + 1) + //weights for each group of threads
                      sizeof(scalar_t) * (threads + 1) * threads; // accumulator for the result

        const int per_group = 32/n;
        const dim3 threads3(n, per_group);
        const dim3 blocks((overall_samples + per_group - 1) / per_group);
        // the reuse_part only really works well when there is enough
        cond_mul_cuda_forward_deep_reuse_kernel<scalar_t, true><<<blocks, threads3, shared_size>>>(
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
            weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
      }else if(((n == 1) || (n == 2) || (n == 4)  || (n == 8) || (n == 16)) && //TODO: find something that works better for 8!!! (templating?)
                m%32 == 0 &&
                shared_size < 16000){
        //version with less use of shared memory
        //same or better than the next one

        //even though this implementation has quite some bank conflicts it gets by with way less shared memory.
        //TODO: check if bank conflicts are still a thing here)
        shared_size = //sizeof(scalar_t) * (threads) + //store values for each thread
                      sizeof(scalar_t) * threads * (threads / n + 1); // accumulator for the result

        const int per_group = 32/n;
        const dim3 threads3(n, per_group);
        const dim3 blocks((overall_samples + per_group - 1) / per_group);

        cond_mul_cuda_forward_deep_small_shared_kernel<scalar_t, true><<<blocks, threads3, shared_size>>>(
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
            weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());


      }
      else if(((n == 1) || (n == 2) || (n == 4)  || (n == 8)) &&// || (n == 16)) &&  //TODO: find something that works better for 8!!! (templating?)
         m%32 == 0 &&
         shared_size < 16000){
         //version with shared memory
         //same or batter than the one without shared memory in  only in a few cases

         std::cout << "deep branch with " << shared_size << "bytes of shared memory" << std::endl;

        const int per_group = 32/n;
        const dim3 threads3(n, per_group);
        const dim3 blocks((overall_samples + per_group - 1) / per_group);
        cond_mul_cuda_forward_deep_kernel<scalar_t, true><<<blocks, threads3, shared_size>>>(
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
            weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());

      }else{
         //version without shared memory
         //std::cout << "wide branch" << std::endl;

         size_t per_group = 256/n;// it actually doesn't matter if this were 32 threads. works just the same
         const dim3 threads3(n, per_group);

         //shared size could be useful but isn't seemingly
         //shared_size = sizeof(scalar_t) * m * per_group;// + sizeof(int32_t) * per_group;
         const dim3 blocks((overall_samples + per_group - 1) / per_group);
         cond_mul_cuda_forward_wide_kernel<scalar_t><<<blocks, threads3>>>(
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
            weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
      }
      /*
      if(n >= 8 || shared_size > 32000){//n%8 == 0){ // maybe this is preferrable for all n bigger than 8 or 16 (not only multiple of n
        std::cout << "wide branch" << std::endl;

        size_t per_group = 256/n;
        const dim3 threads3(n, per_group);

        //shared size could be useful but isn't seemingly
        //shared_size = sizeof(scalar_t) * m * per_group;// + sizeof(int32_t) * per_group;
        const dim3 blocks((overall_samples + per_group - 1) / per_group);
        cond_mul_cuda_forward_wide_kernel<scalar_t><<<blocks, threads3>>>(
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
            weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
      }
      else if(shared_size > 48000 && false){ // 48000 is the smalles supported shared memory size (i think)
        //TODO: remove! this is slow
        threads = 1024;
        const dim3 blocks((overall_samples + threads - 1) / threads);
        cond_mul_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
            weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
      }else if(true){
        std::cout << "deep branch with " << shared_size << "bytes of shared memory" << std::endl;

        const int per_group = 32/n;
        const dim3 threads3(n, per_group);
        const dim3 blocks((overall_samples + per_group - 1) / per_group);
        cond_mul_cuda_forward_deep_kernel<scalar_t><<<blocks, threads3, shared_size>>>(
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
            weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());

      }else{
        //TODO: REMOVE! this is slow
        std::cout << "fast branch" << std::endl;
        //the fast branch that tries to utilize shared memory as good as possible
        //unfortunately the fast branch is not really faster in most cases but just in some

        threads = m;
        threads = 32;
        if(threads>1024){
          threads = 128;
        }
        size_t shared_size = sizeof(scalar_t) * (m*n + n + m + n*threads);
        int group_size = 1;
        dim3 blocks(overall_samples);
        if(overall_samples > 128){
            group_size = overall_samples / 128;
            blocks.x = (overall_samples + group_size - 1) / group_size;
        }

        cond_mul_cuda_forward_fast_kernel<scalar_t><<<blocks, threads, shared_size>>>(
            group_size,
            input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
            weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());


      }
      */
  }));

  return {output};
}

std::vector<torch::Tensor> cond_mul_cuda_backward(
        torch::Tensor grad_output,//gradient of output
        torch::Tensor input,
        torch::Tensor inds,
        torch::Tensor weights) {
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




    int32_t *sizes_gpu;
    int32_t *starting_inds_gpu;
    int32_t *counters_gpu;
    int32_t *ind_lookup_gpu;
    cudaMalloc(&sizes_gpu, sizeof(int32_t) * weights.size(0));
    cudaMalloc(&starting_inds_gpu, sizeof(int32_t) * weights.size(0));
    cudaMalloc(&counters_gpu, sizeof(int32_t) * weights.size(0));
    cudaMalloc(&ind_lookup_gpu, sizeof(int32_t) * grad_output.size(0));

    cudaMemset(sizes_gpu, 0, sizeof(int32_t) * weights.size(0));
    cudaMemset(counters_gpu, 0, sizeof(int32_t) * weights.size(0));
    { //DEBUG: TODO: REMOVE
        //download to cpu
        std::vector<int32_t> sizes_cpu(weights.size(0));
        cudaMemcpy(&sizes_cpu[0], sizes_gpu, sizeof(int32_t) * weights.size(0), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        //accumulate the sizes to get the starting positions (on CPU)
        std::vector<int32_t> starting_inds_cpu(weights.size(0));
        int count = 0;
        for(int i=0;i<weights.size(0);i++){
            //std::cout << "sizes_cpu " << sizes_cpu[i] << std::endl;
        }
    }

    //count occurence of each class
    int threads = 256;
    dim3 blocks((overall_samples + threads - 1) / threads);

    count_classes<<<blocks, threads>>>(grad_output.size(0),
                                    inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),
                                    sizes_gpu);

    //download to cpu
    std::vector<int32_t> sizes_cpu(weights.size(0));
    cudaMemcpy(&sizes_cpu[0], sizes_gpu, sizeof(int32_t) * weights.size(0), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    //accumulate the sizes to get the starting positions (on CPU)
    std::vector<int32_t> starting_inds_cpu(weights.size(0));
    int count = 0;
    for(int i=0;i<weights.size(0);i++){
        starting_inds_cpu[i] = count;
        //std::cout << "starting_ind " << starting_inds_cpu[i] << std::endl;
        count += sizes_cpu[i];
    }
    if(count != grad_output.size(0)){
        std::cout << "counted samples " << count << " vs overall samples " << grad_output.size(0) << std::endl;
    }
    //TODO: upload accumulated
    cudaMemcpy(starting_inds_gpu, &starting_inds_cpu[0], sizeof(int32_t) * weights.size(0), cudaMemcpyHostToDevice);

    //setup lookup buffer
    setup_indices<<<blocks, threads>>>( grad_output.size(0),
                                    inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),
                                    sizes_gpu,
                                    starting_inds_gpu,
                                    ind_lookup_gpu,
                                    counters_gpu);


    //TODO: calc gradients for input, w and b
    AT_DISPATCH_FLOATING_TYPES(weights.type(), "cond_mul_forward_cuda", ([&] {
        //gradient for input: (basically the opposite of the forward path with transposed weights and zeroed bias
        const int m = weights_t.size(1);
        const int n = weights_t.size(2);
        size_t per_group = 256/n;
        const dim3 threads3(n, per_group);

        //shared size could be useful but isn't seemingly
        //shared_size = sizeof(scalar_t) * m * per_group;// + sizeof(int32_t) * per_group;
        dim3 blocks((overall_samples + per_group - 1) / per_group);
        cond_mul_cuda_forward_wide_kernel<scalar_t><<<blocks, threads3>>>(
                    grad_output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(), // input
                    inds.packed_accessor<int32_t,1,torch::RestrictPtrTraits,size_t>(),//indices are in cheaper datatype
                    weights_t.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), // transposed weights
                    bias_back_zero.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(), // no bias (zero bias)
                    grad_input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>()); //output


        // gradient for b
        /*
        cond_mul_cuda_backward_b_kernel<<<blocks, threads3>>>(
                        const int32_t *sample_count,
                        const int32_t *starting_inds,
                        const int32_t *lookup_buffer,
                        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_output,
                        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_b){
                        */
                        //TODO: reinsert

        threads = 256;
        blocks.x = (grad_bias.size(0) * grad_bias.size(2) + threads - 1) / threads;
        cond_mul_cuda_backward_b_kernel<scalar_t><<<blocks, threads>>>(
                    sizes_gpu,
                    starting_inds_gpu,
                    ind_lookup_gpu,
                    grad_output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    grad_bias.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());



        threads = 256;
        blocks.x = (grad_weights.size(0) * grad_weights.size(1) * grad_weights.size(2) + threads - 1) / threads;
        cond_mul_cuda_backward_w_kernel<scalar_t><<<blocks, threads>>>(
                    sizes_gpu,
                    starting_inds_gpu,
                    ind_lookup_gpu,
                    grad_output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
                    grad_weights.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());


        //gradient for w
        /*
        blocks.x = grad_bias.size
        __global__ void cond_mul_cuda_backward_w_kernel(
                        const int32_t *class_count,
                        const int32_t *sizes,
                        const int32_t *lookup_buffer,
                        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_output,
                        const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
                        torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> grad_w){ */

    }));




    //free all the buffers we created
    cudaFree(sizes_gpu);
    cudaFree(starting_inds_gpu);
    cudaFree(ind_lookup_gpu);
    cudaFree(counters_gpu);

  //auto d_gate_weights = d_gates.flatten(1, 2);
  //auto d_weights = d_gate_weights.t().mm(X);
  //auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

  //auto d_X = d_gate_weights.mm(weights);
  //auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  //auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {grad_input, grad_weights, grad_bias};
}
