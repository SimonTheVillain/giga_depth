from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cond_mul_cuda',
    ext_modules=[
        CUDAExtension('cond_mul_cuda', [
            'cond_mul_cuda.cpp',
            'cond_mul_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
