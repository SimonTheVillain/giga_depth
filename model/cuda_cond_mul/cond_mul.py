import math
from torch import nn
from torch.autograd import Function
import torch

installed = False
if installed:
    import cond_mul_cuda
else:
    #jit version of this!!!
    from torch.utils.cpp_extension import load
    # if there is more than one gpu type connected to this computer we need to provide the according architectures
    # to nvcc
    extra_cuda_cflags = []
    for device in range(0, torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(device)
        extra_cuda_cflags.append(f"-arch=sm_{cap[0]}{cap[1]}")
    cond_mul_cuda = load(
        'cond_mul_cuda', ['model/cuda_cond_mul/cond_mul_cuda.cpp', 'model/cuda_cond_mul/cond_mul_cuda_kernel.cu'],
        verbose=True,
        extra_cuda_cflags=extra_cuda_cflags)

torch.manual_seed(42)


class Cond_Mul_Function(Function):
    @staticmethod
    def forward(ctx, input, inds, weights, bias):
        outputs = cond_mul_cuda.forward(input, inds, weights, bias)
        variables = [input] + [inds] + [weights]
        ctx.save_for_backward(*variables)

        return outputs[0]

    @staticmethod
    def backward(ctx, grad_h):
        outputs = cond_mul_cuda.backward(
            grad_h.contiguous(), *ctx.saved_variables)
        grad_input, grad_weights, grad_bias = outputs
        return grad_input, None, grad_weights, grad_bias # what to do about the inds

# module takes input of shape(n, c_in) + indices (inds) of length n to generate the output (n, c_out)
class CondMul(nn.Module):
    def __init__(self, classes, input_features, output_features):
        super(CondMul, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.register_parameter(name='w', param=nn.Parameter(torch.randn((classes, input_features, output_features))))
        self.register_parameter(name='b', param=nn.Parameter(torch.randn((classes, 1, output_features))))

    def forward(self, input, inds):
        assert len(input.shape) == 2, \
            "Expect input to be of shape (n, input_features)."
        assert input.shape[1] == self.input_features, \
            f"Expect the second dimension of input to be of size input_features {self.input_features}."
        assert len(inds.shape) == 1, \
            "Expecting inds to be of shape (n)"
        assert inds.shape[0] == input.shape[0], \
            "Expecting dim[0] of input and inds to be of same size."
        return Cond_Mul_Function.apply(input, inds, self.w, self.b)
