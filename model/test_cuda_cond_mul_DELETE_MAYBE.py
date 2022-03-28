import torch
import torch.nn as nn
import numpy as np
from .cuda_cond_mul.cond_mul import CondMul
from .cuda_cond_mul.reference_cond_mul import RefCondMul, RefCondMulConv
from .cuda_cond_mul.reference_cond_mul import Ref2CondMul
import time
import sys

device = torch.cuda.current_device()
test_small = False
if test_small:
    classes = 3
    batch_size = 6
    input_features = 4
    output_features = 2
    X = torch.randn((batch_size, input_features), device=device)
    #X = torch.zeros((batch_size, input_features), device=device)
    X.requires_grad_(True)
    inds = torch.zeros((batch_size), device=device, dtype=torch.int32)
    inds[0:3] = 1
    X.requires_grad_(True)

    #linear = MyPyLinear(input_features, output_features)
    #linear = RefCondMul(classes, input_features, output_features).cuda(device)
    #linear = Ref2CondMul(classes, input_features, output_features).cuda(device)
    linear_ref = RefCondMul(classes, input_features, output_features).cuda(device)
    linear_custom = CondMul(classes, input_features, output_features).cuda(device)
    linear_custom.w.serialized_data = linear_ref.w.serialized_data.clone()
    linear_custom.b.serialized_data = linear_ref.b.serialized_data.clone()

    print("forward: (custom)")
    y = linear_custom(X, inds)
    print(y)

    print("forward: (reference)")
    y = linear_ref(X, inds)
    print(y)

    X.grad = None # zero grad of input
    print("show the accumulative properties of backpropagation (custom)")
    y = linear_custom(X, inds)
    loss = torch.mean(y)
    loss.backward()
    #print(X.grad)
    #print(linear_custom.b.grad)
    print(linear_custom.w.grad)

    y = linear_custom(X, inds)
    loss = torch.mean(y)
    loss.backward()
    #print(X.grad)
    #print(linear_custom.b.grad)
    print(linear_custom.w.grad)


    X.grad = None # zero grad of input
    print("show the accumulative properties of backpropagation (reference)")
    y = linear_ref(X, inds)
    loss = torch.mean(y)
    loss.backward()
    #print(X.grad)
    #print(linear_ref.b.grad)
    print(linear_ref.w.grad)

    y = linear_ref(X, inds)
    loss = torch.mean(y)
    loss.backward()
    #print(X.grad)
    #print(linear_ref.b.grad)
    print(linear_ref.w.grad)


def small_gradient_experiment():
    m_ref_cond = RefCondMul(2, 1, 1).cuda()
    with torch.no_grad():
        m_ref_cond.w[0] = 1
        m_ref_cond.w[1] = 0
        m_ref_cond.b[0] = 0
        m_ref_cond.b[1] = 1

    m_ref = nn.Conv2d(1, 2, 1).cuda()
    with torch.no_grad():
        m_ref.weight[0] = 1
        m_ref.weight[1] = 0
        m_ref.bias[0] = 0
        m_ref.bias[1] = 1

    m = CondMul(2, 1, 1).cuda()
    with torch.no_grad():
        m.w[0] = 1
        m.w[1] = 0
        m.b[0] = 0
        m.b[1] = 1

    inds = torch.tensor(np.array([0, 1, 0, 1, 1, 1])).cuda()
    data = torch.tensor(np.array([0, 1, 2, 3, 4, 5])).cuda().type(torch.float)
    data.requires_grad_(True)
    print("###############cond_mul###############")
    x1 = m(data.reshape((6, 1)), inds.type(torch.int32))
    print(f"x1 {x1}")
    x1 = torch.sum(x1)
    x1.backward()
    print(f"data.grad {data.grad}")
    print(f"w.grad {m.w.grad}")
    print(f"b.grad {m.b.grad}")
    data.grad.zero_()

    print("###############cond_mul_ref###############")
    x1 = m_ref_cond(data.reshape((6, 1)), inds)
    print(f"x1 {x1}")
    x1 = torch.sum(x1)
    x1.backward()
    print(f"data.grad {data.grad}")
    print(f"w.grad {m_ref_cond.w.grad}")
    print(f"b.grad {m_ref_cond.b.grad}")
    data.grad.zero_()

    print("###############cond_mul_convolutional reference###############")
    x1 = m_ref(data.reshape((1, 1, 1, 6)))
    x1 = torch.gather(x1, 1,  inds.reshape((1, 1, 1, 6)))
    print(f"x1 {x1}")
    x1 = torch.sum(x1)
    x1.backward()
    print(f"data.grad {data.grad}")
    print(f"w.grad {m_ref.weight.grad}")
    print(f"b.grad {m_ref.bias.grad}")
    data.grad.zero_()


def measure_time(model, type_ind, width, height, classes):

    runs = 100

    test = torch.rand((width*height, model.w.shape[1]), dtype=torch.float32).cuda()
    test_ind = torch.randint(0, classes*height, (int(width*height / 4), ), dtype=type_ind).cuda()
    test_ind = test_ind.repeat(4, 1)
    test_ind = test_ind.transpose(0, 1).flatten()

    test_ind = torch.randint(0, classes*height, (int(width*height), ), dtype=type_ind).cuda()
    warm_up = True
    with torch.no_grad():
        if warm_up:
            model(test, test_ind)

        torch.cuda.synchronize()

        tsince = int(round(time.time() * 1000))
        for i in range(0, runs):
            model(test, test_ind)
            torch.cuda.synchronize()

        ttime_elapsed = int(round(time.time() * 1000)) - tsince
        print('test time elapsed {}ms'.format(ttime_elapsed / runs))
        torch.cuda.synchronize()


def measure_time_two_way(model, type_ind, width, height, classes, absolute_random = False):
    runs = 10
    test = torch.rand((width*height, model.w.shape[1]), dtype=torch.float32).cuda()
    #test[:] = 1
    #random with 4 elements repeating
    test_ind = torch.randint(0, classes*height, (int(width*height / 4), ), dtype=type_ind).cuda()
    test_ind = test_ind.repeat(4, 1)
    test_ind = test_ind.transpose(0, 1).flatten()

    #absolute random
    if absolute_random:
        test_ind = torch.randint(0, classes*height, (int(width*height), ), dtype=type_ind).cuda()
    warm_up = True
    if warm_up:
        model(test, test_ind)

    model.zero_grad()

    torch.cuda.synchronize()
    tsince = int(round(time.time() * 1000))
    for _ in range(runs):
        y = model(test, test_ind)
        torch.cuda.synchronize()
    ttime_elapsed_forward = int(round(time.time() * 1000)) - tsince

    torch.cuda.synchronize()
    tsince = int(round(time.time() * 1000))
    torch.mean(y).backward()
    torch.cuda.synchronize()
    ttime_elapsed_backward = int(round(time.time() * 1000)) - tsince

    print('test time elapsed forward {}ms backward {}ms'.format(ttime_elapsed_forward/runs, ttime_elapsed_backward))


def measure_time_two_way_conv(model, width, height):
    runs = 10
    test = torch.rand((1, model.weight.shape[1], width, height, ), dtype=torch.float32).cuda()
    warm_up = True
    if warm_up:
        model(test)

    model.zero_grad()

    torch.cuda.synchronize()
    tsince = int(round(time.time() * 1000))
    for _ in range(runs):
        y = model(test)
        torch.cuda.synchronize()
    ttime_elapsed_forward = int(round(time.time() * 1000)) - tsince

    torch.cuda.synchronize()
    tsince = int(round(time.time() * 1000))
    torch.mean(y).backward()
    torch.cuda.synchronize()
    ttime_elapsed_backward = int(round(time.time() * 1000)) - tsince

    print('test time elapsed forward {}ms backward {}ms'.format(ttime_elapsed_forward/runs, ttime_elapsed_backward))

def measure_time_two_way_per_line_conv(width, height, m, n):
    conv = nn.Conv2d(m * height, n * height, 1, groups=height).cuda()

    test = torch.rand((1, m * height, 1, width), dtype=torch.float32).cuda()
    runs = 10
    warm_up = True
    if warm_up:
        conv(test)

    conv.zero_grad()

    torch.cuda.synchronize()
    tsince = int(round(time.time() * 1000))
    for _ in range(runs):
        y = conv(test)
        torch.cuda.synchronize()
    ttime_elapsed_forward = int(round(time.time() * 1000)) - tsince

    torch.cuda.synchronize()
    tsince = int(round(time.time() * 1000))
    torch.mean(y).backward()
    torch.cuda.synchronize()
    ttime_elapsed_backward = int(round(time.time() * 1000)) - tsince

    print('test time elapsed forward {}ms backward {}ms'.format(ttime_elapsed_forward/runs, ttime_elapsed_backward))

def compare_models(model, model_ref, model_conv, batch_size, width, height, classes):

    # fill all the models with the same set of weights
    model.w.serialized_data = model_ref.w.serialized_data.clone()
    model.b.serialized_data = model_ref.b.serialized_data.clone()

    # copy over the weights for a convolutional implementation!
    #print(model.w.data.shape)
    w = model_ref.w.serialized_data.clone()
    w = w.permute(0, 2, 1).reshape(-1, w.shape[1], 1, 1)
    model_conv.conv.weight.serialized_data = w
    b = model_ref.b.serialized_data.clone()
    b = b.permute(0, 2, 1).reshape(-1)
    model_conv.conv.bias.serialized_data = b

    #print(model_conv.conv.weight.data.shape)
    #generate test input:
    x = torch.rand((batch_size * width * height, model.w.shape[1]), dtype=torch.float32, requires_grad=True).cuda()
    x1 = x.clone()
    x2 = x.clone()
    x3 = x.clone()
    x1.retain_grad()
    x2.retain_grad()
    x3.retain_grad()
    test_ind = torch.randint(0, classes * height, (int(batch_size * width * height),), dtype=torch.int32).cuda()
    y1 = model(x1, test_ind)
    y2 = model_ref(x2, test_ind)
    y3 = model_conv(x3, test_ind)
    print("pure difference between y1-y2")
    #print(y1)
    #print(y2)
    if True:
        print(y1-y2)
        print("find outlier along pixel:")
        maxval, where = torch.max(torch.abs(y1-y2), 0)
        for ind, val in zip(where, maxval):
           print(f"ind {ind}, value {val}")
        print("find outlier along channels")
        maxval, where = torch.max(torch.abs(y1-y2), 1)
        for ind, val in zip(where, maxval):
           print(f"ind {ind}, value {val}")
    print("comparisons y1-y2 / y1-y3 (values around 0.5 are fine, all the other stuff should be smaller than e-05)")
    maximum = max(y1.abs().max(), y2.abs().max(), y3.abs().max())
    print(torch.max(torch.abs(torch.div(y1 - y2, y2))))
    print(torch.max(torch.abs(torch.div(y1 - y2, y1))))
    print(torch.max(torch.abs(torch.div(y1 - y3, y3))))
    print(torch.max(torch.abs(torch.div(y1 - y3, y1))))
    print(torch.max(torch.abs(torch.div(y2 - y3, y3))))

    print("comparison for the backpropagation !")
    (y1*y1).sum().backward()
    (y2*y2).sum().backward()
    (y3*y3).sum().backward()
    print("input gradients:")
    if False:
        print(x1.grad)
        print(x1.grad - x2.grad)
        print("find outlier along pixel:")
        maxval, where = torch.max(torch.abs(x1.grad - x2.grad), 0)
        print(where.shape)
        print(where)
        print(maxval)
        for ind, val in zip(where, maxval):
            print(f"ind {ind}, value {val}")
        print("find outlier along channels")
        maxval, where = torch.max(torch.abs(x1.grad - x2.grad), 1)
        print(where)
        print(maxval)
        for ind, val in zip(where, maxval):
            print(f"ind {ind}, value {val}")

    print(torch.max(torch.abs(torch.div(x1.grad - x2.grad, 1))))
    print(torch.max(torch.abs(torch.div(x1.grad - x3.grad, 1))))

    print("bias gradients:")
    b1 = model.b.grad
    b2 = model_ref.b.grad
    b3 = model_conv.conv.bias.grad
    b3 = b3.reshape(b1.shape)
    print(torch.max(torch.abs(torch.div(b1 - b2, 1))))#b2
    print(torch.max(torch.abs(torch.div(b1 - b3, 1))))#b3

    print("weight gradients:")
    w1 = model.w.grad
    w2 = model_ref.w.grad
    w3 = model_conv.conv.weight.grad
    w3 = w3.reshape((w1.shape[0], w1.shape[2], w1.shape[1])).permute(0, 2, 1)
    print(torch.max(torch.abs(torch.div(w1 - w2, 1))))#w2
    print(torch.max(torch.abs(torch.div(w1 - w3, 1))))#w3

    #print(torch.max(torch.abs(torch.div(in_g1 - in_g2, in_g1))))
    #print(torch.max(torch.abs(torch.div(in_g1 - in_g3, in_g1))))



#small_gradient_experiment()
#sys.exit(0)
batch_size = 1#32*32# 32*32
width = 608#8#8#8#60#8
height = 1#4480#4480#448#4480#448
classes = 12 # (classes per line)
m = 16
n = 16#output channels
absolute_random = False
#width = 1 #TODO: remove these debug measures
#height = 1
#m = 64
#n = 1


linear_custom = CondMul(classes * height, m, n).cuda(device)


if True:
    linear_ref = RefCondMul(classes * height, m, n).cuda(device) # 32 as output wouldn't work here
    linear_conv = RefCondMulConv(classes*height, m, n).cuda(device)
    compare_models(linear_custom, linear_ref, linear_conv, batch_size, width, height, classes)
    sys.exit(0)


print("time used by inference (custom)")
measure_time_two_way(linear_custom, torch.int32, width, height, classes, absolute_random)
sys.exit(0)
print("time used by inference (reference)")
measure_time_two_way(linear_ref, torch.int64, width, height, classes, absolute_random)

# TODO: comparison to basic 1x1 convolution
n = 128
print("time used by inference (reference convolution)")
test = nn.Conv2d(m, n, 1).cuda()
measure_time_two_way_conv(test, width, height)

# check out linewise convolution:
n = 128
print("time used by line-wise convolution (reference)")
measure_time_two_way_per_line_conv(width, height, m, n)


#print(linear_custom.w.grad)
#print(linear_custom.b.grad)
#print(X.grad)
