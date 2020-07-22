import torch
import torch.nn as nn
from cuda_cond_mul.cond_mul import CondMul
from cuda_cond_mul.reference_cond_mul import RefCondMul
from cuda_cond_mul.reference_cond_mul import Ref2CondMul
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
    linear_custom.w.data = linear_ref.w.data.clone()
    linear_custom.b.data = linear_ref.b.data.clone()

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

def compare_models(model1, model2, type_ind, width, height, classes):

    model1.w.data = model2.w.data.clone()
    model1.b.data = model2.b.data.clone()
    test = torch.rand((width * height, model1.w.shape[1]), dtype=torch.float32).cuda()
    #test[:] = 1

    test_ind = torch.randint(0, classes * height, (int(width * height),), dtype=type_ind).cuda()
    with torch.no_grad():
        y1 = model1(test, test_ind)
        y2 = model2(test, test_ind)
        #print(y1)
        #print(y2)
        print(torch.max(torch.abs(y1-y2)))

#sys.exit(0)
width = 608
height = 448
classes = 128 # (classes per line)
m = 128
n = 8
absolute_random = True
#width = 1 #TODO: remove these debug measures
#height = 1
#m = 32
#n = 2

linear_ref = RefCondMul(classes * height, m, n).cuda(device) # 32 as output wouldn't work here
linear_custom = CondMul(classes * height, m, n).cuda(device)
#linear_ref.w.data[:] = 1 #TODO: remove these debug measures
#linear_ref.b.data[:] = 0
#linear_custom.w.data[:] = 1
#linear_custom.b.data[:] = 0
compare_models(linear_custom, linear_ref, torch.int32, width, height, classes)
#sys.exit(0)
print("time used by inference (custom)")
measure_time_two_way(linear_custom, torch.int32, width, height, classes, absolute_random)
#sys.exit(0)
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
