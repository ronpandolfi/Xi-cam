from __future__ import absolute_import, print_function
import pyopencl as cl
import pkg_resources as pkg
import numpy as np

def test():


    devices = []
    for item in cl.get_platforms():
        devices.append(item.get_devices())

    device = devices[0][0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context, device)
    test =(10*np.random.rand(10,10,10)).astype(np.int8)
    filename = "xicam/plugins/f3d/OpenCL/MedianFilter.cl"
    program = cl.Program(context, pkg.resource_string(__name__, filename)).build()
    kernel = cl.Kernel(program, "MedianFilter")
    # kernel = cl.Kernel(program, "test")

    inputBuffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR, hostbuf=test)
    outputBuffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, test.nbytes)
    # program.test(queue, test.shape, None, inputBuffer, outputBuffer)
    # program.MedianFilter(queue, test.shape, None, inputBuffer, outputBuffer,
    #                      np.int32(10), np.int32(10), np.int32(1), np.int32(1))
    kernel.set_args(inputBuffer, outputBuffer, np.int32(10), np.int32(10), np.int32(10), np.int32(5))

    # kernel.set_args(inputBuffer, outputBuffer)
    # cl.enqueue_nd_range_kernel(queue, kernel, test.shape, None)

    localSize = [np.int32(1),np.int32(1)]
    globalSize = [np.int32(10),np.int32(10)]

    cl.enqueue_nd_range_kernel(queue, kernel, globalSize, localSize)
    c = np.empty_like(test)
    cl.enqueue_copy(queue, c, outputBuffer)

    # print(kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device))


    # cl.enqueue_nd_range_kernel(queue, kernel, [0,0], [0, 0])
    # print 'input: ', test
    # print 'c: ', c
    # print(test)
    print(inputBuffer.offset)
    print('====================================')
    # print(c)
    print(outputBuffer.offset)

    # return test, c

def test2():

    a_np = np.random.rand(50000).astype(np.float32)
    b_np = np.random.rand(50000).astype(np.float32)
    #
    # ctx = cl.create_some_context()
    # queue = cl.CommandQueue(ctx)
    #
    devices = []
    for item in cl.get_platforms():
        devices.append(item.get_devices())

    device = devices[0][0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, device)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

    filename = "xicam/plugins/f3d/OpenCL/test2.cl"
    prg = cl.Program(ctx, pkg.resource_string(__name__, filename)).build()

    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    kernel = cl.Kernel(prg, 'sum')
    kernel.set_args(a_g, b_g, res_g)
    localSize = [np.int32(10),np.int32(10)]
    globalSize = [np.int32(10),np.int32(10)]
    cl.enqueue_nd_range_kernel(queue, kernel, globalSize, localSize)
    # prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, res_g)

    # Check on CPU with Numpy:
    print(res_np)
    # print(res_np - (a_np + b_np))
    # print(np.linalg.norm(res_np - (a_np + b_np)))

def testBuffer():

    devices = []
    for item in cl.get_platforms():
        devices.append(item.get_devices())

    device = devices[0][0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, device)
    inputBuffer = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, 10000)

    inputBuffer.release()
    print(inputBuffer.reference_count)

if __name__ == '__main__':
    testBuffer()
    # test()
    # test2()