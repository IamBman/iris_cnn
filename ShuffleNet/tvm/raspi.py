import numpy as np
import onnx
import tvm.relay as relay
import tvm
from tvm import rpc
from tvm.contrib import graph_executor as runtime
import sys
import time

sys.path.append("../..") 
from tool import my_transform


model_path = "../shufflenet_v2.onnx"
onnx_model = onnx.load(model_path)

img_path = "../../S5750L00.png"
img_data = my_transform(img_path)

local_demo = False
if local_demo:
    target = "llvm"
else:
    target ='llvm -mtriple=aarch64-linux-gnu'

shape_dict={'modelInput' : [1, 1, 32, 32]}

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)

lib_fname = './net.tar'
lib.export_library(lib_fname)

# 从远程设备获取 RPC session。
if local_demo:
    remote = rpc.LocalSession()
else:
    host = "192.168.1.6"
    port = 9090
    remote = rpc.connect(host, port)

# 将库（library）上传到远程设备并加载它
remote.upload(lib_fname)
rlib = remote.load_module("net.tar")

# 建立远程 runtime模块
dev = remote.cpu(0)
module = runtime.GraphModule(rlib["default"](dev))
# 设置输入数据
module.set_input('modelInput' , img_data)
# 运行
import timeit

timing_number = 10
timing_repeat = 10
unoptimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
}

print(unoptimized)
# 得到输出
out = module.get_output(0)
# 得到 top1 分类结果
top1 = np.argmax(out.numpy())
labels = np.load('../../label_dir.npy',allow_pickle=True)

print(labels[top1])
