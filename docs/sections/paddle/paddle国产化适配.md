## 1、针对飞腾（aarch）的编译步骤

### 步骤1：启动 Docker 容器

在飞腾版本的 Docker 中启动一个新的容器。

```bash
docker run -itd --name paddle  sanmaomashi/python:3.9.16-debain10-aarch64
docker exec -it paddle bash
```

### 步骤2：编译 CMake

[CMake](https://github.com/Kitware/CMake) 是一个跨平台的构建工具，用于控制软件编译过程。本步骤下载并编译安装 CMake。

```bash
wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4.tar.gz
tar -xzf cmake-3.26.4.tar.gz && cd cmake-3.26.4
./bootstrap && make && sudo make install
```

### 步骤3：编译 patchelf

[patchelf](https://github.com/NixOS/patchelf) 是一个工具，用于修改 ELF 可执行文件和库的动态链接。首先我们需要检查系统中是否已经安装了 patchelf，如果没有则进行安装。

在终端输入 `patchelf --version`，如果 `patchelf` 已经安装，将会显示 `patchelf` 的版本信息。如果没有安装，则进行如下编译安装：

```bash
./bootstrap.sh && ./configure
make && make check && sudo make install
```

### 步骤4：安装 Python 依赖库

使用 pip 命令安装 PaddlePaddle 所需的 Python 依赖库。

```text
httpx
numpy>=1.13
protobuf>=3.20.2 ; platform_system != "Windows"
protobuf>=3.1.0, <=3.20.2 ; platform_system == "Windows"
Pillow
decorator
astor
paddle_bfloat==0.1.7
opt_einsum==3.3.0
pyyaml
```

### 步骤5：编译 PaddlePaddle

首先从 GitHub 克隆 PaddlePaddle 代码库，然后在 develop 分支下进行编译。完成后，可以在 Paddle/build/python/dist 目录下找到生成的 .whl 文件，并使用 pip 命令进行安装。

 ```bash
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
# 切换到develop分支下进行编译
git checkout develop
# 创建并进入一个叫 build 的目录下：
mkdir build && cd build
# 设置进程允许打开的最大文件数
ulimit -n 4096
# 执行 cmake
cmake .. -DPY_VERSION=3.9.16 -DPYTHON_EXECUTABLE=`which python3` -DWITH_ARM=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF -DWITH_GPU=OFF
# 编译
make TARGET=ARMV8 -j$(nproc)
# 编译成功后进入Paddle/build/python/dist目录下找到生成的.whl包。
pip install -U（whl 包的名字)
 ```

> `nproc` 是一个Linux命令，它返回当前系统上可用的处理器数量（CPU核心数）。当你在 `make` 命令中使用 `-j$(nproc)` 参数时，你告诉 `make` 可以同时进行的最大任务数等于你的处理器数量。这样，`make` 将会并行执行多个任务，以加速编译过程。如果你不确定你的处理器数量，你可以直接在命令行中运行 `nproc` 命令来查看，这将会输出一个数字，代表你的处理器数量。

## 2、针对龙芯的编译步骤



## 3、验证安装

安装完成后，可以进入 python 解释器，输入`import paddle` ，然后输入 `paddle.utils.run_check()`进行验证。

```python
import paddle

paddle.utils.run_check()
```

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。

在 mobilenetv1 和 resnet50 模型上测试

```bash
wget -O profile.tar https://paddle-cetc15.bj.bcebos.com/profile.tar?authorization=bce-auth-v1/4409a3f3dd76482ab77af112631f01e4/2020-10-09T10:11:53Z/-1/host/786789f3445f498c6a1fd4d9cd3897ac7233700df0c6ae2fd78079eba89bf3fb
tar xf profile.tar && cd profile
python resnet.py --model_file ResNet50_inference/model --params_file ResNet50_inference/params
# 正确输出应为：[0.0002414  0.00022418 0.00053661 0.00028639 0.00072682 0.000213
#              0.00638718 0.00128127 0.00013535 0.0007676 ]
python mobilenetv1.py --model_file mobilenetv1/model --params_file mobilenetv1/params
# 正确输出应为：[0.00123949 0.00100392 0.00109539 0.00112206 0.00101901 0.00088412
#              0.00121536 0.00107679 0.00106071 0.00099605]
python ernie.py --model_dir ernieL3H128_model/
# 正确输出应为：[0.49879393 0.5012061 ]
```

## 4、卸载

请使用以下命令卸载 PaddlePaddle：

```bash
pip uninstall paddlepaddle
```

或

```bash
pip3 uninstall paddlepaddle
```

## 5、Problem

### git clone慢

使用镜像地址加速github

```bash
cd Paddle
vim CMakeLists.txt
```

在165行set(GIT_URL "https://github.com")修改为set(GIT_URL "https://ghproxy.com/https://github.com")



