# OMPL_Python_bindings

针对多python环境，作了编译安装脚本修改，当前自用

# Steps to Install OMPL with Python Bindings

1. First, make sure you have all the required library dependencies installed:
```bash
sudo apt update
sudo apt install libboost-all-dev cmake libeigen3-dev
sudo apt install python3-pip python3-pyparsing python3-numpy
pip3 install pygccxml pyplusplus castxml
```

2. **IMPORTANT STEP!**: install the compatible versions of these libraries:
```bash
pip install 'importlib_metadata<4.0'
pip install 'pygccxml==2.2.1'
```

3. Navigate to your `home` (or `root`, if using Docker) directory, and clone the ompl github repository with the **python bindings fix**:

```bash
# git clone --depth 1 --branch fix/pybindings-generation-latest https://github.com/ompl/ompl.git
# 这一步拉取本仓库
```

4. Go to the `ompl` directory you just cloned:

```bash
cd ompl
```

5. Make a directory where you will build the OMPL library and go to the new directory

```bash
mkdir -p build/Release
cd build/Release
```

6. Run the cmake command with `PYTHON_BINDINGS` enabled and the libboost library specified:

```cmake
# 修改CMakeLists.txt，把所有带有'FIXME'字样的地方，修改成实际路径，一定要和python3.8的包对应上！！！
# 同时，要多留意cmake的输出，我添加了一些调试信息，一定要关于pybinding调试信息全部通过才行！！！
```

```bash
cmake ../.. -DOMPL_BUILD_PYBINDINGS=ON
```

7. Next, call the make command to generate the python bindings:

```bash
make -j $(nproc) update_bindings
```

8. Then, compile the OMPL library:

```bash
make j$(nproc)
```

9. Finally, install the library and bindings to your system (or container, if in Docker):

```bash
sudo make install
```

10. Confirm OMPL is installed with python bindings:

```bash
python
from ompl import base
```

If no error like this (below) pops up, then congratulations! You have installed OMPL with python bindings!

```bash
    from ompl.util._util import *
ModuleNotFoundError: NO module named 'ompl.util._util'

# 这种错误一般是因为castxml错误，这个包里py-bindings/generate_bindings.py已经改对了
# 那么错误的话就是因为cmake调取的castxml版本不对，可能还需要检查一下pygccxml版本是否为2.2.1
```

如果报错如下，那么必须检查boost.python和boost.numpy调取的包是否为选定的python3.8的环境下的包（这个必须一致，否则就会出现链接错误的问题）：

```bash
>>> from ompl import base
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/lib/python3/dist-packages/ompl/base/__init__.py", line 2, in <module>
    from ompl.base._base import *
ImportError: /usr/lib/python3/dist-packages/ompl/base/_base.so: undefined symbol: _ZN5boost6python5numpy11from_objectERKNS0_3api6objectERKNS1_5dtypeEiiNS1_7ndarray7bitflagE
```

*NOTE: If the error (above) pops up, check out this thread [(here)](https://github.com/ompl/ompl/issues/1110) and see possible fixes. It could be an issue with your PyPy library version, or some other library.*

11. **PS:** If you're using Docker, don't forget to commit the container after the library is installed!

```bash
docker commit <container_id or container_name> <image_id> 
```

12. 如果你需要将安装好的包，移入你需要的包目录下，可以进入`/usr/lib/python3/dist-packages`(可能不是这个，要去到你的python包安装路径下)，找到`ompl`文件夹，然后拷贝到你的工作环境下调取，暂未知能否跨机使用（理论上来说必须经过本机编译，或者通过github action批量制作不同系统环境下的预编译包wheel）。

---


The Open Motion Planning Library (OMPL)
=======================================

Continuous Integration Status
-----------------------------

[![Build](https://github.com/ompl/ompl/actions/workflows/build.yml/badge.svg?branch=pr-github-actions)](https://github.com/ompl/ompl/actions/workflows/build.yml)
[![Format](https://github.com/ompl/ompl/actions/workflows/format.yml/badge.svg?branch=pr-github-actions)](https://github.com/ompl/ompl/actions/workflows/format.yml?branch=pr-github-actions)

Installation
------------

Visit the [OMPL installation page](https://ompl.kavrakilab.org/core/installation.html) for
detailed installation instructions.

OMPL has the following required dependencies:

* [Boost](https://www.boost.org) (version 1.58 or higher)
* [CMake](https://www.cmake.org) (version 3.12 or higher)
* [Eigen](http://eigen.tuxfamily.org) (version 3.3 or higher)

The following dependencies are optional:

* [Py++](https://github.com/ompl/ompl/blob/main/doc/markdown/installPyPlusPlus.md) (needed to generate Python bindings)
* [Doxygen](http://www.doxygen.org) (needed to create a local copy of the documentation at
  https://ompl.kavrakilab.org/core)
* [Flann](https://github.com/flann-lib/flann/tree/1.9.2) (FLANN can be used for nearest neighbor queries by OMPL)
* [Spot](http://spot.lrde.epita.fr) (Used for constructing finite automata from LTL formulae.)

Once dependencies are installed, you can build OMPL on Linux, macOS,
and MS Windows. Go to the top-level directory of OMPL and type the
following commands:

    mkdir -p build/Release
    cd build/Release
    cmake ../..
    # next step is optional
    make -j 4 update_bindings # if you want Python bindings
    make -j 4 # replace "4" with the number of cores on your machine
