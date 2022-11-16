
This project is a compiler for GPU Applications which aims to increase reliability.

## INSTALL
```
git clone https://github.com/parsiyte/GPU-Kernel-Redundancy.git
mkdir build
cd build
cmake -G Ninja -DLLVM_ENABLE_PROJECTS='clang;llvm' -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" ../llvm
ninja
```
Please keep in mind that "ninja" command might fail due to memory usage. It resumes where it left off when restarted.

USAGE

Modify the source code by inserting the following pragma notation after the function call you wish to increase its reliability.

``#pragma redundant in <INPUT> out <OUTPUT> Scheme SCHEME``

Fill the <INPUT> and <OUTPUT> field as required. Keep in mind that the stated output must be the last argument in the function call.

Modify the example Makefile by changing LLVM_LOCATION, CUDA_LOCATION, SM, and CP according to your settings and GPU features.
Then execute the make command by including SCHEME variable.

Keep in my that we used CUDA-10. Best of our knowledge, LLVM-10 had problem with subsequent CUDA releases.

## DEMO APPLICATIONS

```
git clone https://github.com/parsiyte/FTGPGPU-apps.git
cd FTGPGPU-apps
cd 2DConvolution
make INPUT_SIZE=LARGE_DATASET SCHEME=MKE
```
