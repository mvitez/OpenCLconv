# OpenCLconv
2D convolution experiments on Nexus 5 with OpenCL

This is a suite of tests that performs 512x512 image convolution with a 9x9 kernel using OpenCL.

I tried every combination of float, half (16 bit float), char, scalar or vector (with 4 or 16 components).
I tried to use regular buffers for the image (and regular operations to access it) and image2D buffers (and special functions to access its pixels).

The maximum performance I could obtain was 2.2 Gop/s.

If OpenCL is used with the ARM CPU, the performance is better (5 Gop/s).

Just for comparison, the same code on the GeForce GTX 690 runs @ 465 Gop/s using only one of the two cores.

I tried to do partial sums, use #pragma unroll, give different optimization options to the OpenCL compiler, precalculate some variables, there was no change, so it looks that the compiler did its job in optimization. Doing partial sums actually worsened the situation, so it has no sense to try to optimize the C code by yourself, it's better to leave this job to the compiler.


Instructions:

Compilation (you need Android NDK, I used android-ndk-r9d):

arm-linux-androideabi-gcc -O3 -march=armv7-a -mfloat-abi=softfp -mfpu=neon -I. clconv.c -o clconv libOpenCL.so

(change clconv.c to whatever source you are compiling, they are all independent).

Run:

Install SSHServer from Ice Cold Apps, configure it and copy clconv and libOpenCL.so to

/data/data/com.icecoldapps.sshserver

with SFTP.

SSH to Nexus 5 and give these commands:

cd /data/data/com.icecoldapps.sshserver
chmod 0755 clconv
LD_LIBRARY_PATH=.
./clconv

Be aware that you don't have list access to /data and /data/data (ls will fail).

libOpenCL.so is taken from here:
https://maxlv.net/how-to-enable-opencl-on-nexus-5/

I have tried also with the latest drivers from Qualcomm (necessary to run the clconv_cpu.c, because the drivers present on the device lack the CPU OpenCL compiler):

https://developer.qualcomm.com/mobile-development/maximize-hardware/mobile-gaming-graphics-adreno/tools-and-resources

In this case you need to copy to the device all the libraries present int Qualcomm-libs and point LD_LIBRARY_PATH to them.

Don't read the instructions given in the websites given in the links above, it's not necessary to root the device, unlock the bootloader or anything like this, just copy the files to /data/data/com.icecoldapps.sshserver, where you have access with SFTP, but enter directly in that directory, SFTP cannot list the /data directory, nor ssh can.
