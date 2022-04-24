# Update 2

_TLDR: things work; things do not work well_

First things first, I really didn't have the time and dedication to work on this as much as I would have liked. The past month has been busy both with school and otherwise. However, I did manage to find some time to setup and write an initial version of the GPU-backed packing algorithm. It works; it will return a result, eventually. However, it does not work well. In fact, when factoring in the copying time, it is probably worse than a CPU-only method. There are several reasons for this.

The problem as designed is, roughly, to do an image convolution, such as a Gaussian blur. A shocking discovery was that there is no built-in CUDA utility to perform this natively. Fortunately there are plenty of open-source implementations of this process. However, there are two key differences between our algorithm here and a true image convolution

* A normal image convolution is small, usually acting on 3x3 to 9x9 region of an image. My "convolution" would act on a roughly 16x500 region. This is too big to be done efficiently with a single thread for the convolution. Thus, both the image and the convolution need to be partitioned across multiple GPU threads.
* A convolution is run over the whole image; the packing can stop after the first valid placement is found.

_I'm trying to avoid using the word Kernel since both the convolution and the GPU use the word kernel to mean radically different things_

Together, these two factors motivate the development of a custom and optimized packing algorithm. And here is where things get crazy. There are multiple competing constraints that need to satisfied and that are all competing with one another:

* A single thread is very weak, many threads are very strong. 
* There is a limit of GPU threads in a GPU block (this limit is 1024).
* There is fast software cache shared memory per-block. It is really really fast, but also limited. Using the maximum number (1024) of threads per block may require more global memory accesses, which is slow.
* This shared memory has a unique structure that requires certain access patterns for maximum performance.
* Global memory requires certain access patterns for maximum performance.
* There is (formally) no synchronization between GPU blocks. Attempting to implement synchronization may deadlock the entire process.
  * It is possible to synchronize within a block.
* Starting and stopping multiple blocks is slow. Ideally, the blocks produce an answer before exiting. Also, starting and stopping blocks will complicate the shared per-block memory and may require re-fetching from global memory.
* Copying between CPU (RAM) and GPU is slow. Ideally, all the packing should occur exclusively on the GPU.

All of these are design constraints that define solutions and performance. Writing a robust algorithm for doing so is challenging. The memory access patterns is hard enough, coupled with the lack of inter-block synchronization. 

I have ideas for next steps. I also fortunately have more time now to test them. Really the next several weeks is for working on refining this algorithm. There are also simple auxiliary items to do. Right now the workflow requires running several python scripts and manually copying the results around. Likewise, I do not have a good output visualizer to generate graphics quickly and easily. 