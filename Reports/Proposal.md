# Proposal

> Disclaimer: I'm actually expecting to not complete this project, or at least be unhappy with the results. This is a very very hard problem and I am well aware of this fact. Finding a solution over the next six-ish weeks will be difficult. Really, I just wanted a project where I could practice GPU programming.

### A word on why

For this project, I propose the development and implementation of a two-round raster based two dimensional polygon packing system. This system is modeled for a theoretical production workflow based on __Contract Manufacturing__ principles (a.k.a. __Manufacturing as a Service__). In this theoretical model, the manufacturer offers the service of two-dimensional sheet operations, such as CNC Routing. The manufacturer provides this service turn-key, where the customer uploads parts directly and receives physical parts sometime in the future. To offer this service at a competitive cost, the manufacturer must minimize waste of the stock material. Likewise the manufacturer must quickly and accurately price contracts.

These objectives of maximizing packing efficiency while also minimizing overhead time (and monetary) costs is a difficult problem. To solve the time problem, a massively parallel approach can be taken. However, constructing massively parallel systems comes with large initial costs for specialized hardware and ongoing costs for (likely significant) power consumption. In fact, it would be possible to develop a formula which relates power consumption, packing efficiency, and packing time, allowing one the be quantified in terms of the others.

However, there is a significantly cheaper alternative for constructing massively parallel arrays: __Graphics Processing Units__ (GPUs). For example, the Nvidia RTX 3070 contains 5888 __CUDA__ Cores, roughly equivalent to 5888 threads of execution. These cores are less powerful than a standard processor, but their sheer quantity more than makes up the difference. Assuming no memory limitations, this processor could sample 92 square inches at 8 dpi resolution by assigning one sample to each of the cores. The RTX 3070 is clocked at 1.50 GHz with 8 GB of GDDR6 RAM all at 220W of power consumption. This is a powerful device readily available at a low cost of _$499.00_.

Returning to our contract manufacturing company. The __GPU__ accelerated packing solves our time and costs constraints. However, an efficient packing is still of utmost importance. There exists some literature in this area; see the literature review section for more information. At a high level, there is still room for improvement, which is one key focus for this project.

While this is a hard problem to solve, there exists an equally important problem: that of the user interface. While the __GPU__ is a relatively common piece of hardware, it is not assumed to be ubiquitous or available for this purpose. Therefore, the contract manufacturer, wishing to provide the best service, offers the packing on dedicated GPUs. This cloud packing farm would be similar in design and function to that of a __render farm__ in which the service provided manages a group of resources (__GPUs__) to serve client requests in a managed, scalable, and transparent manner. Additionally this service would need to be secure and reliable, the latter being of particular note in light of Facebook's recent outage.

### A word on what

The project can be divided into two parts. First the packing algorithm. There are packing algorithms, and there are GPU accelerated packing algorithms. See the literature review section for a more through discussion. I am going to focus on a subset of the packing problem which is formalized in the following way. The problem is a derivative of the __NP-Hard__ two-dimensional __Bin Packing Problem__. In this problem two dimensional polygons, called parts, are packed into one or more larger two-dimensional bounding polygon, called sheets, so that the total number of sheets is minimized. These sheets can represent the available raw material from which to cut parts or machine work volume on which to cut parts. The parts can be regular or irregular, convex or concave. The parts may be rotated within the 2D plane to allow for better packing, but cannot be flipped. Finally, the parts may have interior holes (or concave regions) into which other parts can nest.

There exists some software, both open source and commercial, that can perform these calculations for some/all subsets of this problem. The open source variants run on the __CPU__ in a single or lightly threaded manner. These implementations will provide a benchmark for packing efficiency and time taken for the new algorithm. 

For the new algorithm, the expected workflow is to be __raster__ based. We can convert the part polygons into a raster based representation where black pixels (with value 0) are not filled and white pixels (with value 1) are filled. Placing a polygon in the bin involves sliding the mask of the part polygon over the bin (much like a __kernel convolution__ in image processing). As this is a greedy algorithm, there is much significance in the ordering of the parts for placement. This is discussed more in the literature review.

The second portion of this project focuses on migrating the prior portion to a simulated production environment. This is accomplished by distributing the algorithm across multiple compute nodes. Here there is potential for quite a few interesting area of work. First, serving contracts in a maximally efficient manner: can multiple nodes work concurrently to complete a contract in better time or with better efficiency. Alternatively, a scheduler for assigning contracts to nodes in a way to maximize node efficiency.

### Goals

The goals of this project are as follows:

* Part 1: An efficient GPU Packing algorithm
  * Find or generate test part polygons and test contracts either in a random manner or from some pre-constructed dataset.
  * Develop/Implement an algorithm for packing arbitrary, irregular polygons in 2D space, informed by prior work
  * Benchmark this algorithm against the existing a open source packing algorithms based principally on time and secondarily on efficiency.
  * Quantify the exchange between efficiency cost and time cost. 
* Part 2: A scalable Packing Farm
  * Develop a distributed system for organizing packing nodes into a larger service. 
  * Nodes should be able to cleanly enter and leave the pool, but crashes of the node and the scheduler are not expected.
  * Service contract in a timely and reliable manner.
* Stretch Goals:
  * A Proper Polygon based GPU accelerated implementation
  * Multi-node collaboration for _contract_ efficiency
  * Multi-node scheduling for _resource_ efficiency
  * Nice user interface for the Packing Farm