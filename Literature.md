# Literature Review

_The proposal introduces a variant of the bin packing problem; this section formalizes it_

At its core, the algorithm consists of the following key elements:

* A two-dimensional __bin packing__ algorithm
* For __irregular polygons__
* That runs on a __GPU__
* Backed by a raster-based implementation
* Produces acceptable results in reasonable time

To some extent, building the system is a arguably less complex problem. There are certainly challenges to the construction of such a system, and research-backed design decisions that could significantly impact the resulting efficiency. However, for the sake of this project (and thus this literature review), this information is relatively unimportant and omitted. 

## A two-dimensional bin packing algorithm

The proposed algorithm is a specific variant of the two-dimensional bin packing problem. In this problem, polygons are packed into bins (described by a bounding polygon) such that the total number of bins is minimized, an NP-Hard problem [Lodi]. There exists several metrics by which the TODO

## For irregular polygons

Irregular polygons introduce significant additional challenges. Calculating intersection between two polygons becomes a significant computation [Chen]. Likewise, the selection of polygons ordering will impact the results. Chen et. al. report a 20% difference in utilization attributable to the selection algorithm alone. The best performing, largest polygon first, has 80% utilization while smallest polygon first has only a 60% utilization. 

TODO - elaborate

## That runs on a GPU

There is significant literature on the use of a GPU.

TODO


There exists an interesting example of GPU accelerated processing by Wei Yang on Github [Yang]. This example focuses on packing rectangular polygons based on a binary-tree subdivision of the space. The example claims a nine times improvement over a CPU based implementation of the same algorithm. This is an interesting example of potential improvement mad possible though GPU based processing.


## Backed by a raster-based implementation

The algorithm's idea is to rasterize the polygons into a reduced representation and then try to place these rasterized polygons over the image in an efficient manner. The first half is a studied area. Li et. al. performed rasterization to GID problems and showed that, while approximating the higher resolution vector representation, still maintained a relatively high degree of accuracy. Li et. al. also demonstrated the ability to use these approximations in computing boundary length and moment of intertia, which may prove useful in this project.   

It is the latter part, the placement of rasterized polygons over the bin, that is perhaps the most confusing hole in present research. This is a relatively simple operation. It can be seen as equivalent to using OpenCV `matchTemplate()` or `filter2D()`, both of which scans a matrix over a larger image to find matching regions [OpenCV]. In this specific case, the source image would be the rasterized polygon and the base image the available space inside the bin. Both of the functions support CUDA acceleration [OpenCV] already. Therefore the question arises as to why such implementations are not well researched. It is likely that representations produce less accurate results, but the potential trade off in recovered computation time may offset this cost.

However, at this point we must return to Chen et. al.: _Two-Dimensional Packing For Irregular Shaped Objects_. This paper explores both a rasterized (referred to as rectilinear) representation of objects and a polygon representation. They conclude that the rectilinear representation, while providing benefits in computing area and intersections, significantly reduces the solutions available. Never the less, they still show success in using and packing objects with a rectilinear representation.

## In reasonable time

It is difficult to compare times directly with one another, but establishing a rough sense of scale is possible.

For Chen et. al., a greedy algorithm ran in the order of 1000s to low 10000s of seconds and more complex genetic algorithms in low to high 10000s of seconds. A Tabu Search variant showed much more variance, resolving some datasets in mid 100s of seconds and others in about 10000 seconds.

TODO

Of course, the times set out by these papers are hardware dependent, and hardware has improved significantly in the up to 20 years since publication. Therefore, a more fair time measurement is needed. 

There exists two open-source CPU based nesting program: SVGnest and Deepnest. These programs will serve as a benchmark for both the packing and time efficiency of the algorithm. These programs both run continuously, searching for better inputs until the user stops the program. Thus it should be possible to develop a fitness vs time function for any input data and determine where the boundary between methods occurs. 

## Comments on Polygons

So far, the specific polygons have been kept relatively arbitrary, but discussing a more formal design of the polygons is potentially very important. There is not a ground-truth dataset. Therefore, it will be necessary to design a representative sampling of data over the 