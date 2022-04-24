# Update 1

### Things That Work

* Convert Arbitrary 3D Model to 2D Model
  * Identify major Laser Cut Axis
  * Project, reduce points; determine exterior shape
* Convert 2D Model to SVG for existing packing programs
* Rasterize (efficiently) a 2D Model into a 2D Representation
* Pack 2D Rasters over a region
  * Currently done with an OpenCV Hack based on template matching

### Things That Need Work

* Proper GPU based packing
* Proper Comparison against Existing Methods
* More Approachable

### In Many Words

The work is proceeding largely according to plan. I spent more time than intended working with various geometric model types. For example, I scanned my hard drives and located approximately 6,000 3D models of various formats. Some, but not all, of these files are items that could be laser cut (i.e. are principally 2D shapes with a constant thickness).  I decided that it was really important that I used these real-world files for this project; as opposed to randomly generating 2D polygons. This was an excellent learning opportunity; I've become very familiar with manipulating and converting various 3D file formats: SolidWorks Part (sldprt), Autodesk Inventor Part (.ipt), and STL files. I also developed a simple algorithm for determining if an arbitrary STL is a 2.5D part, and getting the corresponding 2D polygon (including with holes). This polygon representation can be exported to a SVG (for working with existing CPU-Based packing solutions) and with can be converted to a Raster representation (for use with my system).

After this, I wrote a quick and dirty CPU based packer that utilizes an OpenCV hack to determine valid placements. I represent the sheet as all white pixels and the raster as all white pixels (the empty/holey regions are represented as black). Then I perform a template match over the sheet with the raster masked with the raster. This looks for a matching shape/size region that is all white and ignores the black parts of the raster. When a valid placement is found, the raster is XOR'ed with the  sheet, turning the taken positions to black. This is a rather slow function, but it was quick to write and has an optional GPU acceleration. However, I probably will not explore this: the template matching will look for all positions that the raster fits and thus inherently will always do quite a bit of extra work. Time is better spent moving directly forward, rather than forward-ish.

The focus now is to take the raster representation and write a more efficient, GPU (CUDA) backed, algorithm. One point of concern is the visibility of the results. Up to this point, I have been working in Python 3.9. I have many good visualization functions, and will probably continue to call them to generate graphics. The question is the GPU (CUDA C/C++) interface with these functions. It is conceptually possible to use Python bindings to call appropriate C/C++: NumPy, a popular C based library for numeric computing in Python. Python bindings is also something that I want to explore so if it is easy enough, it will happen. 

Things are moving at a comfortable pace. The focus of the next few weeks is to focus on the GPU based acceleration.