CXX:=nvcc# which compiler to use

CPP_FLAGS=--gpu-architecture=compute_61
CPP_COMP_COM=$(CXX) $(CPP_FLAGS)
CC_COMPLIE_NO_LINK_AUTO=$(CPP_COMP_COM) -c -o $@ $<
CC_COMPILE_LINK_EXE_AUTO=$(CPP_COMP_COM) -o $@ $< OFiles/*.o

EXECUTABLE=Main

$(EXECUTABLE) : Main.cu OFiles/Packer.o pch.h env.hpp OFiles/RasterUtils.o OFiles/CudaMemManager.o
	$(CC_COMPILE_LINK_EXE_AUTO)

.PHONY: $(EXECUTABLE) tests clean force run

force : clean $(EXECUTABLE) run

clean :
	rm -f *.o
	rm -f $(EXECUTABLE)
	rm -f OFiles/*.o
	mkdir -p OFiles/

run:
	./$(EXECUTABLE)

# rules for creating o files
OFiles/Raster.o : Raster.cpp Raster.hpp pch.h
	$(CC_COMPLIE_NO_LINK_AUTO)

OFiles/RasterUtils.o : RasterUtils.cpp RasterUtils.hpp pch.h
	$(CC_COMPLIE_NO_LINK_AUTO)

OFiles/Packer.o : Packer.cu Packer.cuh OFiles/Raster.o pch.h
	$(CC_COMPLIE_NO_LINK_AUTO)

OFiles/CudaMemManager.o : CudaMemManager.cu CudaMemManager.cuh OFiles/Raster.o pch.h
	$(CC_COMPLIE_NO_LINK_AUTO)
