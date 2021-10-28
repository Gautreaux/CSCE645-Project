CXX:=g++# which compiler to use
CPPFLAGS=# empty default value for GCC flags, can be overriden on command line
CPPSTD=c++17# which c++ version to use
CPPFLAGS_COM=-Wall -Wextra -Wpedantic -Werror # annoying, but good flags

CPP_COMP_COM=$(CXX) $(CPPFLAGS) $(CPPFLAGS_COM) -std=$(CPPSTD) 
CC_COMPLIE_NO_LINK_AUTO=$(CPP_COMP_COM) -c -o $@ $<
CC_COMPILE_LINK_EXE_AUTO=$(CPP_COMP_COM) -o $@ $< OFiles/*.o

EXECUTABLE=Main

$(EXECUTABLE) : Main.cpp OFiles/Raster.o
	$(CC_COMPILE_LINK_EXE_AUTO)

.PHONY: $(EXECUTABLE) tests clean

force : clean $(EXECUTABLE)

clean :
	rm -f *.o
	rm -f $(EXECUTABLE)
	rm -f OFiles/*.o

release : clean
	make CPPFLAGS="$(CPPFLAGS) -Ofast"

debug : clean
	make CPPFLAGS="$(CPPFLAGS) -g"

# rules for creating o files
OFiles/Raster.o : Raster.cpp Raster.hpp
	$(CC_COMPLIE_NO_LINK_AUTO)