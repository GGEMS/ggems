CUDA_FLAGS := --generate-code arch=compute_30,code=sm_30 -Xptxas="-v" --relocatable-device-code true -lcudadevrt --compiler-options '-fPIC' -use_fast_math

SOURCES = $(wildcard */*.cu)

BUILDDIR = build
LIBDIR = lib
SOURCEDIR=sources


all: clean dir $(patsubst %.cu,%.o, $(wildcard */*.cu)) 
	make copy
	make install

dir:
	mkdir -p $(BUILDDIR)
	mkdir -p $(LIBDIR)

copy :
	mv */*.o $(BUILDDIR)

%.o: %.cu
	nvcc $(CUDA_FLAGS) $^ -c -o $@
	
buildo: */*.cu
	nvcc $(CUDA_FLAGS) */*.cu -c -o $(BUILDDIR)/ggems.o
	
buildso :
	nvcc -shared -arch=sm_30 -o lib/libggems.so $(BUILDDIR)/ggems.o
	
builda : $(BUILDDIR)/*.o
	ar r -o $(LIBDIR)/libggems.a $(BUILDDIR)/*.o
	
shared: $(patsubst $(BUILDDIR)/%.o, $(LIBDIR)/lib%.so, $(wildcard $(BUILDDIR)/*.o))
	
lib/lib%.so: $(BUILDDIR)/%.o
	nvcc -shared -arch=sm_30 -o lib/libtest.so $(BUILDDIR)/*.o
	
install: $(patsubst $(BUILDDIR)/%.o, $(LIBDIR)/lib%.a, $(wildcard $(BUILDDIR)/*.o))

sources:
	mkdir -p $(SOURCEDIR)
	cp */*.cuh $(SOURCEDIR)
	
lib/lib%.a: $(BUILDDIR)/%.o
	ar r -o $@ $^
	
clean :
# 	rm *.so | true
	rm */*.o | true
	rm */*.so | true
	rm */*~ | true
	rm */*.a | true
	rm *~ | true
	
cleanall: 
	rm -rf $(BUILDDIR) | true
	rm -rf $(LIBDIR) | true
	rm -rf $(SOURCEDIR) | true
	rm *~ | true
	
	
