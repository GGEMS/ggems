CUDA_FLAGS := -m64 -arch=sm_30 --shared -Xcompiler -fPIC
CUDALIBS := -L/usr/local/cuda/lib64 -lcudart
CUDA_FLAGS2 := --generate-code arch=compute_30,code=sm_30 -Xptxas="-v" --relocatable-device-code true -lcudadevrt

SOURCES = $(wildcard */*.cu)
BUILDDIR = build
OBJECTS = $(patsubst */*.cu,build/%.o,$(SOURCES))
# all: $(patsubst %.cu, lib%.so, $(wildcard *.cu))
# all:  $(wildcard *.cu)

all: dir $(wildcard */*.cu) copy

dir:
	mkdir -p $(BUILDDIR)

print:
	
	ls $(SOURCES)
	echo "......."
	ls $(OBJECTS)

lib%.so: %.cu %.h
	nvcc $(CUDA_FLAGS) -c -o $@ $<
	
libstructures.so: structures.cu structures.cuh
	nvcc $(CUDA_FLAGS) $(CUDALIBS) -c -o $@ $<
	
libphoton.so: photon.cu photon.h
	nvcc $(CUDA_FLAGS) $(CUDALIBS) -c -o $@ $<
	

$(OBJECTS): %.cu : %.cu %.cuh
# 	nvcc  $(CUDA_FLAGS) -c -o $@ $<
	nvcc --generate-code arch=compute_30,code=sm_30  --relocatable-device-code true -lcudadevrt $(OBJECTS) -c
	
install:
	cp */*.o $(BUILDDIR)
	
	
test : dir
	cd detector; make test
	cd global; make test
	cd maths; make test
	cd navigation; make test
	cd processes; make test
	cp detector/*.o $(BUILDDIR)
	cp global/*.o $(BUILDDIR)
	cp maths/*.o $(BUILDDIR)
	cp navigation/*.o $(BUILDDIR)
	cp processes/*.o $(BUILDDIR)
	
clean :
# 	rm *.so | true
	rm */*.o | true
	rm */*.so | true
	rm */*~ | true
	