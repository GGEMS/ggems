CUDA_FLAGS := -m64 -arch=sm_30 --shared -Xcompiler -fPIC
CUDALIBS := -L/usr/local/cuda/lib64 -lcudart
CUDA_FLAGS2 := --generate-code arch=compute_30,code=sm_30 -Xptxas="-v" --relocatable-device-code true -lcudadevrt


# all: $(patsubst %.cu, lib%.so, $(wildcard *.cu))
all:  $(wildcard *.cu)

lib%.so: %.cu %.h
	nvcc $(CUDA_FLAGS) -c -o $@ $<
	
libstructures.so: structures.cu structures.cuh
	nvcc $(CUDA_FLAGS) $(CUDALIBS) -c -o $@ $<
	
libphoton.so: photon.cu photon.h
	nvcc $(CUDA_FLAGS) $(CUDALIBS) -c -o $@ $<
	
install :
	cp *.so ../lib
	
%.cu : %.cu %.cuh
	nvcc  $(CUDA_FLAGS) -c -o $@ $<
	
test : 
	nvcc --generate-code arch=compute_30,code=sm_30 -Xptxas="-v" --relocatable-device-code true -lcudadevrt *.cu -c
	
clean :
	rm *.so | true
	rm *.o | true
	rm *~ | true