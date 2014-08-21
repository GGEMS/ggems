CUDA_FLAGS := -m64 -arch=sm_30 --shared -Xcompiler -fPIC
CUDALIBS := -L/usr/local/cuda/lib64 -lcudart
CUDA_FLAGS2 := --generate-code arch=compute_30,code=sm_30 -Xptxas="-v" --relocatable-device-code true -lcudadevrt


all: $(patsubst %.cu, lib%.so, $(wildcard *.cu))
# all:  $(wildcard *.cu)

print:
	$(patsubst %.cu, lib%.so, $(wildcard *.cu))

lib%.so: %.cu %.cuh
	nvcc $(CUDA_FLAGS) $(CUDALIBS) -o $@ $< -c
	
# libstructures.so: structures.cu structures.cuh
# 	nvcc $(CUDA_FLAGS) $(CUDALIBS) -c -o $@ $<
	
# libphoton.so: photon.cu photon.cuh
# 	nvcc $(CUDA_FLAGS) $(CUDALIBS) -c -o $@ $<
	
install :
	cp *.so ../lib
	
# %.cu : %.cu %.cuh
# 	nvcc  $(CUDA_FLAGS) -c -o $@ $<
	
test : 
	nvcc $(CUDA_FLAGS2) *.cu -c
	
clean :
	rm *.so | true
	rm *.o | true
	rm *~ | true