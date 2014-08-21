CUDA_FLAGS := -m64 -arch=sm_30 --shared -Xcompiler -fPIC
CUDA_FLAGS2 := --generate-code arch=compute_30,code=sm_30 -Xptxas="-v" --relocatable-device-code true -lcudadevrt

all: $(patsubst %.cu, lib%.so, $(wildcard *.cu))

lib%.so: %.cu %.h
	nvcc $(CUDA_FLAGS) -c -o $@ $<
	
install :
	cp *.so ../lib
	
test : 
	nvcc $(CUDA_FLAGS2) *.cu -c
	
clean :
	rm *.so | true