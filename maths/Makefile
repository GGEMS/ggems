CUDA_FLAGS := -m64 -arch=sm_30 --shared -Xcompiler -fPIC
CUDALIBS := 

CUDA_FLAGS2 := --generate-code arch=compute_30,code=sm_30 -Xptxas="-v" --relocatable-device-code true -lcudadevrt


# all: $(patsubst %.cu, lib%.so, $(wildcard *.cu))
all:  $(wildcard *.cu)

lib%.so: %.cu %.cuh
	nvcc $(CUDA_FLAGS) -c -o $@ $<


%.o: %.cu %.h
	nvcc -arch=sm_30 -dc -o $@ $<
	

libmaths.so: *.cu *.h
	nvcc $(CUDA_FLAGS) -c -o $@ fun.cu
	
%.cu : %.cu 
	nvcc  $(CUDA_FLAGS) -c -o $@ 
	
test : 
	nvcc --generate-code arch=compute_30,code=sm_30 -Xptxas="-v" --relocatable-device-code true -lcudadevrt *.cu -c
	
# link :
# 	nvcc -arch=sm_30 *.o -dlink -o maths.o
	
# libfun.so: fun.cu fun.h
# 	nvcc $(CUDA_FLAGS) $(CUDALIBS) -c -o $@ $<

# libvector.so : vector.cu vector.h
# 	nvcc $(CUDA_FLAGS) $(CUDALIBS) -c -o $@ $<

libraytracing.so : raytracing.cu raytracing.cuh
	nvcc $(CUDA_FLAGS) $(CUDALIBS) -c -o $@ $< -l/home/yannick/GGEMS/maths/lvector
	
	
install :
	cp *.so ../lib
	
clean :
	rm *.so | true
	rm *.o | true