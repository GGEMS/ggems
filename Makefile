CUDA_FLAGS := --generate-code arch=compute_30,code=sm_30 --relocatable-device-code true -lcudadevrt --compiler-options '-fPIC' -use_fast_math
#CUDA_FLAGS := --generate-code arch=compute_30,code=sm_30 -Xptxas="-v" --relocatable-device-code true -lcudadevrt --compiler-options '-fPIC' -use_fast_math

FLAGS := -lggems -lelectron -lelectron_navigator -lglobal -lmain_navigator -lmeshed -lphoton -lphoton_navigator -lprng -lproton -lproton_navigator -lraytracing -lsphere -lstructures -lvector -lvoxelized -lparticles -lsandia_table -lshell_data -lfun -lcross_sections_builder


G4DIRHEADERS = $(G4HOME)/include/Geant4
G4DIRLIBS = $(G4HOME)/lib
CLHEPHEADERS = $(CLHEPHOME)/include
CLHEPLIBS = $(CLHEPHOME)/lib

G4INCLUDES = -I$(G4DIRHEADERS) -I$(CLHEPHEADERS)
G4LIBS = -L$(G4DIRLIBS) -lG4materials -lG4global -lG4particles -lG4processes -lG4intercoms 
#-L$(CLHEPLIBS) -lCLHEP-2.1.1.0

SOURCES = $(wildcard */*.cu)

BUILDDIR = build
LIBDIR = lib

# totoall : dir $(patsubst %.cu,%.o, $(wildcard */*.cu)) 
# 	$(patsubst %.cu,%.o, $(wildcard */*.cu)) 
# 	make install
	
all: clean dir $(patsubst %.cu,%.o, $(wildcard */*.cu)) 
	make copy
	make install

dir:
	mkdir -p $(BUILDDIR)
	mkdir -p $(LIBDIR)

copy :
	mv */*.o $(BUILDDIR)
	
	
#biggems:
#	nvcc $(CUDA_FLAGS) -c -o ggems.o global/ggems.cu -Llib -laabb -lbuilder -ldosimetry -lelectron -lelectron_navigator -lfun -lglobal -lmain_navigator -lmeshed -lphoton -lphoton_navigator -lprng -lproton -lproton_navigator -lraytracing -lsphere -lstructures -lvector -lvoxelized $(G4INCLUDES) $(G4LIBS)	

%.o: %.cu 
	nvcc $^ -c -o $@ $(G4INCLUDES) $(G4LIBS) $(CUDA_FLAGS)
	
#processes/photon.o: processes/photon.cu
#	nvcc $(CUDA_FLAGS) $^ -c -o $@ -lsandia_table -lshell_data $(G4INCLUDES) $(G4LIBS)

# processes/sandia_table.o: processes/sandia_table.cu
# 	nvcc $(CUDA_FLAGS) $^ -c -o $@ $(G4INCLUDES) $(G4LIBS)
# 	mv $@ $(BUILDDIR)
# 	ar r -o $(LIBDIR)/libsandia_table.a $(BUILDDIR)/*.o
	
#global/ggems.o: global/ggems.cu
#	nvcc $(CUDA_FLAGS) $^ -c -o $@ $(G4INCLUDES) $(G4LIBS) -L$(LIBDIR) -lsandia_table -lshell_data -lcross_sections_builder	
	

	
#geometry/materials.o: geometry/materials.cu
#	nvcc $(CUDA_FLAGS) $^ -c -o $@ $(G4INCLUDES) $(G4LIBS)
	
buildo: */*.cu
	nvcc */*.cu -c -o $(BUILDDIR)/ggems.o $(CUDA_FLAGS)
	
#buildso :
#	nvcc -shared -arch=sm_30 -o lib/libggems.so $(BUILDDIR)/ggems.o
	
builda : $(BUILDDIR)/*.o
	ar r -o $(LIBDIR)/libggems.a $(BUILDDIR)/*.o
	
#shared: $(patsubst $(BUILDDIR)/%.o, $(LIBDIR)/lib%.so, $(wildcard $(BUILDDIR)/*.o))
	
#lib/lib%.so: $(BUILDDIR)/%.o
#	nvcc -shared -arch=sm_30 -o $@ $^ -L$(LIBDIR) 
	
#lib/libphoton.so : $(BUILDDIR)/photon.o
#	nvcc -shared -arch=sm_30 -o $@ $^ -L$(LIBDIR) -lfun -lstructures -lprng
	
# 	$(BUILDDIR)/photon.o $(BUILDDIR)/fun.o $(BUILDDIR)/structures.o $(BUILDDIR)/prng.o
	
#lib/libraytracing.so : $(BUILDDIR)/raytracing.o 
#	nvcc -shared -arch=sm_30 -o $@ $^ -L$(LIBDIR) -lvector

# lib/libraytracing.so : $(BUILDDIR)/raytracing.o lib/libvector.so
# 	nvcc -shared -arch=sm_30 -o $@ $^ -L$(LIBDIR) -lvector
	
install: $(patsubst $(BUILDDIR)/%.o, $(LIBDIR)/lib%.a, $(wildcard $(BUILDDIR)/*.o))

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
	rm *~ | true
	
	
