# GGEMS makefile
#
# JB 20/03/2016
################################################################

## Users options ###############################################

# Options
FAST_MATH_ENABLED := yes
FLOAT_DOUBLE_PRECISION := yes
VERBOSE := no
DEBUG := yes

# SM targetted
SMS := 20 30 35 37 50 52

# Main directory and sources
BUILDDIR := build
SRCDIR := src

# Release directories
RELEASEDIR := release
BINDIR := $(RELEASEDIR)/bin
LIBDIR := $(RELEASEDIR)/lib
DATADIR := $(RELEASEDIR)/data
INCDIR := $(RELEASEDIR)/include
DOCDIR := $(RELEASEDIR)/doc

################################################################
## FLAGS #######################################################
################################################################

NVCC_FLAGS := --relocatable-device-code true -lcudadevrt --compiler-options -w --std=c++11
#NVCCFLAGS := --relocatable-device-code true -lcudadevrt --compiler-options '-fPIC'

ifeq ($(FAST_MATH_ENABLED),yes)
NVCC_FLAGS += -use_fast_math
endif

# By default double is used in the code
ifeq ($(FLOAT_DOUBLE_PRECISION),no)
NVCC_FLAGS += --define-macro SINGLE_PRECISION
endif

ifeq ($(VERBOSE),yes)
NVCC_FLAGS += -Xptxas="-v"
endif

ifeq ($(DEBUG),yes)
NVCC_FLAGS += --define-macro DEBUG
endif

# Generate code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval NVCC_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

###############################################################
## COMPILATION ################################################
###############################################################

VPATH := ${SRCDIR}/detectors \
         ${SRCDIR}/data \
         ${SRCDIR}/geometries \
         ${SRCDIR}/global \
         ${SRCDIR}/tools \
         ${SRCDIR}/navigators \
         ${SRCDIR}/processes \
         ${SRCDIR}/sources

# Glob CU
SOURCES :=
$(foreach dir,$(VPATH),$(eval SOURCES += $(wildcard $(dir)/*.cu)))

# Glob Include Dir
INC_FLAGS :=
$(foreach dir,$(VPATH),$(eval INC_FLAGS += -I$(dir)))

#$(info $(NVCC_FLAGS))


OBJ := $(patsubst %.cu,$(BUILDDIR)/%.o, $(notdir $(SOURCES)))
	
#$(info $(OBJ))

all: dir $(OBJ)
	# nothing

dir:
	mkdir -p $(BUILDDIR)

copy :
	mv */*.o $(BUILDDIR)
	
$(BUILDDIR)/%.o : %.cu
	nvcc -c $< -o $@ $(NVCC_FLAGS) $(INC_FLAGS)


#%.o: %.cu 
#	nvcc -c $^ -o $(BUILDDIR)/$(notdir $@) $(NVCC_FLAGS) $(INC_FLAGS)
	

#biggems:
#	nvcc $(CUDA_FLAGS) -c -o ggems.o global/ggems.cu -Llib -laabb -lbuilder -ldosimetry -lelectron -lelectron_navigator -lfun -lglobal -lmain_navigator -lmeshed -lphoton -lphoton_navigator -lprng -lproton -lproton_navigator -lraytracing -lsphere -lstructures -lvector -lvoxelized $(G4INCLUDES) $(G4LIBS)	

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
	
