#!/usr/bin/env python
import os, sys, glob, subprocess

#
#
# GGEMS Builder manager - 6/10/2016 - JB
#
#

## some def
def printUsage():
    print('')
    print('.:: GGEMS Builder manager - v1 - JB ::.')
    print('')
    print('Usage:')
    print('  ./gbuilder.py compile [SourceDir]')
    print('  ./gbuilder.py install [InstallDir]')    

    sys.exit()

def printError(txt):
    print('[ERROR] -', txt)
    sys.exit()

def printCompile(txt, i, n):
    print('[%i/%i] Compiling %s' % (i+1, n, txt) )

### Read params ###################################

if len(sys.argv) <= 2:
    printUsage()

cmd        = sys.argv[1]
TargetPath = sys.argv[2]

# checking
if cmd not in ("compile", "install"):
    printUsage()

### Common params #################################

cudaHome = "/usr/local/cuda"

#################################################
###    Compile option    ########################
#################################################

if cmd == "compile":

    ### Options #####################################    

    FlagStatic          = True
    FlagFastMath        = True
    FlagDoublePrecision = True
    FlagDebug           = True
    FlagVerbose         = False
    deviceArch          = [20, 30, 35, 37, 50, 52]
    #deviceArch          = [20]
    nvccFlags           = "--relocatable-device-code true -lcudadevrt --compiler-options -w --std=c++11"

    sourceDir           = ( TargetPath+"detectors",  \
                            TargetPath+"geometries", \
                            TargetPath+"global", \
                            TargetPath+"navigators", \
                            TargetPath+"processes", \
                            TargetPath+"sources", \
                            TargetPath+"tools" )

    ### PreProcessing ###############################

    if FlagFastMath:
        nvccFlags += " -use_fast_math"

    if not FlagDoublePrecision:
        nvccFlags += " --define-macro SINGLE_PRECISION"

    if FlagDebug:
        nvccFlags += " --define-macro DEBUG"

    if FlagVerbose:
        nvccFlags += ' -Xptxas="-v"'

    for arch in deviceArch:
        nvccFlags += " -gencode arch=compute_%s,code=sm_%s" % (arch, arch)

    ### Print some info #################################

    txt = subprocess.check_output(["nvcc", "--version"]).split()
    txt = str(txt[-1])
    
    print("Cuda computation tools release %s" % txt)

    txt = subprocess.check_output(["g++", "--version"]).split()
    txt = str(txt[3])
    
    print("g++ computation tools release %s" % txt)
    print("")

    ### Compile all cu files ############################

    cuFiles = []

    # find cu files
    for dir in sourceDir:
        cuFiles.extend( glob.glob(dir+"/*.cu") )

    if len(cuFiles) == 0:
        printError("No cuda files were found!")

    # get obj names
    objFiles = []
    for file in cuFiles:
        name = os.path.basename(file)
        name = name.replace(".cu", ".o")
        objFiles.append(name)    

    # build include dir
    incDir = ""
    for dir in sourceDir:
        incDir += " -I%s" % dir

    if FlagStatic:
        # compilation
        n = len(cuFiles)
        for i in range(n):
            src = cuFiles[i]
            obj = objFiles[i]
            printCompile(src, i, n)

            os.system("nvcc -cudart static -dc %s -o %s %s %s" % (src, obj, nvccFlags, incDir))

        # compile the dynamic link
        allObj = ""
        for obj in objFiles:
            allObj += " %s" % obj

        print("* Compiling dynamic link")
        os.system("nvcc -cudart static -dlink %s -o link.o %s" % (allObj, nvccFlags))

        print("* Compiling ggems lib")
        os.system("nvcc --lib -o libggems.a link.o %s" % allObj)
        os.system("ranlib libggems.a")

    else:

        # compilation
        n = len(cuFiles)
        for i in range(n):
            src = cuFiles[i]
            obj = objFiles[i]
            printCompile(src, i, n)

            os.system("nvcc -Xcompiler '-fPIC' -dc %s -o %s %s %s" % (src, obj, nvccFlags, incDir))
            
        # compile the dynamic link
        allObj = ""
        for obj in objFiles:
            allObj += " %s" % obj

        print("* Compiling dynamic link")
        os.system("nvcc -Xcompiler '-fPIC' -dlink %s -o link.o %s" % (allObj, nvccFlags))

        # then compile the ggems library    
        print("* Linking ggems library")
        os.system("g++ -shared -o libggems.so %s link.o -L%s/lib64 -lcudart" % (allObj, cudaHome))

        # finally,  copy all includes to the build directory for convenience
        for dir in sourceDir:
            os.system("cp %s/*.cuh ." % dir)

#################################################
###    Install option    ########################
#################################################

if cmd == "install":

    # check if include and lib directories are already created
    if not os.path.exists(TargetPath+"lib"):
        os.makedirs(TargetPath+"lib")
    else:
        os.system("rm %s/*.so" % (TargetPath+"lib"))

    if not os.path.exists(TargetPath+"include"):
        os.makedirs(TargetPath+"include")
    else:
        os.system("rm %s/*.cuh" % (TargetPath+"include"))

    # install
    if not os.path.exists("libggems.so"):
        printError("libggems.so not found, please compile the code before installing.")

    print("* Installing includes")
    os.system("cp *.cuh %s" % TargetPath+"include")
    #os.system("cp -r %s/include/* %s" % (cudaHome, TargetPath+"include"))

    print("* Installing libggems")
    os.system("cp *.so %s" % TargetPath+"lib")

    #print("* Installing libcuda")
    #os.system("cp -L %s/lib64/libcudart.so %s" % (cudaHome, TargetPath+"lib"))
