cmake_minimum_required (VERSION 2.8.0)

project (IntegrationPointsExperiment)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Location to search files or libraries
set(SEARCH_DIRS ENV PATH /usr/local/opt/ /usr/include /usr/local/include /usr/lib /usr/lib64 /usr/local/lib /usr/local/pzlib/include /usr/local/pzlib/lib .. )

#enabling NEOPZ library
find_path(NEOPZ_INCLUDE pz_config.h PATHS ${SEARCH_DIRS})
if(NEOPZ_INCLUDE-NOTFOUND)
	set (NEOPZ_INCLUDE "" CACHE PATH "Directory where NEOPZ header files can be found")
else()
include_directories(${NEOPZ_INCLUDE})
include_directories(${NEOPZ_INCLUDE}/Python)
include_directories(${NEOPZ_INCLUDE}/Util)
include_directories(${NEOPZ_INCLUDE}/PerfUtil)
include_directories(${NEOPZ_INCLUDE}/Common)
include_directories(${NEOPZ_INCLUDE}/Save)
include_directories(${NEOPZ_INCLUDE}/Matrix)
include_directories(${NEOPZ_INCLUDE}/Topology)
include_directories(${NEOPZ_INCLUDE}/Geom)
include_directories(${NEOPZ_INCLUDE}/SpecialMaps)
include_directories(${NEOPZ_INCLUDE}/Refine)
include_directories(${NEOPZ_INCLUDE}/Shape)
include_directories(${NEOPZ_INCLUDE}/Material)
include_directories(${NEOPZ_INCLUDE}/Material/REAL)
include_directories(${NEOPZ_INCLUDE}/Material/REAL/Plasticity)
include_directories(${NEOPZ_INCLUDE}/Material/Complex)
include_directories(${NEOPZ_INCLUDE}/Multigrid)
include_directories(${NEOPZ_INCLUDE}/Mesh)
include_directories(${NEOPZ_INCLUDE}/Mesh/REAL)
include_directories(${NEOPZ_INCLUDE}/StrMatrix)
include_directories(${NEOPZ_INCLUDE}/Integral)
include_directories(${NEOPZ_INCLUDE}/Frontal)
include_directories(${NEOPZ_INCLUDE}/Optimization)
include_directories(${NEOPZ_INCLUDE}/Pre)
include_directories(${NEOPZ_INCLUDE}/Post)
include_directories(${NEOPZ_INCLUDE}/Random)
include_directories(${NEOPZ_INCLUDE}/Analysis)
include_directories(${NEOPZ_INCLUDE}/SubStruct)
include_directories(${NEOPZ_INCLUDE}/LinearSolvers)
include_directories(${NEOPZ_INCLUDE}/External)
include_directories(${NEOPZ_INCLUDE}/External/sloan)
include_directories(${NEOPZ_INCLUDE}/Publications)

endif()

find_library(NEOPZ_LIB NAMES libpz.a PATHS ${SEARCH_DIRS})
if(NEOPZ_LIB-NOTFOUND)
    set (NEOPZ_LIB "" CACHE PATH "Directory where the NEOPZ library can be found")
endif()

#enabling pthread library
find_path(PTHREAD_INCLUDE NAMES PThread.h pthread.h PATHS ${SEARCH_DIRS} ${PROJECT_SOURCE_DIR}/../externallibs/pthread ${PROJECT_SOURCE_DIR}/../externallibs/pthread/include ${PROJECT_SOURCE_DIR}/../pthread )
find_library(PTHREAD_LIB NAMES libpthread.dylib pthread.lib pthreadVC2.lib libPThread.so libPThread.a libpthread.so libpthread.a PATHS ${SEARCH_DIRS} ${PROJECT_SOURCE_DIR}/../externallibs/pthread/lib  ${PROJECT_SOURCE_DIR}/../pthread/lib)
if(PTHREAD_INCLUDE-NOTFOUND)
	set (PTHREAD_INCLUDE "" CACHE PATH "Directory where pthread.h can be found")
endif()
if(PTHREAD_LIB-NOTFOUND)
	set (PTHREAD_LIB "" CACHE PATH "Directory where the pthread library can be found")
endif()
include_directories(${PTHREAD_INCLUDE})


#enabling MKL library
option(USING_MKL "Whether the MKL library will be linked in" ON)
if(USING_MKL)
    ####add_definitions(-DUSING_MKL)
    ####add_definitions(-DUSING_LAPACK)
    #set(USING_LAPACK ON)
    ####add_definitions(-DUSING_BLAS)
    #set(USING_BLAS ON)
if (WIN32)
    set(MKL_ROOT “opt/intel/compilers_and_libraries/mac”)
    find_path(MKL_INCLUDE NAMES mkl.h PATHS ${MKL_ROOT}/mkl/include)
    find_library(MKL_LIB_INTEL NAMES mkl_rt.lib PATHS ${MKL_ROOT}/mkl/lib/intel64_win)
    find_library(MKL_LIB_CORE NAMES mkl_core.lib PATHS ${MKL_ROOT}/mkl/lib/intel64_win)
    find_library(MKL_LIB_THREAD NAMES mkl_intel_thread.lib PATHS ${MKL_ROOT}/mkl/lib/intel64_win)
    find_library(COMPOSER_OMP NAMES libiomp5md.lib PATHS ${MKL_ROOT}/compiler/lib/intel64_win)
else()
    find_path(MKL_INCLUDE NAMES mkl.h PATHS ${SEARCH_DIRS} /opt/intel/mkl/include /softwares/intel/mkl/include)
    find_library(MKL_LIB_INTEL NAMES libmkl_intel_lp64.so libmkl_intel_lp64.dylib PATHS
    ${SEARCH_DIRS} /opt/intel/mkl/lib /opt/intel/mkl/lib/intel64/ /softwares/intel/mkl/lib /softwares/intel/mkl/lib/intel64)
    find_library(MKL_LIB_CORE NAMES libmkl_core.so libmkl_core.dylib PATHS ${SEARCH_DIRS}
    /opt/intel/mkl/lib /opt/intel/mkl/lib/intel64/ /softwares/intel/mkl/lib /softwares/intel/mkl/lib/intel64/)
    find_library(MKL_LIB_THREAD NAMES libmkl_intel_thread.so libmkl_intel_thread.dylib PATHS
    ${SEARCH_DIRS} /opt/intel/mkl/lib /opt/intel/mkl/lib/intel64/ /softwares/intel/mkl/lib /softwares/intel/mkl/lib/intel64/)
    find_library(COMPOSER_OMP NAMES libiomp5.so libiomp5.dylib PATHS ${SEARCH_DIRS}
    /opt/intel/composer_xe/compiler/lib
    /opt/intel/composerxe/lib/intel64
    /opt/intel/lib /softwares/intel/lib/intel64
    /opt/intel/compilers_and_libraries/linux/lib/intel64_lin )

#find_path(OMP_INCLUDE NAMES omp.h PATHS ${SEARCH_DIRS} /opt/intel/composer_xe_2013_sp1.1.103/compiler/include)
#include_directories(${OMP_INCLUDE})
endif()

if(MKL_INCLUDE-NOTFOUND)
    set (MKL_INCLUDE "" CACHE PATH "Directory where mkl.h can be found")
else()
    #include_directories(${MKL_INCLUDE})
endif()
    if(MKL_LIB-NOTFOUND)
    set (MKL_LIB_INTEL "" CACHE PATH "Directory where the mkl library can be found")
else()
    link_libraries(${MKL_LIB_INTEL})
    link_libraries(${MKL_LIB_CORE})
    link_libraries(${MKL_LIB_THREAD})
    link_libraries(${COMPOSER_OMP})
endif()
endif(USING_MKL)

file(GLOB headers *.h)
file(GLOB sources *.cpp)

set(main_SRCS ${sources} ${headers})

add_executable(IntegrationPointsExperiment ${main_SRCS})

target_link_libraries(IntegrationPointsExperiment ${NEOPZ_LIB} ${PTHREAD_LIB})
