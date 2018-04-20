cmake_minimum_required(VERSION 2.8)
#set(CMAKE_C_COMPILIER /usr/local/bin/gcc)
#set(CMAKE_CXX_COMPILER /usr/local/bin/g++)

project(MinervaSegs)

include_directories(
  basic
  /opt/mshadow/
  ../LibN3L
  
)

add_definitions(-DUSE_CUDA=0)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O0" )
ELSE()
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O3" )
ENDIF()
#set (CMAKE_CXX_FLAGS "-std=c++11 -g -O3 -DNDEBUG -DKENLM_MAX_ORDER=6")

####for   openblas
add_definitions(-DMSHADOW_USE_CUDA=0)
add_definitions(-DMSHADOW_USE_CBLAS=1)
add_definitions(-DMSHADOW_USE_MKL=0)

SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lopenblas")
####endfor openblas

####for   cuda
#add_definitions(-DMSHADOW_USE_CUDA=1)
#add_definitions(-DMSHADOW_USE_CBLAS=1)
#add_definitions(-DMSHADOW_USE_MKL=0)

#SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lcudart -lcublas -lcurand" )
#include_directories(
#  $(USE_CUDA_PATH)/include
#)
#LINK_DIRECTORIES($(USE_CUDA_PATH)/lib64)
####endfor cuda



#add_subdirectory(basic)

#aux_source_directory(. DIR_SRCS)





add_executable(LSTMCRFMLLabeler LSTMCRFMLLabeler.cpp)
add_executable(UnLSTMCRFMLLabeler UnLSTMCRFMLLabeler.cpp)




target_link_libraries(LSTMCRFMLLabeler openblas)
target_link_libraries(UnLSTMCRFMLLabeler openblas)
