cmake_minimum_required(VERSION 3.10)
PROJECT(Fuzzy_C_Means CXX)
SET( PROJ_NAME  "Fuzzy_C_Means" )
set(CMAKE_CXX_STANDARD 11)


############################     Portion needed for OpenCV - Part_1   ######################################

#set(CMAKE_BUILD_TYPE Release)
# set("OpenCV_DIR" "/modules/opencv/3.4.1/share/OpenCV/")
set(CMAKE_PREFIX_PATH "/usr/local/Cellar/opencv@3/")

set(OpenCV_INCLUDE_DIRS "/usr/local/Cellar/opencv@3/include")
set(OpenCV_LIBS "/usr/local/Cellar/opencv@3/lib")

find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()
#################################    End of Portion Needed for OpenCV - Part_1  ############################







############################     Project files and folder inclusion    ######################################
include_directories(.)
include_directories(util)
include_directories(util/hdr)

add_executable(Fuzzy_C_Means
        util/hdr/BasicAlgo.h
        util/hdr/spline.h
        util/BasicAlgo.cpp
        util/TiffImageReader.cpp
        util/TiffImageReader.hpp
        TextGraphicsSeperation.cpp
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_S1_S2/Fuzzy_C_Means_S1.cpp
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_S1_S2/hdr/Fuzzy_C_Means_S1.h
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_S1_S2/Fuzzy_C_Means_S2.cpp
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_S1_S2/hdr/Fuzzy_C_Means_S2.h
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_M1/Fuzzy_C_Means_M1.cpp
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_M1/hdr/Fuzzy_C_Means_M1.h
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_FGfsm/Fuzzy_C_Means_FGfsm.cpp
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_FGfsm/hdr/Fuzzy_C_Means_FGfsm.h
        PatternRecogAlgos/Clustering/K_Means_Clustering.cpp
        PatternRecogAlgos/Clustering/K_Means_Clustering.h
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_FGfsm/Fuzzy_C_Means_FGfsm_S1.cpp
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_FGfsm/hdr/Fuzzy_C_Means_FGfsm_S1.h
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_FGfsm/Fuzzy_C_Means_FGfsm_S2.cpp
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_FGfsm/hdr/Fuzzy_C_Means_FGfsm_S2.h
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_FLIcm/Fuzzy_C_Means_FLIcm.cpp
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_FLIcm/hdr/Fuzzy_C_Means_FLIcm.h
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_EN_FCM/Fuzzy_C_Means_ENFCM.cpp
        PatternRecogAlgos/Fuzzy_C_Means/Fuzzy_C_Means_EN_FCM/hdr/Fuzzy_C_Means_ENFCM.h)
############################     End of Project files and folder inclusion    ######################################








############################     Portion needed for OpenCV - Part_2     ############################################
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( Fuzzy_C_Means ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(Fuzzy_C_Means opencv_core opencv_features2d opencv_flann opencv_highgui opencv_imgproc opencv_ml
        opencv_objdetect opencv_photo opencv_stitching opencv_superres opencv_video opencv_highgui opencv_imgproc
        opencv_imgcodecs opencv_ximgproc opencv_calib3d opencv_imgcodecs)
#################################    End of Portion Needed for OpenCV - Part_2     #################################








############################     Portion needed for OpenMP    ######################################

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
#[[message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()

set(CMAKE_C_COMPILER "/usr/local/Cellar/llvm/6.0.0/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/local/Cellar/llvm/6.0.0/bin/clang++")
set(OPENMP_LIBRARIES "/usr/local/Cellar/llvm/6.0.0/lib")
set(OPENMP_INCLUDES "/usr/local/Cellar/llvm/6.0.0/include")

OPTION (USE_OpenMP "Use OpenMP to enamble <omp.h>" ON)

# Find OpenMP
if(APPLE AND USE_OpenMP)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
        set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
        set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    endif()
endif()

if(USE_OpenMP)
    find_package(OpenMP REQUIRED)
endif(USE_OpenMP)

if (OPENMP_FOUND)
    include_directories("${OPENMP_INCLUDES}")
    link_directories("${OPENMP_LIBRARIES}")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)]]
#################################    End of Portion Needed for OpenMP     #################################








############################     Start of Portion needed for Boost    ######################################
#[[FIND_PACKAGE(Boost ${BOOST_MIN_VERSION} REQUIRED)
SET (BOOST_ROOT "/usr/local/Cellar/boost@1.60/1.60.0/")
SET (BOOST_INCLUDEDIR "/usr/local/Cellar/boost@1.60/1.60.0/include")
SET (BOOST_LIBRARYDIR "/usr/local/Cellar/boost@1.60/1.60.0/lib")

SET (BOOST_MIN_VERSION "1.51.0")
set (Boost_NO_BOOST_CMAKE ON)
if (NOT Boost_FOUND)
    message(FATAL_ERROR "Fatal error: Boost (version >= 1.51) required.")
else()
    message(STATUS "Setting up BOOST")
    message(STATUS " Includes - ${Boost_INCLUDE_DIRS}")
    message(STATUS " Library  - ${Boost_LIBRARY_DIRS}")
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
endif (NOT Boost_FOUND)

find_package(Boost COMPONENTS regex date_time system filesystem graph program_options)

find_package(Boost REQUIRED regex date_time system filesystem graph program_options)
find_package(Boost COMPONENTS program_options REQUIRED)


if(Boost_FOUND)
    include_directories(${BOOST_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
    add_definitions(${Boost_DEFINITIONS})

    target_link_libraries(Fuzzy_C_Means ${Boost_LIBRARIES})
    target_link_libraries( Fuzzy_C_Means ${Boost_PROGRAM_OPTIONS_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${Boost_REGEX_LIBRARY} ${Boost_SYSTEM_LIBRARY})
endif()]]
############################     End of Portion needed for Boost     ########################################