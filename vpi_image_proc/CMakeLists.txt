cmake_minimum_required(VERSION 3.0.2)
project(vpi_image_proc)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  cv_bridge
  image_geometry
  image_transport
  roscpp
)
find_package(OpenCV 4.6 REQUIRED)
find_package(Boost 1.58 REQUIRED COMPONENTS system serialization filesystem thread date_time regex timer chrono)
find_package(vpi 2.2 REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES vpi_image_proc
 CATKIN_DEPENDS cmake_modules cv_bridge image_geometry image_transport roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(vpi_image_proc src/vpi_image_proc_node.cpp)
target_link_libraries(vpi_image_proc ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} vpi)