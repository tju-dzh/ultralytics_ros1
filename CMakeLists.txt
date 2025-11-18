cmake_minimum_required(VERSION 3.0.2)
project(ultralytics_ros)

## Compile as C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
  visualization_msgs
  message_generation
  cv_bridge
  image_geometry
  message_filters
  pcl_conversions
  pcl_ros
  tf2_eigen
  tf2_ros
)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

## Generate messages
add_message_files(
  FILES
  YoloResult.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  vision_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    vision_msgs
    visualization_msgs
    message_runtime
    cv_bridge
    image_geometry
    message_filters
    pcl_conversions
    pcl_ros
    tf2_eigen
    tf2_ros
  DEPENDS
    OpenCV
    PCL
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

## Declare a C++ executable: tracker_with_cloud_node
add_executable(tracker_with_cloud_node
  src/tracker_with_cloud_node.cpp
)

add_dependencies(tracker_with_cloud_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(tracker_with_cloud_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)

#############
## Install ##
#############

## Mark executables for installation
install(TARGETS tracker_with_cloud_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark Python scripts for installation
catkin_install_python(PROGRAMS
  scripts/tracker_node_ros1.py
  scripts/multi_camera_tracker_node_ros1.py
  scripts/convert_to_tensorrt.py
  scripts/web_server.py
  scripts/optimize_models_for_orin.sh
  scripts/ground_station_info.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation
install(DIRECTORY
  launch
  rviz
  web_interface
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
