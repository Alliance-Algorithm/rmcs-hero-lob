cmake_minimum_required(VERSION 3.24)
project(rmcs_hero_lob)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-Wall -Wextra -Wpedantic -O3)

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rmcs_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(type_description_interfaces REQUIRED)

find_package(OpenCV 4.5 REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(Ceres REQUIRED)

find_package(rosidl_typesupport_cpp REQUIRED)
find_package(pluginlib REQUIRED)

find_package(rmcs_executor REQUIRED)
find_package(rmcs_utility REQUIRED)
find_package(rmcs_description REQUIRED)
find_package(fast_tf REQUIRED)
find_package(hikcamera REQUIRED)

set(RMCS_MSGS_SOURCE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../rmcs_msgs/include")

include_directories(
  ${PROJECT_SOURCE_DIR}/include/
  ${PROJECT_SOURCE_DIR}/src/
  ${PROJECT_SOURCE_DIR}/test/
  ${RMCS_MSGS_SOURCE_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${type_description_interfaces_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${rmcs_msgs_INCLUDE_DIRS}
  ${rmcs_executor_INCLUDE_DIRS}
  ${rmcs_utility_INCLUDE_DIRS}
  ${rmcs_description_INCLUDE_DIRS}
  ${fast_tf_INCLUDE_DIRS}
  ${hikcamera_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${OpenVINO_INCLUDE_DIRS}
  ${Ceres_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
)

file(GLOB_RECURSE PROJECT_SOURCE
  CONFIGURE_DEPENDS
  ${PROJECT_SOURCE_DIR}/src/*.cpp
  ${PROJECT_SOURCE_DIR}/src/*.cc
)

add_library(
  ${PROJECT_NAME} SHARED
  ${PROJECT_SOURCE}
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rmcs_msgs
  std_msgs
  sensor_msgs
  cv_bridge
  geometry_msgs
  visualization_msgs
  rosidl_typesupport_cpp
  pluginlib
)

target_link_libraries(
  ${PROJECT_NAME}
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${visualization_msgs_LIBRARIES}
  ${type_description_interfaces_LIBRARIES}
  
  ${OpenCV_LIBS}
  ${OpenVINO_LIBRARIES}
  ${Ceres_LIBRARIES}
  ${rmcs_executor_LIBRARIES}
  ${rmcs_utility_LIBRARIES}
  ${rmcs_description_LIBRARIES}
  ${fast_tf_LIBRARIES}
  ${hikcamera_LIBRARIES}
)

install(
  TARGETS 
    ${PROJECT_NAME}
  DESTINATION 
    lib/
)

install(
  DIRECTORY include/
  DESTINATION include
)

pluginlib_export_plugin_description_file(
  rmcs_executor plugins.xml
)

ament_package()
