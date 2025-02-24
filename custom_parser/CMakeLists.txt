cmake_minimum_required(VERSION 3.10)
project(custom_parser)

# Include DeepStream headers
include_directories(/opt/nvidia/deepstream/deepstream-6.2/sources/includes)

# Include CUDA headers
include_directories(/usr/local/cuda/include)

# Add the source file
add_library(custom_parser SHARED custom_parser.cpp)

# Link against DeepStream libraries
link_directories(/opt/nvidia/deepstream/deepstream-6.2/lib/)
target_link_libraries(custom_parser /opt/nvidia/deepstream/deepstream-6.2/lib/libnvds_infer.so)