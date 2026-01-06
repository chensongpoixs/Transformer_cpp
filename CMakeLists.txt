cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Transformer)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
# 查找LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 源文件列表
set(SOURCES
    main.cpp
    embeddings.cpp
    attention.cpp
    layer_norm.cpp
    feed_forward.cpp
    sublayer_connection.cpp
    encoder.cpp
    decoder.cpp
    generator.cpp
    transformer.cpp
    utils.cpp
)

# 头文件列表
set(HEADERS
    config.h
    embeddings.h
    attention.h
    layer_norm.h
    feed_forward.h
    sublayer_connection.h
    encoder.h
    decoder.h
    generator.h
    transformer.h
    utils.h
)

# 创建可执行文件（包含源文件和头文件，头文件用于IDE支持）
add_executable(transformer ${SOURCES} ${HEADERS})

# 链接LibTorch
target_link_libraries(transformer "${TORCH_LIBRARIES}")

# 设置编译属性
set_property(TARGET transformer PROPERTY CXX_STANDARD 17)

# 如果使用CUDA
if(TORCH_CUDA_FOUND)
    enable_language(CUDA)
    set_property(TARGET transformer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# 安装规则（可选）
install(TARGETS transformer DESTINATION bin)

