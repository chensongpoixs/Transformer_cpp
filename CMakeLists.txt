cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Transformer)



if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "current platform: windows")
    set(DEPS_DIR "D:/dep/c_dep" CACHE FILEPATH "deps dir")
elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
	message(STATUS "current platform: unix") 
	set(DEPS_DIR "/home/linux-desktop/deps")
endif()

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

# 查找LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# SentencePiece配置
option(USE_SENTENCEPIECE "Use SentencePiece library" ON)

 
 add_definitions(-DNOMINMAX 
                -DUSE_CUDA
                )


 

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}

)


if(USE_SENTENCEPIECE)
        add_definitions(-DUSE_SENTENCEPIECE
                         
        )
        include_directories(${DEPS_DIR}/sentencepiece/include)  
        message(STATUS "sentencepiece  suppt") 
endif()


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
    train_utils.cpp
    data_loader.cpp
    train.cpp
    tokenizer_wrapper.cpp
    cuda_stream_manager.cpp
    beam_search.cpp
    bleu.cpp
    logger.cpp
    gpu_profiler.cpp
)

# 头文件列表
set(HEADERS
    config.h
    embeddings.h
    attention.h
    layer_norm.h
    feed_forward.h
    sublayer_connection.h
    cuda_stream_manager.h
    encoder.h
    decoder.h
    generator.h
    transformer.h
    utils.h
    train_utils.h
    data_loader.h
    train.h
    tokenizer_wrapper.h
    beam_search.h
    bleu.h
    logger.h
    gpu_profiler.h
)

# 创建可执行文件
add_executable(transformer ${SOURCES} ${HEADERS})

# VS2022相关设置
if(MSVC)
    # 在解决方案中为transformer设置文件夹
    set_property(TARGET transformer PROPERTY FOLDER "Transformer")
    source_group("Source Files" FILES ${SOURCES})
    source_group("Header Files" FILES ${HEADERS})

    # 在MSBuild中忽略 MSB3025 警告（不要将其视为错误）
    # MSB3025: 源文件和目标文件相同，一般是无害警告
    # 通过 VS_GLOBAL_WarningsNotAsErrors 告诉MSBuild不要把该警告当成错误
    set_property(TARGET transformer PROPERTY VS_GLOBAL_WarningsNotAsErrors "MSB3025")
endif()

# 链接LibTorch
target_link_libraries(transformer "${TORCH_LIBRARIES}")

# 链接SentencePiece
if(USE_SENTENCEPIECE )
    #add_dependencies(transformer sentencepiece_static)
    target_link_libraries(transformer 
                            ${DEPS_DIR}/sentencepiece/lib/x64/Debug/sentencepiece.lib
                            
                            )
    message(STATUS "已链接SentencePiece: ${SENTENCEPIECE_LIBRARIES}")
endif()

# 设置编译属性
set_property(TARGET transformer PROPERTY CXX_STANDARD 17)

# CUDA支持
if(TORCH_CUDA_FOUND)
    enable_language(CUDA)
    set_property(TARGET transformer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# 安装规则（可选）
install(TARGETS transformer DESTINATION bin)
