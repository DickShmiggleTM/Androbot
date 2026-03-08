cmake_minimum_required(VERSION 3.22.1)
project("androbot_inference" VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for ARM64
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8-a+dotprod+fp16 -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGGML_USE_NEON -DGGML_ARM_FP16")
endif()

# ==========================================
# llama.cpp integration via pre-built AAR or
# source inclusion (stubs for build system)
# ==========================================

# Core inference sources
set(INFERENCE_SOURCES
    inference_engine.cpp
    llama_jni.cpp
    mmap_loader.cpp
    kv_cache_manager.cpp
    tokenizer.cpp
    context_window.cpp
)

# Build the shared library
add_library(
    androbot_inference
    SHARED
    ${INFERENCE_SOURCES}
)

# Android system libraries
find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

target_link_libraries(
    androbot_inference
    ${log-lib}
    ${android-lib}
    ${jnigraphics-lib}
)

# Include directories
target_include_directories(androbot_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Export all symbols for JNI
set_target_properties(androbot_inference PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VISIBILITY_INLINES_HIDDEN OFF
)
