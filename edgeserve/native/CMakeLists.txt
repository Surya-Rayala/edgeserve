# edgeserve-worker: the process that owns the GPU and runs one TensorRT Edge-LLM
# engine. It is built as a subdirectory of the pinned Edge-LLM tree
# (build.sh passes -DEDGELLM_EXTERNAL_SUBDIR=<this dir>), so it
# links exactly the runtime and plugin build that the engines are made with.

add_executable(edgeserve-worker worker.cpp)
target_link_libraries(edgeserve-worker PRIVATE edgellmCore commonLibraryExt)
target_include_directories(edgeserve-worker PRIVATE ${COMMON_INCLUDE_DIRS})
add_cross_build_link_options(edgeserve-worker)
# Next to libNvInfer_edgellm_plugin.so and the examples' parent dir.
set_target_properties(edgeserve-worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  ${CMAKE_BINARY_DIR})

# Diagnostic: the prefill (FMHA) and decode (XQA) attention kernels against a CPU reference (not used by the
# server; edgeserve setup runs it; see docs/PLATFORMS.md, 'Checking a platform').
add_executable(attention-check attention_check.cpp)
target_link_libraries(attention-check PRIVATE edgellmCore edgellmPluginJit commonLibraryExt)
target_include_directories(attention-check PRIVATE ${COMMON_INCLUDE_DIRS})
add_cross_build_link_options(attention-check)
set_target_properties(attention-check PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
