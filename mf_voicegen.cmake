set(MF_MIN_VERSION "1.0.0")
set(MF_NEEDS_LILA OFF)

file(GLOB_RECURSE _mf_voicegen_sources CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/community/mf_voicegen/src/*.cpp"
)

add_library(mf_voicegen OBJECT
    ${_mf_voicegen_sources}
)

target_include_directories(mf_voicegen PUBLIC
    "${CMAKE_SOURCE_DIR}/community/mf_voicegen/src"
)

# The kernels are GLSL files that ShaderFoundry compiles when they are first used.
target_compile_definitions(mf_voicegen PUBLIC
    MF_VOICEGEN_SHADER_DIR="${CMAKE_SOURCE_DIR}/community/mf_voicegen/shaders"
)

set_target_properties(mf_voicegen PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(mf_voicegen PUBLIC MayaFlux::MayaFluxLib)

if(MF_NEEDS_LILA)
    target_link_libraries(mf_voicegen PUBLIC MayaFlux::MayaFluxHost)
endif()
