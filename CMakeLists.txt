cmake_minimum_required(VERSION 3.1)

project(terraingen)

add_definitions(/MP) # Enable multi-processor compilation.
add_definitions(/W4)
add_definitions(-DUNICODE -D_UNICODE)
add_definitions(-D__SSE3__ -D__SSSE3__ -D__SSE4_1__)

add_definitions(-DPNG_ALLOW_BENIGN_ERRORS=1)
add_definitions(-DPNG_INTEL_SSE=1)
add_definitions(-DPNG_NO_SETJMP=1)

add_definitions(-DBUILD_TESTS=1)
add_definitions(-DOPENCL_OPENGL_INTEROP=1)


if(WIN32)
	if(MSVC_VERSION GREATER_EQUAL 1920 AND MSVC_VERSION LESS 1930)
		set(TERRAINGEN_VS_VER 2019)
	elseif(MSVC_VERSION GREATER_EQUAL 1930 AND MSVC_VERSION LESS 1940)
		set(TERRAINGEN_VS_VER 2022)
	else()
		message(FATAL_ERROR "Unhandled MSVC version")
	endif()
endif()


SET(GLARE_CORE_LIBS $ENV{GLARE_CORE_LIBS})

#---------------------------- GLARE_CORE_TRUNK ----------------------------
set(GLARE_CORE_TRUNK "" CACHE FILEPATH "Glare-core trunk source directory")
MESSAGE("GLARE_CORE_TRUNK: '${GLARE_CORE_TRUNK}'")

if("${GLARE_CORE_TRUNK}" STREQUAL "")
	MESSAGE(FATAL_ERROR "Please specify the Glare-core trunk source directory with -DGLARE_CORE_TRUNK=XXX")
endif()

set(GLARE_CORE_TRUNK_DIR_ENV "${GLARE_CORE_TRUNK}")  # Needs to be set for opengl.cmake

#---------------------------- SDL_BUILD_DIR ----------------------------
set(SDL_BUILD_DIR "" CACHE FILEPATH "SDL build directory")
MESSAGE("SDL_BUILD_DIR: '${SDL_BUILD_DIR}'")

if("${SDL_BUILD_DIR}" STREQUAL "")
	MESSAGE(FATAL_ERROR "Please specify the SDL2 build directory with -DSDL_BUILD_DIR=XXX")
endif()
if(NOT EXISTS "${SDL_BUILD_DIR}/include/SDL2/SDL.h")
	MESSAGE(FATAL_ERROR "Could not find SDL.h from your SDL2 build at '${SDL_BUILD_DIR}/include/SDL2/SDL.h'. Please specify the SDL2 build directory with -DSDL_BUILD_DIR=XXX")
endif()
#---------------------------- End SDL_BUILD_DIR ----------------------------

add_definitions(-DBASE_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

set(jpegdir		"${GLARE_CORE_LIBS}/libjpeg-turbo/libjpeg-turbo-3.0.0-vs${TERRAINGEN_VS_VER}-install")

# These libraries are checked into the glare-core repo to make setting up builds easier.
set(zlibdir		"${GLARE_CORE_TRUNK}/zlib")
#set(pugixmldir	"${GLARE_CORE_TRUNK}/pugixml/src")
set(pngdir		"${GLARE_CORE_TRUNK}/libpng")
set(zstddir		"${GLARE_CORE_TRUNK}/zstd")

#set(imguidir 	"${GLARE_CORE_LIBS}/imgui")
set(imguidir 	"${GLARE_CORE_TRUNK}/ImGui")
set(libjpegturbodir "${GLARE_CORE_LIBS}/libjpeg-turbo-builds")

if(WIN32)
	set(libjpegturbodir "${libjpegturbodir}/vs_${TERRAINGEN_VS_VER}_64") # Append dir suffix for VS version and bitness.
endif()

include_directories("./")
include_directories(${SDL_BUILD_DIR}/include/SDL2)
if(WIN32)
	include_directories(${SDL_BUILD_DIR}/include-config-release/SDL2) # For SDL_config.h.  NOTE: SDL_config.h seems to be the same for both Debug and Release configs, at least on Windows.
else()
	include_directories(${SDL_BUILD_DIR}/include-config-/SDL2)
endif()
include_directories(${GLARE_CORE_TRUNK})
include_directories(${GLARE_CORE_TRUNK}/utils)
include_directories(${GLARE_CORE_TRUNK}/opengl)
include_directories(${zstddir}/lib)
include_directories(${zstddir}/lib/common) # for xxhash
include_directories(${GLARE_CORE_TRUNK}/little_cms/include)
include_directories(${GLARE_CORE_TRUNK}/giflib/lib)
include_directories(${GLARE_CORE_TRUNK}/opencl/khronos)
#include_directories(${GLARE_CORE_TRUNK}/opencl/)

include_directories(${jpegdir}/include)  # libjpeg-turbo-master (jpegdir) has most of the source,
include_directories(${libjpegturbodir})  # libjpegturbodir has jconfig.h
include_directories(${pngdir})
include_directories(${zlibdir})
include_directories(${imguidir})



#============== Tracy profiler ==============
include_directories("${GLARE_CORE_TRUNK}/tracy/public")
set(tracy_files "${GLARE_CORE_TRUNK}/tracy/public/TracyClient.cpp")


#============== OpenEXR ==============
include(${GLARE_CORE_TRUNK}/OpenEXR/openexr.cmake)

#============== OpenCL ==============
set (opencl
${GLARE_CORE_TRUNK}/opencl/OpenCL.cpp
${GLARE_CORE_TRUNK}/opencl/OpenCL.h
${GLARE_CORE_TRUNK}/opencl/OpenCLBuffer.cpp
${GLARE_CORE_TRUNK}/opencl/OpenCLBuffer.h
${GLARE_CORE_TRUNK}/opencl/OpenCLCommandQueue.cpp
${GLARE_CORE_TRUNK}/opencl/OpenCLCommandQueue.h
${GLARE_CORE_TRUNK}/opencl/OpenCLContext.cpp
${GLARE_CORE_TRUNK}/opencl/OpenCLContext.h
${GLARE_CORE_TRUNK}/opencl/OpenCLDevice.cpp
${GLARE_CORE_TRUNK}/opencl/OpenCLDevice.h
${GLARE_CORE_TRUNK}/opencl/OpenCLImage.cpp
${GLARE_CORE_TRUNK}/opencl/OpenCLImage.h
${GLARE_CORE_TRUNK}/opencl/OpenCLKernel.cpp
${GLARE_CORE_TRUNK}/opencl/OpenCLKernel.h
${GLARE_CORE_TRUNK}/opencl/OpenCLPlatform.cpp
${GLARE_CORE_TRUNK}/opencl/OpenCLPlatform.h
${GLARE_CORE_TRUNK}/opencl/OpenCLProgram.cpp
${GLARE_CORE_TRUNK}/opencl/OpenCLProgram.h
${GLARE_CORE_TRUNK}/opencl/OpenCLProgramCache.cpp
${GLARE_CORE_TRUNK}/opencl/OpenCLProgramCache.h
#${GLARE_CORE_TRUNK}/opencl/OpenCLTests.cpp
#${GLARE_CORE_TRUNK}/opencl/OpenCLTests.h
)

#============== ImGui ==============
set (imgui
${imguidir}/imgui.cpp
${imguidir}/imgui.h
${imguidir}/imgui_demo.cpp
${imguidir}/imgui_draw.cpp
${imguidir}/imgui_internal.h
${imguidir}/imgui_widgets.cpp
${imguidir}/imgui_tables.cpp
${imguidir}/imstb_rectpack.h
${imguidir}/imstb_textedit.h
${imguidir}/imstb_truetype.h
${imguidir}/backends/imgui_impl_sdl2.cpp
${imguidir}/backends/imgui_impl_sdl2.h
${imguidir}/backends/imgui_impl_opengl3.cpp
${imguidir}/backends/imgui_impl_opengl3.h
)


#============== libpng ==============
set(libpng
${pngdir}/png.c ${pngdir}/pngerror.c ${pngdir}/pngget.c ${pngdir}/pngmem.c ${pngdir}/pngpread.c ${pngdir}/pngread.c ${pngdir}/pngrio.c ${pngdir}/pngrtran.c
${pngdir}/pngrutil.c ${pngdir}/pngset.c ${pngdir}/pngtrans.c ${pngdir}/pngwio.c ${pngdir}/pngwrite.c ${pngdir}/pngwtran.c ${pngdir}/pngwutil.c
${pngdir}/png.h
${pngdir}/pngconf.h
${pngdir}/pngdebug.h
${pngdir}/pnginfo.h
${pngdir}/pnglibconf.h
${pngdir}/pngpriv.h
${pngdir}/pngstruct.h

${pngdir}/intel/intel_init.c
${pngdir}/intel/filter_sse2_intrinsics.c
)


#============== zlib ==============
set(zlib
${zlibdir}/adler32.c
${zlibdir}/compress.c
${zlibdir}/crc32.c
${zlibdir}/deflate.c
${zlibdir}/inffast.c
${zlibdir}/inflate.c
${zlibdir}/inftrees.c
${zlibdir}/trees.c
${zlibdir}/uncompr.c
${zlibdir}/zutil.c
${zlibdir}/crc32.h
${zlibdir}/deflate.h
${zlibdir}/inffast.h
${zlibdir}/inffixed.h
${zlibdir}/inflate.h
${zlibdir}/inftrees.h
${zlibdir}/trees.h
${zlibdir}/zconf.h
${zlibdir}/zlib.h
${zlibdir}/zutil.h
)


#============== pugixml ==============
#set(pugixml
#${pugixmldir}/pugixml.cpp
#${pugixmldir}/pugixml.hpp
#${pugixmldir}/pugiconfig.hpp
#)

# MESSAGE(${pugixml})

FILE(GLOB giflib "${GLARE_CORE_TRUNK}/giflib/lib/*.c" "${GLARE_CORE_TRUNK}/giflib/lib/*.cpp" "${GLARE_CORE_TRUNK}/giflib/lib/*.h")
FILE(GLOB little_cms "${GLARE_CORE_TRUNK}/little_cms/src/*.c"  "${GLARE_CORE_TRUNK}/little_cms/include/*.h")
FILE(GLOB zstandard 
"${zstddir}/lib/*.h"
"${zstddir}/lib/common/*.c"
"${zstddir}/lib/common/*.h"
"${zstddir}/lib/compress/*.c"
"${zstddir}/lib/compress/*.h"
"${zstddir}/lib/decompress/*.c"
"${zstddir}/lib/decompress/*.h"
)

SOURCE_GROUP(libpng FILES ${libpng})
SOURCE_GROUP(zlib FILES ${zlib})
#SOURCE_GROUP(pugixml FILES ${pugixml})
SOURCE_GROUP(giflib FILES ${giflib})
SOURCE_GROUP(little_cms FILES ${little_cms})
SOURCE_GROUP(zstandard FILES ${zstandard})


#============== glare-core/indigo ==============
set(indigo
${GLARE_CORE_TRUNK}/indigo/TextureServer.cpp
${GLARE_CORE_TRUNK}/indigo/TextureServer.h

${GLARE_CORE_TRUNK}/dll/IndigoMesh.cpp
${GLARE_CORE_TRUNK}/dll/include/IndigoMesh.h
${GLARE_CORE_TRUNK}/dll/IndigoAllocation.cpp
${GLARE_CORE_TRUNK}/dll/include/IndigoAllocation.h
)

#============== graphics ==============
set(graphics
${GLARE_CORE_TRUNK}/graphics/AnimationData.cpp
${GLARE_CORE_TRUNK}/graphics/AnimationData.h
${GLARE_CORE_TRUNK}/graphics/BatchedMesh.cpp
${GLARE_CORE_TRUNK}/graphics/BatchedMesh.h
${GLARE_CORE_TRUNK}/graphics/bitmap.cpp
${GLARE_CORE_TRUNK}/graphics/bitmap.h
${GLARE_CORE_TRUNK}/graphics/Image.cpp
${GLARE_CORE_TRUNK}/graphics/Image.h
${GLARE_CORE_TRUNK}/graphics/Map2D.cpp
${GLARE_CORE_TRUNK}/graphics/Map2D.h
${GLARE_CORE_TRUNK}/graphics/ImageMap.cpp
${GLARE_CORE_TRUNK}/graphics/ImageMap.h
${GLARE_CORE_TRUNK}/graphics/imformatdecoder.cpp
${GLARE_CORE_TRUNK}/graphics/imformatdecoder.h
${GLARE_CORE_TRUNK}/graphics/jpegdecoder.cpp
${GLARE_CORE_TRUNK}/graphics/jpegdecoder.h
${GLARE_CORE_TRUNK}/graphics/PNGDecoder.cpp
${GLARE_CORE_TRUNK}/graphics/PNGDecoder.h
${GLARE_CORE_TRUNK}/graphics/tgadecoder.cpp
${GLARE_CORE_TRUNK}/graphics/tgadecoder.h
${GLARE_CORE_TRUNK}/graphics/bmpdecoder.cpp
${GLARE_CORE_TRUNK}/graphics/bmpdecoder.h
${GLARE_CORE_TRUNK}/graphics/EXRDecoder.cpp
${GLARE_CORE_TRUNK}/graphics/EXRDecoder.h
${GLARE_CORE_TRUNK}/graphics/FloatDecoder.cpp
${GLARE_CORE_TRUNK}/graphics/FloatDecoder.h
${GLARE_CORE_TRUNK}/graphics/GifDecoder.cpp
${GLARE_CORE_TRUNK}/graphics/GifDecoder.h
${GLARE_CORE_TRUNK}/graphics/DXTCompression.cpp
${GLARE_CORE_TRUNK}/graphics/DXTCompression.h
${GLARE_CORE_TRUNK}/graphics/KTXDecoder.cpp
${GLARE_CORE_TRUNK}/graphics/KTXDecoder.h
${GLARE_CORE_TRUNK}/graphics/CompressedImage.cpp
${GLARE_CORE_TRUNK}/graphics/CompressedImage.h
${GLARE_CORE_TRUNK}/graphics/PerlinNoise.cpp
${GLARE_CORE_TRUNK}/graphics/PerlinNoise.h
${GLARE_CORE_TRUNK}/graphics/Voronoi.cpp
${GLARE_CORE_TRUNK}/graphics/Voronoi.h
${GLARE_CORE_TRUNK}/graphics/GridNoise.cpp
${GLARE_CORE_TRUNK}/graphics/GridNoise.h
${GLARE_CORE_TRUNK}/graphics/TextureProcessing.cpp
${GLARE_CORE_TRUNK}/graphics/TextureProcessing.h
${GLARE_CORE_TRUNK}/graphics/SRGBUtils.cpp
${GLARE_CORE_TRUNK}/graphics/SRGBUtils.h
${GLARE_CORE_TRUNK}/graphics/Colour4f.cpp
${GLARE_CORE_TRUNK}/graphics/Colour4f.h
${GLARE_CORE_TRUNK}/graphics/Image4f.cpp
${GLARE_CORE_TRUNK}/graphics/Image4f.h
${GLARE_CORE_TRUNK}/graphics/BoxFilterFunction.cpp
${GLARE_CORE_TRUNK}/graphics/BoxFilterFunction.h
${GLARE_CORE_TRUNK}/graphics/FilterFunction.cpp
${GLARE_CORE_TRUNK}/graphics/FilterFunction.h
)

include(${GLARE_CORE_TRUNK}/opengl/opengl.cmake)

set(utils
${GLARE_CORE_TRUNK}/utils/BestFitAllocator.cpp
${GLARE_CORE_TRUNK}/utils/BestFitAllocator.h
${GLARE_CORE_TRUNK}/utils/BufferViewInStream.cpp
${GLARE_CORE_TRUNK}/utils/BufferViewInStream.h
${GLARE_CORE_TRUNK}/utils/Clock.cpp
${GLARE_CORE_TRUNK}/utils/Clock.h
${GLARE_CORE_TRUNK}/utils/CycleTimer.cpp
${GLARE_CORE_TRUNK}/utils/CycleTimer.h
${GLARE_CORE_TRUNK}/utils/ConPrint.cpp
${GLARE_CORE_TRUNK}/utils/ConPrint.h
${GLARE_CORE_TRUNK}/utils/StringUtils.cpp
${GLARE_CORE_TRUNK}/utils/StringUtils.h
${GLARE_CORE_TRUNK}/utils/Mutex.cpp
${GLARE_CORE_TRUNK}/utils/Mutex.h
${GLARE_CORE_TRUNK}/utils/Lock.cpp
${GLARE_CORE_TRUNK}/utils/Lock.h
${GLARE_CORE_TRUNK}/utils/DynamicLib.cpp
${GLARE_CORE_TRUNK}/utils/DynamicLib.h
${GLARE_CORE_TRUNK}/utils/StandardPrintOutput.cpp
${GLARE_CORE_TRUNK}/utils/StandardPrintOutput.h
${GLARE_CORE_TRUNK}/utils/FileUtils.cpp
${GLARE_CORE_TRUNK}/utils/FileUtils.h
${GLARE_CORE_TRUNK}/utils/MemMappedFile.cpp
${GLARE_CORE_TRUNK}/utils/MemMappedFile.h
${GLARE_CORE_TRUNK}/utils/PlatformUtils.cpp
${GLARE_CORE_TRUNK}/utils/PlatformUtils.h
${GLARE_CORE_TRUNK}/utils/TaskManager.cpp
${GLARE_CORE_TRUNK}/utils/TaskManager.h
${GLARE_CORE_TRUNK}/utils/Task.cpp
${GLARE_CORE_TRUNK}/utils/Task.h
${GLARE_CORE_TRUNK}/utils/Condition.cpp
${GLARE_CORE_TRUNK}/utils/Condition.h
${GLARE_CORE_TRUNK}/utils/MyThread.cpp
${GLARE_CORE_TRUNK}/utils/MyThread.h
${GLARE_CORE_TRUNK}/utils/TaskRunnerThread.cpp
${GLARE_CORE_TRUNK}/utils/TaskRunnerThread.h
${GLARE_CORE_TRUNK}/utils/ThreadManager.cpp
${GLARE_CORE_TRUNK}/utils/ThreadManager.h
${GLARE_CORE_TRUNK}/utils/Parser.cpp
${GLARE_CORE_TRUNK}/utils/Parser.h
${GLARE_CORE_TRUNK}/utils/FileHandle.cpp
${GLARE_CORE_TRUNK}/utils/FileHandle.h
${GLARE_CORE_TRUNK}/utils/Timer.cpp
${GLARE_CORE_TRUNK}/utils/Timer.h
${GLARE_CORE_TRUNK}/utils/MemAlloc.cpp
${GLARE_CORE_TRUNK}/utils/MemAlloc.h
${GLARE_CORE_TRUNK}/utils/FileInStream.cpp
${GLARE_CORE_TRUNK}/utils/FileInStream.h
${GLARE_CORE_TRUNK}/utils/FileOutStream.cpp
${GLARE_CORE_TRUNK}/utils/FileOutStream.h
${GLARE_CORE_TRUNK}/utils/OutStream.cpp
${GLARE_CORE_TRUNK}/utils/OutStream.h
${GLARE_CORE_TRUNK}/utils/InStream.cpp
${GLARE_CORE_TRUNK}/utils/InStream.h
${GLARE_CORE_TRUNK}/utils/BufferInStream.cpp
${GLARE_CORE_TRUNK}/utils/BufferInStream.h
${GLARE_CORE_TRUNK}/utils/BufferOutStream.cpp
${GLARE_CORE_TRUNK}/utils/BufferOutStream.h
${GLARE_CORE_TRUNK}/utils/TestUtils.cpp
${GLARE_CORE_TRUNK}/utils/TestUtils.h
${GLARE_CORE_TRUNK}/utils/RuntimeCheck.cpp
${GLARE_CORE_TRUNK}/utils/RuntimeCheck.h
${GLARE_CORE_TRUNK}/utils/MessageableThread.cpp
${GLARE_CORE_TRUNK}/utils/MessageableThread.h
${GLARE_CORE_TRUNK}/utils/GeneralMemAllocator.cpp
${GLARE_CORE_TRUNK}/utils/GeneralMemAllocator.h
${GLARE_CORE_TRUNK}/utils/PoolAllocator.cpp
${GLARE_CORE_TRUNK}/utils/PoolAllocator.h
${GLARE_CORE_TRUNK}/utils/KillThreadMessage.cpp
${GLARE_CORE_TRUNK}/utils/KillThreadMessage.h
${GLARE_CORE_TRUNK}/utils/ThreadMessage.cpp
${GLARE_CORE_TRUNK}/utils/ThreadMessage.h

)

set(maths
${GLARE_CORE_TRUNK}/maths/SSE.cpp
${GLARE_CORE_TRUNK}/maths/SSE.h
${GLARE_CORE_TRUNK}/maths/Vec4f.cpp
${GLARE_CORE_TRUNK}/maths/Vec4f.h
${GLARE_CORE_TRUNK}/maths/Matrix4f.cpp
${GLARE_CORE_TRUNK}/maths/Matrix4f.h
)


FILE(GLOB double_conversion "${GLARE_CORE_TRUNK}/double-conversion/*.cc" "${GLARE_CORE_TRUNK}/double-conversion/*.h")


SOURCE_GROUP(indigo FILES ${indigo})
SOURCE_GROUP(graphics FILES ${graphics})
SOURCE_GROUP(opengl FILES ${opengl})
SOURCE_GROUP(utils FILES ${utils})
SOURCE_GROUP(maths FILES ${maths})
SOURCE_GROUP(double-conversion FILES ${double_conversion})
SOURCE_GROUP(imgui FILES ${imgui})
SOURCE_GROUP(opencl FILES ${opencl})


add_executable(terraingen
terraingen.cpp
erosion_kernel.cl
#notes.txt
${indigo}
${graphics}
${opengl}
${utils}
${maths}
${double_conversion}
${openexr_all_files}
${libjpg}
${libpng}
${zlib}
#${pugixml}
${giflib}
${little_cms}
${zstandard}
${imgui}
${opencl}
)


#if(WIN32)
#	SET(TURBOJPEG_LIB ${libjpegturbodir}/$(Configuration)/turbojpeg-static.lib)
#else()
#	SET(TURBOJPEG_LIB ${libjpegturbodir}/libjpeg.a)
#endif()


target_link_libraries(terraingen
debug "${SDL_BUILD_DIR}/Debug/SDL2d.lib" 
debug "${SDL_BUILD_DIR}/Debug/SDL2maind.lib" 
optimized "${SDL_BUILD_DIR}/RelWithDebInfo/SDL2.lib" 
optimized "${SDL_BUILD_DIR}/RelWithDebInfo/SDL2main.lib" 
Opengl32.lib
debug     "${jpegdir}/lib/turbojpeg-static.lib" 
optimized "${jpegdir}/lib/turbojpeg-static.lib" 
)
