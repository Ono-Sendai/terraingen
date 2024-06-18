
# Building TerrainGen


## Requirements

Ruby (for build_jpegturbo.rb script)

## Instructions

Clone TerrainGen source somewhere, e.g.

	git clone https://github.com/Ono-Sendai/terraingen.git c:/code/terraingen

Build SDL 2 (https://www.libsdl.org/) somewhere on disk.

Clone the Glare Core source somewhere, e.g.

	git clone https://github.com/glaretechnologies/glare-core.git c:/code/glare-core

Set GLARE_CORE_LIBS environment variable to something like c:/code.  This is where libjpegturbo will be built.

Build libjpegturbo with scripts/build_jpegturbo.rb from Glare core scripts dir:

	cd c:/code/glare-core/scripts
	ruby build_jpegturbo.rb


Now generate the TerrainGen project with CMake, specifying the directory where you cloned Glare Core and also where you built SDL.

	mkdir c:/code/terraingen_build
	cd c:/code/terraingen_build
	cmake c:/code/terraingen -DGLARE_CORE_TRUNK=c:/code/glare-core -DSDL_BUILD_DIR=C:/programming/SDL/sdl_2.30.0_build


## Running

You will need to copy a few files into your build output directory (e.g. c:/code/terraingen_build) to run TerrainGen:

Copy SDL2d.dll from C:\programming\SDL\sdl_2.30.0_build\Debug
Copy SDL2.dll from C:\programming\SDL\sdl_2.30.0_build\Release
Copy resources dir from c:/code/terraingen
Copy gl_data and shaders dirs from C:/code/glare-core/opengl
