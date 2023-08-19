/*=====================================================================
terraingen.cpp
--------------
Copyright Nicholas Chapman 2023 -
=====================================================================*/


#include <graphics/PNGDecoder.h>
#include <graphics/ImageMap.h>
#include <graphics/EXRDecoder.h>
#include <maths/GeometrySampling.h>
#include <dll/include/IndigoMesh.h>
#include <graphics/PerlinNoise.h>
#include <opengl/OpenGLShader.h>
#include <opengl/OpenGLProgram.h>
#include <opengl/OpenGLEngine.h>
#include <opengl/GLMeshBuilding.h>
#include <indigo/TextureServer.h>
#include <maths/PCG32.h>
#include <opengl/VBO.h>
#include <opengl/VAO.h>
#include <opencl/OpenCL.h>
#include <opencl/OpenCLKernel.h>
#include <opencl/OpenCLBuffer.h>
#include <utils/Exception.h>
#include <utils/StandardPrintOutput.h>
#include <utils/IncludeWindows.h>
#include <utils/PlatformUtils.h>
#include <utils/FileUtils.h>
#include <utils/ConPrint.h>
#include <utils/StringUtils.h>
#include <GL/gl3w.h>
#include <SDL_opengl.h>
#include <SDL.h>
#include <imgui.h>
#include <backends/imgui_impl_opengl3.h>
#include <backends/imgui_impl_sdl.h>
#include <iostream>
#include <fstream>
#include <string>


const int W = 512;
const int H = 512;


typedef struct
{
	float height; // terrain height (b)
	float water; // water height (d)
	float suspended; // Amount of suspended sediment. (s)

	float u, v; // velocity

} TerrainState;


typedef struct
{
	float f_L, f_R, f_T, f_B; // outflow flux

} FlowState;

typedef struct
{
	float flux[8];

} ThermalErosionState;


typedef struct 
{
	float delta_t; // time step
	float r; // rainfall rate
	float A; // cross-sectional 'pipe' area
	float g; // gravity accel magnitude. positive.
	float l; // virtual pipe length
	float l_x; // width between grid points in x direction
	float l_y;

	float K_c;// = 0.01; // 1; // sediment capacity constant
	float K_s;// = 0.01; // 0.5; // dissolving constant.
	float K_d;// = 0.01; // 1; // deposition constant
	float K_dmax;// = 0.1f; // Maximum erosion depth: water depth at which erosion stops.
	float K_e; // Evaporation constant

	float K_t; // thermal erosion constant
	float max_talus_angle;
	float tan_max_talus_angle;
} Constants;


class Simulation
{
public:
	int sim_iteration;

	Array2D<TerrainState> terrain_state;
	Array2D<FlowState> flow_state;
	Array2D<ThermalErosionState> thermal_erosion_state;


	OpenCLKernelRef flowSimulationKernel;
	OpenCLKernelRef thermalErosionFluxKernel;
	OpenCLKernelRef waterAndVelFieldUpdateKernel;
	OpenCLKernelRef erosionAndDepositionKernel;
	OpenCLKernelRef sedimentTransportationKernel;
	OpenCLKernelRef thermalErosionMovementKernel;
	OpenCLKernelRef evaporationKernel;

	OpenCLBuffer terrain_state_buffer;
	OpenCLBuffer flow_state_buffer_a;
	OpenCLBuffer flow_state_buffer_b;
	OpenCLBuffer thermal_erosion_state_buffer;
	OpenCLBuffer constants_buffer;

	Timer timer;

	Simulation()
	{
		sim_iteration = 0;

		terrain_state.resize(W, H);
		flow_state.resize(W, H);
	}


	void readBackToCPUMem(OpenCLCommandQueueRef command_queue)
	{
		// Read back terrain state buffer to CPU mem
		terrain_state_buffer.readTo(command_queue, /*dest ptr=*/&terrain_state.elem(0, 0), /*size=*/W * H * sizeof(TerrainState), /*blocking read=*/true);

		// TEMP: just for debugging (showing/reading flux)
		flow_state_buffer_a.readTo(command_queue, /*dest ptr=*/&flow_state.elem(0, 0), /*size=*/W * H * sizeof(FlowState), /*blocking read=*/true);
	}


	void doSimIteration(OpenCLCommandQueueRef command_queue)
	{
		OpenCLBuffer* cur_flow_state_buffer   = &flow_state_buffer_a;
		OpenCLBuffer* other_flow_state_buffer = &flow_state_buffer_b;

		const int num_iters = 10; // Should be even so that we end up with cur_flow_state_buffer == flow_state_buffer_a
		for(int z=0; z<num_iters; ++z)
		{
			flowSimulationKernel->setKernelArgBuffer(0, terrain_state_buffer);
			flowSimulationKernel->setKernelArgBuffer(1, *cur_flow_state_buffer); // source
			flowSimulationKernel->setKernelArgBuffer(2, *other_flow_state_buffer); // destination
			flowSimulationKernel->setKernelArgBuffer(3, constants_buffer);
			flowSimulationKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);

			mySwap(cur_flow_state_buffer, other_flow_state_buffer); // Swap pointers

			thermalErosionFluxKernel->setKernelArgBuffer(0, terrain_state_buffer);
			thermalErosionFluxKernel->setKernelArgBuffer(1, thermal_erosion_state_buffer); // source
			thermalErosionFluxKernel->setKernelArgBuffer(2, constants_buffer);
			thermalErosionFluxKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);  // Swap pointers

			waterAndVelFieldUpdateKernel->setKernelArgBuffer(0, *cur_flow_state_buffer);
			waterAndVelFieldUpdateKernel->setKernelArgBuffer(1, terrain_state_buffer);
			waterAndVelFieldUpdateKernel->setKernelArgBuffer(2, constants_buffer);
			waterAndVelFieldUpdateKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);
			
			erosionAndDepositionKernel->setKernelArgBuffer(0, terrain_state_buffer);
			erosionAndDepositionKernel->setKernelArgBuffer(1, constants_buffer);
			erosionAndDepositionKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);
		
			sedimentTransportationKernel->setKernelArgBuffer(0, terrain_state_buffer);
			sedimentTransportationKernel->setKernelArgBuffer(1, constants_buffer);
			sedimentTransportationKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);
			
			thermalErosionMovementKernel->setKernelArgBuffer(0, thermal_erosion_state_buffer);
			thermalErosionMovementKernel->setKernelArgBuffer(1, terrain_state_buffer);
			thermalErosionMovementKernel->setKernelArgBuffer(2, constants_buffer);
			thermalErosionMovementKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);

			evaporationKernel->setKernelArgBuffer(0, terrain_state_buffer);
			evaporationKernel->setKernelArgBuffer(1, constants_buffer);
			evaporationKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);
		}

		assert(cur_flow_state_buffer == &flow_state_buffer_a);

		sim_iteration += num_iters;
	}
};



// Pack normal into GL_INT_2_10_10_10_REV format.
inline static uint32 packNormal(const Vec3f& normal)
{
	int x = (int)(normal.x * 511.f);
	int y = (int)(normal.y * 511.f);
	int z = (int)(normal.z * 511.f);
	// ANDing with 1023 isolates the bottom 10 bits.
	return (x & 1023) | ((y & 1023) << 10) | ((z & 1023) << 20);
}


enum HeightFieldShow
{
	HeightFieldShow_TerrainOnly,
	HeightFieldShow_TerrainAndWater
};

const char* HeightFieldShow_strings[] = 
{
	"terrain only",
	"terrain and water"
};


enum TextureShow
{
	TextureShow_WaterDepth,
	TextureShow_WaterSpeed,
	TextureShow_SuspendedSediment
};
const char* TextureShow_strings[] = 
{
	"water depth",
	"water speed",
	"suspended sediment"
};


enum InitialTerrainShape
{
	InitialTerrainShape_Hat,
	//InitialTerrainShape_Cone,
	InitialTerrainShape_FBM,
	InitialTerrainShape_Perlin
};

const char* InitialTerrainShape_strings[] = 
{
	"hat",
	//"cone",
	"FBM",
	"Perlin noise"
};


OpenGLMeshRenderDataRef makeTerrainMesh(const Simulation& sim, OpenGLEngine* opengl_engine, HeightFieldShow cur_heightfield_show) 
{
	/*
	y
	^ 
	|
	--------------------
	|   /|   /|   /|   /|
	|  / |  / |  / |  / |
	| /  | /  | /  | /  |
	|----|----|----|----|
	|   /|   /|   /|   /|
	|  / |  / |  / |  / |
	| /  | /  | /  | /  |
	|----|----|----|----|
	|   /|   /|   /|   /|
	|  / |  / |  / |  / |
	| /  | /  | /  | /  |
	|----|----|----|----|
	|   /|   /|   /|   /|
	|  / |  / |  / |  / |
	| /  | /  | /  | /  |
	|----|----|----|----|---> x
	*/

	const float chunk_w = (float)sim.terrain_state.getWidth();

	int vert_res = (int)sim.terrain_state.getWidth();
	int quad_res = vert_res - 1; // Number of quads in x and y directions
	
	float quad_w = chunk_w / quad_res; // Width in metres of each quad

	const size_t normal_size_B = 4;
	const size_t vert_size_B = sizeof(float) * (3 + 2) + normal_size_B; // position, normal, uv
	OpenGLMeshRenderDataRef mesh_data = new OpenGLMeshRenderData();
	mesh_data->vert_data.resize(vert_size_B * vert_res * vert_res);

	mesh_data->vert_index_buffer.resize(quad_res * quad_res * 6);

	OpenGLMeshRenderData& meshdata = *mesh_data;

	meshdata.has_uvs = true;
	meshdata.has_shading_normals = true;
	meshdata.batches.resize(1);
	meshdata.batches[0].material_index = 0;
	meshdata.batches[0].num_indices = (uint32)meshdata.vert_index_buffer.size();
	meshdata.batches[0].prim_start_offset = 0;

	meshdata.num_materials_referenced = 1;

	meshdata.setIndexType(GL_UNSIGNED_INT);

	// NOTE: The order of these attributes should be the same as in OpenGLProgram constructor with the glBindAttribLocations.
	size_t in_vert_offset_B = 0;
	VertexAttrib pos_attrib;
	pos_attrib.enabled = true;
	pos_attrib.num_comps = 3;
	pos_attrib.type = GL_FLOAT;
	pos_attrib.normalised = false;
	pos_attrib.stride = vert_size_B;
	pos_attrib.offset = (uint32)in_vert_offset_B;
	meshdata.vertex_spec.attributes.push_back(pos_attrib);
	in_vert_offset_B += sizeof(float) * 3;

	VertexAttrib normal_attrib;
	normal_attrib.enabled = true;
	normal_attrib.num_comps = 4; // 3;
	normal_attrib.type = GL_INT_2_10_10_10_REV; // GL_FLOAT;
	normal_attrib.normalised = true; // false;
	normal_attrib.stride = vert_size_B;
	normal_attrib.offset = (uint32)in_vert_offset_B;
	meshdata.vertex_spec.attributes.push_back(normal_attrib);
	in_vert_offset_B += normal_size_B;

	const size_t uv_offset_B = in_vert_offset_B;
	VertexAttrib uv_attrib;
	uv_attrib.enabled = true;
	uv_attrib.num_comps = 2;
	uv_attrib.type = GL_FLOAT;
	uv_attrib.normalised = false;
	uv_attrib.stride = vert_size_B;
	uv_attrib.offset = (uint32)uv_offset_B;
	meshdata.vertex_spec.attributes.push_back(uv_attrib);
	in_vert_offset_B += sizeof(float) * 2;

	assert(in_vert_offset_B == vert_size_B);

	js::AABBox aabb_os = js::AABBox::emptyAABBox();

	uint8* const vert_data = mesh_data->vert_data.data();

	Timer timer;

	for(int y=0; y<vert_res; ++y)
	for(int x=0; x<vert_res; ++x)
	{
		const float p_x = x * quad_w;
		const float p_y = y * quad_w;
		const float dx = 1.f;
		const float dy = 1.f;

		const float z    = sim.terrain_state.elem(x, y).height                          + (cur_heightfield_show == HeightFieldShow::HeightFieldShow_TerrainAndWater ? sim.terrain_state.elem(x, y).water : 0);
		const float z_dx = sim.terrain_state.elem(myMin(vert_res - 1, x + 1), y).height + (cur_heightfield_show == HeightFieldShow::HeightFieldShow_TerrainAndWater ? sim.terrain_state.elem(myMin(vert_res - 1, x + 1), y).water : 0);
		const float z_dy = sim.terrain_state.elem(x, myMin(vert_res - 1, y + 1)).height + (cur_heightfield_show == HeightFieldShow::HeightFieldShow_TerrainAndWater ? sim.terrain_state.elem(x, myMin(vert_res - 1, y + 1)).water : 0);

		const Vec3f p_dx_minus_p(dx, 0, z_dx - z); // p(p_x + dx, dy) - p(p_x, p_y) = (p_x + dx, d_y, z_dx) - (p_x, p_y, z) = (d_x, 0, z_dx - z)
		const Vec3f p_dy_minus_p(0, dy, z_dy - z);

		const Vec3f normal = normalise(crossProduct(p_dx_minus_p, p_dy_minus_p));

		const Vec3f pos(p_x, p_y, z);
		std::memcpy(vert_data + vert_size_B * (y * vert_res + x), &pos, sizeof(float)*3);

		aabb_os.enlargeToHoldPoint(pos.toVec4fPoint());

		const uint32 packed_normal = packNormal(normal);
		std::memcpy(vert_data + vert_size_B * (y * vert_res + x) + sizeof(float) * 3, &packed_normal, sizeof(uint32));

		Vec2f uv((float)x / vert_res, (float)y / vert_res);
		std::memcpy(vert_data + vert_size_B * (y * vert_res + x) + uv_offset_B, &uv, sizeof(float)*2);
	}

	meshdata.aabb_os = aabb_os;


	uint32* const indices = (uint32*)mesh_data->vert_index_buffer.data();
	for(int y=0; y<quad_res; ++y)
	for(int x=0; x<quad_res; ++x)
	{
		// Trianglulate the quad in this way
		// |----|
		// | \  |
		// |  \ |
		// |   \|
		// |----|--> x

		// bot left tri
		int offset = (y*quad_res + x) * 6;
		indices[offset + 0] = y * vert_res + x; // bot left
		indices[offset + 1] = y * vert_res + x + 1; // bot right
		indices[offset + 2] = (y + 1) * vert_res + x; // top left

		// top right tri
		indices[offset + 3] = y * vert_res + x + 1; // bot right
		indices[offset + 4] = (y + 1) * vert_res + x + 1; // top right
		indices[offset + 5] = (y + 1) * vert_res + x; // top left
	}

	//conPrint("Creating mesh took           " + timer.elapsedStringMSWIthNSigFigs(4));

	mesh_data->indices_vbo_handle = opengl_engine->vert_buf_allocator->allocateIndexData(mesh_data->vert_index_buffer.data(), mesh_data->vert_index_buffer.dataSizeBytes());

	mesh_data->vbo_handle = opengl_engine->vert_buf_allocator->allocate(mesh_data->vertex_spec, mesh_data->vert_data.data(), mesh_data->vert_data.dataSizeBytes());

#if DO_INDIVIDUAL_VAO_ALLOC
	mesh_data->individual_vao = new VAO(mesh_data->vbo_handle.vbo, mesh_data->indices_vbo_handle.index_vbo, mesh_data->vertex_spec);
#endif

	return mesh_data;
}


struct TerrainStats
{
	float total_volume;
};

TerrainStats computeTerrainStats(Simulation& sim)
{
	double sum_terrain_h = 0;
	for(int y=0; y<H; ++y)
	for(int x=0; x<W; ++x)
	{
		sum_terrain_h += sim.terrain_state.elem(x, y).height;
	}
	TerrainStats stats;
	stats.total_volume = (float)sum_terrain_h; // TODO: take into account cell width when != 1.
	return stats;
}




struct UpdateTexResults
{
	float max_value;
};

UpdateTexResults updateTerrainTexture(Simulation& sim, OpenGLTextureRef texture, TextureShow cur_texture_show, float tex_display_max_val)
{
	//const int W = sim.state1.getWidth();
	//const int H = sim.state1.getHeight();
	std::vector<uint8> tex_data(W * H * 3);

	/*conPrint("--------------------");
	for(int x=0; x<W; ++x)
	{
		const float water_h = sim.current_state->elem(x, H/2).water;
		printVar(water_h);
	}*/

	//conPrint("--------------------");
	//for(int x=0; x<W; ++x)
	//{
	//	const int y = H/2;
	//	const float suspended = sim.terrain_state.elem(x, y).suspended;
	//	const float water_h = sim.terrain_state.elem(x, y).water;
	//	const float water_speed = sqrt(Maths::square(sim.terrain_state.elem(x, y).u) + Maths::square(sim.terrain_state.elem(x, y).v));
	//	const float flux_R = sim.flow_state.elem(x, y).f_R;
	//	conPrint("x=" + toString(x) + ",y=" + toString(H/2) + ", suspended: " + toString(suspended) + ", water_h: " + toString(water_h) + ", water_speed: " + toString(water_speed) + ", flux_R: " + toString(flux_R));
	//}

	/*conPrint("--------------------");
	for(int x=0; x<W; ++x)
	{
		const int y = H/2;
		const float water_speed = sqrt(Maths::square(sim.terrain_state.elem(x, y).u) + Maths::square(sim.terrain_state.elem(x, y).v));
		printVar(water_speed);

		const float water_h = sim.terrain_state.elem(x, y).water;
		printVar(water_h);
	}*/
	UpdateTexResults results;
	results.max_value = 0;


	if(cur_texture_show == TextureShow_WaterDepth)
	{
		float max_water_h = 0;
		for(int y=0; y<H; ++y)
		for(int x=0; x<W; ++x)
		{
			const float water_h = sim.terrain_state.elem(x, y).water;
			max_water_h = myMax(max_water_h, water_h);
			const float water_val = myMax(0.0f, water_h / tex_display_max_val);

			tex_data[(y*W + x)*3 + 0] = 0;
			tex_data[(y*W + x)*3 + 1] = 0;
			tex_data[(y*W + x)*3 + 2] = (uint8)myClamp<int>((int)(water_val * 255), 0, 255);
		}
		results.max_value = max_water_h;
	}
	else if(cur_texture_show == TextureShow_WaterSpeed)
	{
		float max_water_speed = 0;
		for(int y=0; y<H; ++y)
		for(int x=0; x<W; ++x)
		{
			const float water_speed = sqrt(Maths::square(sim.terrain_state.elem(x, y).u) + Maths::square(sim.terrain_state.elem(x, y).v));
			max_water_speed = myMax(max_water_speed, water_speed);
			const float water_speed_val = water_speed / tex_display_max_val;

			tex_data[(y*W + x)*3 + 0] = 0;
			tex_data[(y*W + x)*3 + 1] = 0;
			tex_data[(y*W + x)*3 + 2] = (uint8)myClamp<int>((int)(0 + water_speed_val * 255), 0, 255);
		}
		results.max_value = max_water_speed;
	}
	else if(cur_texture_show == TextureShow_SuspendedSediment)
	{
		float max_suspended = 0;
		for(int y=0; y<H; ++y)
		for(int x=0; x<W; ++x)
		{
			const float sediment_h = sim.terrain_state.elem(x, y).suspended;
			max_suspended = myMax(max_suspended, sediment_h);
			const float sediment_val = sediment_h / tex_display_max_val;

			tex_data[(y*W + x)*3 + 0] = 0;
			tex_data[(y*W + x)*3 + 1] = (uint8)myClamp<int>((int)(0 + sediment_val * 255), 0, 255);
			tex_data[(y*W + x)*3 + 2] = 0;
		}
		results.max_value = max_suspended;
	}

	texture->loadIntoExistingTexture(/*mipmap level=*/0, sim.terrain_state.getHeight(), sim.terrain_state.getWidth(), /*row stride B=*/sim.terrain_state.getWidth()*3, tex_data, /*bind needed=*/true);

	return results;
}


void setGLAttribute(SDL_GLattr attr, int value)
{
	const int result = SDL_GL_SetAttribute(attr, value);
	if(result != 0)
	{
		const char* err = SDL_GetError();
		throw glare::Exception("Failed to set OpenGL attribute: " + (err ? std::string(err) : "[Unknown]"));
	}
}


void resetTerrain(Simulation& sim, OpenCLCommandQueueRef command_queue, InitialTerrainShape initial_terrain_shape)
{
	sim.sim_iteration = 0;

	// Set initial state
	TerrainState f;
	f.height = 1.f;
	f.water = 0.0f;
	f.suspended = 0.f;
	f.u = f.v = 0;

	sim.terrain_state.setAllElems(f);

	FlowState flow_state;
	flow_state.f_L = flow_state.f_R = flow_state.f_T = flow_state.f_B = 0;

	sim.flow_state.setAllElems(flow_state);

	if(initial_terrain_shape == InitialTerrainShape::InitialTerrainShape_Hat)
	{
		for(int x=0; x<W; ++x)
		for(int y=0; y<H; ++y)
		{
			float nx = (float)x / W;
			const float tent = (nx < 0.5) ? nx : (1.0f - nx);
			sim.terrain_state.elem(x, y).height = myMax(0.0f, tent*2 - 0.5f) * W / 2.0f;
		}
	}
	else if(initial_terrain_shape == InitialTerrainShape::InitialTerrainShape_FBM)
	{
		for(int x=0; x<W; ++x)
		for(int y=0; y<H; ++y)
		{
			float nx = (float)x / W;
			float ny = (float)y / H;

			//sim.terrain_state.elem(x, y).height = (-nx*nx + nx) * 200.0f;
			//const float r = Vec2f((float)x, (float)y).getDist(Vec2f((float)W/2, (float)H/2));

			//sim.terrain_state.elem(x, y).height = r < (W/4.0) ? 100.f : 0.f;

			const float perlin_factor = PerlinNoise::FBM(nx * 1.f, ny * 1.f, 10) + 1.f;
			sim.terrain_state.elem(x, y).height = perlin_factor * W / 5.0f;// * myMax(1 - (1.1f * r / ((float)W/2)), 0.f) * 200.f;
			//sim.terrain_state.elem(x, y).height = nx < 0.25 ? 0 : (nx < 0.5 ? (nx - 0.25) : (1.0f - (nx-0.25)) * 200.f;
			//const float tent = (nx < 0.5) ? nx : (1.0f - nx);
			//sim.terrain_state.elem(x, y).height = myMax(0.0f, tent*2 - 0.5f) * 200.f;
		}
	}
	else if(initial_terrain_shape == InitialTerrainShape::InitialTerrainShape_Perlin)
	{
		for(int x=0; x<W; ++x)
		for(int y=0; y<H; ++y)
		{
			float nx = (float)x / W;
			float ny = (float)y / H;

			const float perlin_factor = PerlinNoise::noise(nx * 1.f, ny * 1.f) + 1.f;
			sim.terrain_state.elem(x, y).height = perlin_factor * W / 5.0f;
		}
	}

	sim.terrain_state_buffer.copyFrom(command_queue, /*src ptr=*/&sim.terrain_state.elem(0, 0), /*size=*/W * H * sizeof(TerrainState), CL_MEM_READ_WRITE);
}


int main(int, char**)
{
	Clock::init();

	try
	{
		Simulation sim;
		
		//=========================== Init OpenCL================================
		OpenCL* opencl = getGlobalOpenCL();
		if(!opencl)
			throw glare::Exception("Failed to open OpenCL: " + getGlobalOpenCLLastErrorMsg());


		const std::vector<OpenCLDeviceRef> devices = opencl->getOpenCLDevices();

		if(devices.empty())
			throw glare::Exception("No OpenCL devices found");

		OpenCLDeviceRef opencl_device = devices[0]; // Just use first device for now.

		OpenCLContextRef opencl_context = new OpenCLContext(opencl_device);

		std::vector<OpenCLDeviceRef> devices_to_build_for(1, opencl_device);

		const bool profile = false;

		OpenCLCommandQueueRef command_queue = new OpenCLCommandQueue(opencl_context, opencl_device->opencl_device_id, profile);

		std::string build_log;
		OpenCLProgramRef program;
		try
		{
			// Prepend some definitions to the source code
			std::string src = FileUtils::readEntireFile("N:\\terraingen\\trunk\\erosion_kernel.cl");
			src = 
				"#define W " + toString(W) + "\n" +
				"#define H " + toString(H) + "\n" +
				src;

			program = opencl->buildProgram(
				src,
				opencl_context,
				devices_to_build_for,
				"-cl-mad-enable", // compile options
				build_log
			);
		}
		catch(glare::Exception& e)
		{
			conPrint("Build log: " + build_log);
			throw e;
		}

		Constants constants;
		constants.delta_t = 0.001f; // time step
		constants.r = 0.012f; // rainfall rate
		constants.A = 1; // cross-sectional 'pipe' area
		constants.g = 9.81f; // gravity accel.  NOTE: should be negative?
		constants.l = 1.0; // l = pipe length
		constants.l_x = 1; // width between grid points in x direction
		constants.l_y = 1;

		constants.K_c = 0.01f; // sediment capacity constant
		constants.K_s = 0.1f; // dissolving constant.
		constants.K_d = 0.1f; // deposition constant
		constants.K_dmax = 1.f;
		constants.K_e = 1.0; // Evaporation constant
		constants.K_t = 1.0; // Thermal erosion constant
		constants.max_talus_angle = Maths::pi<float>()/4;
		constants.tan_max_talus_angle = std::tan(constants.max_talus_angle);

		sim.terrain_state_buffer.alloc(opencl_context, /*size=*/W * H * sizeof(TerrainState), CL_MEM_READ_WRITE);
		sim.flow_state_buffer_a.alloc(opencl_context, W * H * sizeof(FlowState), CL_MEM_READ_WRITE);
		sim.flow_state_buffer_b.alloc(opencl_context, W * H * sizeof(FlowState), CL_MEM_READ_WRITE);
		sim.thermal_erosion_state_buffer.alloc(opencl_context, W * H * sizeof(ThermalErosionState), CL_MEM_READ_WRITE);
		sim.constants_buffer.allocFrom(opencl_context, &constants, sizeof(Constants), CL_MEM_READ_ONLY);

		sim.flowSimulationKernel = new OpenCLKernel(program, "flowSimulationKernel", opencl_device->opencl_device_id, profile);
		sim.thermalErosionFluxKernel = new OpenCLKernel(program, "thermalErosionFluxKernel", opencl_device->opencl_device_id, profile);
		sim.waterAndVelFieldUpdateKernel = new OpenCLKernel(program, "waterAndVelFieldUpdateKernel", opencl_device->opencl_device_id, profile);
		sim.erosionAndDepositionKernel = new OpenCLKernel(program, "erosionAndDepositionKernel", opencl_device->opencl_device_id, profile);
		sim.sedimentTransportationKernel = new OpenCLKernel(program, "sedimentTransportationKernel", opencl_device->opencl_device_id, profile);
		sim.thermalErosionMovementKernel = new OpenCLKernel(program, "thermalErosionMovementKernel", opencl_device->opencl_device_id, profile);
		sim.evaporationKernel = new OpenCLKernel(program, "evaporationKernel", opencl_device->opencl_device_id, profile);
		

		


		//int primary_W = 1680;
		//int primary_H = 1000;
		int primary_W = 1280;
		int primary_H = 720;
		uint32 window_flags = SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN;

		conPrint("Using primary window resolution   " + toString(primary_W) + " x " + toString(primary_H));

		SDL_Window* win = SDL_CreateWindow("TerrainGen", 100, 100, 1920, 1080, window_flags | SDL_WINDOW_RESIZABLE);
		if(win == nullptr)
			throw glare::Exception("SDL_CreateWindow Error: " + std::string(SDL_GetError()));

		//setGLAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
		//setGLAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
		setGLAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

		setGLAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

		SDL_GLContext gl_context = SDL_GL_CreateContext(win);
		if(!gl_context)
			throw glare::Exception("OpenGL context could not be created! SDL Error: " + std::string(SDL_GetError()));

		if(SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1) != 0)
			throw glare::Exception("SDL_GL_SetAttribute Error: " + std::string(SDL_GetError()));


		gl3wInit();


		// Initialise ImGUI
		ImGui::CreateContext();

		ImGui_ImplSDL2_InitForOpenGL(win, gl_context);
		ImGui_ImplOpenGL3_Init();


		// Create OpenGL engine
		OpenGLEngineSettings settings;
		settings.compress_textures = true;
		settings.shadow_mapping = true;
		Reference<OpenGLEngine> opengl_engine = new OpenGLEngine(settings);
		opengl_engine->are_8bit_textures_sRGB = true;

		TextureServer* texture_server = new TextureServer(/*use_canonical_path_keys=*/false);

		const std::string base_src_dir(BASE_SOURCE_DIR);
		//const std::string indigo_trunk_dir(INDIGO_TRUNK);

		const std::string data_dir = "N:\\glare-core\\trunk/opengl";
		StandardPrintOutput print_output;
		opengl_engine->initialise(data_dir, texture_server, &print_output);
		opengl_engine->setViewport(primary_W, primary_H);
		opengl_engine->setMainViewport(primary_W, primary_H);

		const std::string base_dir = ".";//base_src_dir;

		std::cout << "Finished compiling and linking program." << std::endl;


		const float sun_phi = 1.f;
		const float sun_theta = Maths::pi<float>() / 4;
		opengl_engine->setSunDir(normalise(Vec4f(std::cos(sun_phi) * sin(sun_theta), std::sin(sun_phi) * sin(sun_theta), cos(sun_theta), 0)));
		opengl_engine->setEnvMapTransform(Matrix3f::rotationMatrix(Vec3f(0,0,1), sun_phi));

		/*
		Set env material
		*/
		{
			OpenGLMaterial env_mat;
			env_mat.albedo_texture = opengl_engine->getTexture(base_dir + "/resources/sky_no_sun.exr");
			env_mat.albedo_texture->setTWrappingEnabled(false); // Disable wrapping in vertical direction to avoid grey dot straight up.
			
			env_mat.tex_matrix = Matrix2f(-1 / Maths::get2Pi<float>(), 0, 0, 1 / Maths::pi<float>());

			opengl_engine->setEnvMat(env_mat);
		}

		opengl_engine->setCirrusTexture(opengl_engine->getTexture(base_dir + "/resources/cirrus.exr"));


		//----------------------- Make ground plane -----------------------
		{
			GLObjectRef ground_plane = new GLObject();
			ground_plane->mesh_data = opengl_engine->getUnitQuadMeshData();
			ground_plane->ob_to_world_matrix = Matrix4f::uniformScaleMatrix(10) * Matrix4f::translationMatrix(-0.5f, -0.5f, 0);
			ground_plane->materials.resize(1);
			ground_plane->materials[0].albedo_texture = opengl_engine->getTexture(base_dir + "/resources/obstacle.png");
			ground_plane->materials[0].tex_matrix = Matrix2f(10.f, 0, 0, 10.f);

			//opengl_engine->addObject(ground_plane);
		}


		OpenGLTextureRef terrain_col_tex = new OpenGLTexture(W, H, opengl_engine.ptr(), ArrayRef<uint8>(NULL, 0), OpenGLTexture::Format_SRGB_Uint8, OpenGLTexture::Filtering_Bilinear);

		InitialTerrainShape initial_terrain_shape = InitialTerrainShape::InitialTerrainShape_Perlin;
		HeightFieldShow cur_heightfield_show = HeightFieldShow::HeightFieldShow_TerrainOnly;
		TextureShow cur_texture_show = TextureShow::TextureShow_WaterDepth;
		float tex_display_max_val = 1;



		resetTerrain(sim, command_queue, initial_terrain_shape);


		// Add terrain object
		GLObjectRef terrain_gl_ob = new GLObject();
		terrain_gl_ob->ob_to_world_matrix = Matrix4f::uniformScaleMatrix(0.002f);
		terrain_gl_ob->mesh_data = makeTerrainMesh(sim, opengl_engine.ptr(), cur_heightfield_show);

		

		UpdateTexResults results = updateTerrainTexture(sim, terrain_col_tex, cur_texture_show, tex_display_max_val);

		terrain_gl_ob->materials.resize(1);
		terrain_gl_ob->materials[0].albedo_linear_rgb = Colour3f(0.5f, 0.6f, 0.5f);
		terrain_gl_ob->materials[0].albedo_texture = terrain_col_tex;

		opengl_engine->addObject(terrain_gl_ob);
		

		Timer timer;
		Timer time_since_mesh_update;

		TerrainStats stats = computeTerrainStats(sim);

		float cam_phi = 0;
		float cam_theta = 1.f;
		Vec4f cam_target_pos = Vec4f(0,0,0,1);
		float cam_dist = 4;

		bool sim_running = true;
		

		

		bool quit = false;
		while(!quit)
		{
			//const double cur_time = timer.elapsed();
			

			//TEMP:
			if(SDL_GL_MakeCurrent(win, gl_context) != 0)
			{
				std::cout << "SDL_GL_MakeCurrent failed." << std::endl;
			}


			const Matrix4f T = Matrix4f::translationMatrix(0.f, cam_dist, 0.f);
			const Matrix4f z_rot = Matrix4f::rotationMatrix(Vec4f(0,0,1,0), cam_phi);
			const Matrix4f x_rot = Matrix4f::rotationMatrix(Vec4f(1,0,0,0), -(cam_theta - Maths::pi_2<float>()));
			const Matrix4f rot = x_rot * z_rot;
			const Matrix4f world_to_camera_space_matrix = T * rot * Matrix4f::translationMatrix(-cam_target_pos);

			const float sensor_width = 0.035f;
			const float lens_sensor_dist = 0.03f;
			const float render_aspect_ratio = opengl_engine->getViewPortAspectRatio();


			int gl_w, gl_h;
			SDL_GL_GetDrawableSize(win, &gl_w, &gl_h);

			opengl_engine->setViewport(gl_w, gl_h);
			opengl_engine->setMainViewport(gl_w, gl_h);
			opengl_engine->setMaxDrawDistance(100.f);
			opengl_engine->setPerspectiveCameraTransform(world_to_camera_space_matrix, sensor_width, lens_sensor_dist, render_aspect_ratio, /*lens shift up=*/0.f, /*lens shift right=*/0.f);
			opengl_engine->setCurrentTime((float)timer.elapsed());
			opengl_engine->draw();


			ImGuiIO& imgui_io = ImGui::GetIO();

			// Draw ImGUI GUI controls
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplSDL2_NewFrame(win);
			ImGui::NewFrame();

			//ImGui::ShowDemoWindow();

			ImGui::SetNextWindowSize(ImVec2(600, 700));
			ImGui::Begin("TerrainGen");

			ImGui::TextColored(ImVec4(1,1,0,1), "Simulation parameters");
			bool param_changed = false;
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"delta_t (s)", /*val=*/&constants.delta_t, /*min=*/0.0f, /*max=*/0.01f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"rainfall rate (m/s)", /*val=*/&constants.r, /*min=*/0.0f, /*max=*/0.01f, "%.5f");
			//param_changed = param_changed || ImGui::SliderFloat(/*label=*/"cross-sectional 'pipe' area (m)", /*val=*/&constants.A, /*min=*/0.0f, /*max=*/100.f, "%.5f");
			//param_changed = param_changed || ImGui::SliderFloat(/*label=*/"gravity mag (m/s^2)", /*val=*/&constants.g, /*min=*/0.0f, /*max=*/100.f, "%.5f");
			//param_changed = param_changed || ImGui::SliderFloat(/*label=*/"virtual pipe length (m)", /*val=*/&constants.l, /*min=*/0.0f, /*max=*/100.f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"sediment capacity constant (K_c) ", /*val=*/&constants.K_c, /*min=*/0.0f, /*max=*/4.f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"dissolving constant (K_s) ", /*val=*/&constants.K_s, /*min=*/0.0f, /*max=*/4.f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"deposition constant (K_d) ", /*val=*/&constants.K_d, /*min=*/0.0f, /*max=*/4.f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"erosion depth (K_dmax) ", /*val=*/&constants.K_dmax, /*min=*/0.0f, /*max=*/1.f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"evaporation constant (K_e) ", /*val=*/&constants.K_e, /*min=*/0.0f, /*max=*/1.f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"Thermal erosion constant (K_t) ", /*val=*/&constants.K_t, /*min=*/0.0f, /*max=*/10.f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"Max talus angle (rad) ", /*val=*/&constants.max_talus_angle, /*min=*/0.0f, /*max=*/1.5f, "%.5f");
			constants.tan_max_talus_angle = std::tan(constants.max_talus_angle);

			ImGui::Dummy(ImVec2(100, 40));
			ImGui::TextColored(ImVec4(1,1,0,1), "Visualisation");

			if(ImGui::BeginCombo("heightfield showing", HeightFieldShow_strings[cur_heightfield_show]))
			{
				for(int i=0; i<staticArrayNumElems(HeightFieldShow_strings); ++i)
				{
					const bool selected = cur_heightfield_show == i;
					if(ImGui::Selectable(HeightFieldShow_strings[i], selected))
						cur_heightfield_show = (HeightFieldShow)i;

					if(selected)
						ImGui::SetItemDefaultFocus();
				}

				ImGui::EndCombo();
			}

			if(ImGui::BeginCombo("texture showing", TextureShow_strings[cur_texture_show]))
			{
				for(int i=0; i<staticArrayNumElems(TextureShow_strings); ++i)
				{
					const bool selected = cur_texture_show == i;
					if(ImGui::Selectable(TextureShow_strings[i], selected))
						cur_texture_show = (TextureShow)i;
					if(selected)
						ImGui::SetItemDefaultFocus();
				}

				ImGui::EndCombo();
			}

			ImGui::SliderFloat(/*label=*/"Texture display max val", /*val=*/&tex_display_max_val, /*min=*/0.0f, /*max=*/10.f, "%.5f");


			ImGui::Dummy(ImVec2(100, 40));
			ImGui::TextColored(ImVec4(1,1,0,1), "Simulation control");
			

			bool do_advance = false;
			if(sim_running)
			{
				do_advance = true;
				if(ImGui::Button("pause"))
					sim_running = false;
			}
			else // Else if paused:
			{
				if(ImGui::Button("resume"))
					sim_running = true;

				const bool single_step = ImGui::Button("single step");
				if(single_step)
					do_advance = true;
			}
			
			if(do_advance)
				sim.doSimIteration(command_queue);

			ImGui::Dummy(ImVec2(100, 20));
			if(ImGui::BeginCombo("initial heightfield shape", InitialTerrainShape_strings[initial_terrain_shape]))
			{
				for(int i=0; i<staticArrayNumElems(InitialTerrainShape_strings); ++i)
				{
					const bool selected = initial_terrain_shape == i;
					if(ImGui::Selectable(InitialTerrainShape_strings[i], selected))
						initial_terrain_shape = (InitialTerrainShape)i;
					if(selected)
						ImGui::SetItemDefaultFocus();
				}

				ImGui::EndCombo();
			}

			ImGui::Dummy(ImVec2(100, 20));
			const bool reset = ImGui::Button("Reset terrain");

			ImGui::Dummy(ImVec2(100, 40));
			ImGui::TextColored(ImVec4(1,1,0,1), "Info");
			ImGui::Text((std::string(sim_running ? "Sim running" : "Sim paused") + ", iteration: " + toString(sim.sim_iteration)).c_str());
			
			ImGui::Text(("Total terrain volume: " + doubleToStringScientific(stats.total_volume, 4) + "m^3").c_str());
			ImGui::Text(("max texture value: " + toString(results.max_value)).c_str());
			ImGui::End(); 

			if(reset)
			{
				resetTerrain(sim, command_queue, initial_terrain_shape);
			}
			if(param_changed)
			{
				sim.constants_buffer.copyFrom(command_queue, &constants, sizeof(Constants), /*blocking write=*/true);
			}

			

			// Update terrain object
			if(time_since_mesh_update.elapsed() > 0.1)
			{
				sim.readBackToCPUMem(command_queue);

				opengl_engine->removeObject(terrain_gl_ob);
				terrain_gl_ob = NULL;

				results = updateTerrainTexture(sim, terrain_col_tex, cur_texture_show, tex_display_max_val);

				terrain_gl_ob = new GLObject();
				terrain_gl_ob->ob_to_world_matrix = Matrix4f::translationMatrix(0, 0, 0.01f) * Matrix4f::uniformScaleMatrix(0.002f);
				terrain_gl_ob->mesh_data = makeTerrainMesh(sim, opengl_engine.ptr(), cur_heightfield_show);


				terrain_gl_ob->materials.resize(1);
				terrain_gl_ob->materials[0].albedo_linear_rgb = Colour3f(0.5f, 0.6f, 0.5f);
				terrain_gl_ob->materials[0].albedo_texture = terrain_col_tex;

				opengl_engine->addObject(terrain_gl_ob);


				// Update statistics
				stats = computeTerrainStats(sim);

				time_since_mesh_update.reset();
			}


			

			

			
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

			// Display
			SDL_GL_SwapWindow(win);

			// Handle any events
			SDL_Event e;
			while(SDL_PollEvent(&e))
			{
				if(imgui_io.WantCaptureMouse)
				{
					ImGui_ImplSDL2_ProcessEvent(&e); // Pass event onto ImGUI
					continue;
				}

				if(e.type == SDL_QUIT) // "An SDL_QUIT event is generated when the user clicks on the close button of the last existing window" - https://wiki.libsdl.org/SDL_EventType#Remarks
					quit = true;
				else if(e.type == SDL_WINDOWEVENT) // If user closes the window:
				{
					if(e.window.event == SDL_WINDOWEVENT_CLOSE)
						quit = true;
					else if(e.window.event == SDL_WINDOWEVENT_RESIZED || e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
					{
						int w, h;
						SDL_GL_GetDrawableSize(win, &w, &h);
						
						opengl_engine->setViewport(w, h);
						opengl_engine->setMainViewport(w, h);
					}
				}
				else if(e.type == SDL_KEYDOWN)
				{
				}
				else if(e.type == SDL_MOUSEMOTION)
				{
					//conPrint("SDL_MOUSEMOTION");
					if(e.motion.state & SDL_BUTTON_LMASK)
					{
						//conPrint("SDL_BUTTON_LMASK down");

						const float move_scale = 0.005f;
						cam_phi += e.motion.xrel * move_scale;
						cam_theta = myClamp<float>(cam_theta - (float)e.motion.yrel * move_scale, 0.01f, Maths::pi<float>() - 0.01f);
					}

					if((e.motion.state & SDL_BUTTON_MMASK) || (e.motion.state & SDL_BUTTON_RMASK))
					{
						//conPrint("SDL_BUTTON_MMASK or SDL_BUTTON_RMASK down");

						const float move_scale = 0.005f;

						const Vec4f forwards = GeometrySampling::dirForSphericalCoords(-cam_phi + Maths::pi_2<float>(), Maths::pi<float>() - cam_theta);
						const Vec4f right = normalise(crossProduct(forwards, Vec4f(0,0,1,0)));
						const Vec4f up = crossProduct(right, forwards);

						cam_target_pos += right * -(float)e.motion.xrel * move_scale + up * (float)e.motion.yrel * move_scale;
					}
				}
				else if(e.type == SDL_MOUSEWHEEL)
				{
					//conPrint("SDL_MOUSEWHEEL");
					cam_dist = myClamp<float>(cam_dist - cam_dist * e.wheel.y * 0.2f, 0.01f, 10000.f);
				}
			}
		}
		SDL_Quit();
		return 0;
	}
	catch(glare::Exception& e)
	{
		std::cout << e.what() << std::endl;
		return 1;
	}
}
