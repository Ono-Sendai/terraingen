/*=====================================================================
terraingen.cpp
--------------
Copyright Nicholas Chapman 2024 -
=====================================================================*/


#include <graphics/PNGDecoder.h>
#include <graphics/ImageMap.h>
#include <graphics/EXRDecoder.h>
#include <maths/GeometrySampling.h>
#include <dll/include/IndigoMesh.h>
#include <graphics/PerlinNoise.h>
#include <graphics/SRGBUtils.h>
#include <opengl/OpenGLShader.h>
#include <opengl/OpenGLProgram.h>
#include <opengl/OpenGLEngine.h>
#include <opengl/GLMeshBuilding.h>
#include <indigo/TextureServer.h>
#include <maths/PCG32.h>
#include <opengl/VBO.h>
#include <opengl/VAO.h>
#include <opengl/MeshPrimitiveBuilding.h>
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
#include <backends/imgui_impl_sdl2.h>
#include <fstream>
#include <string>


typedef struct
{
	float height; // terrain height (b)
	float water; // water height (d)
	float suspended;
	float deposited_sed;
	float u, v; // velocity
	float water_vel;
	//float sed_flux;
} TerrainState;


typedef struct
{
	float f_L, f_R, f_T, f_B; // outflow flux
	float sed_f_L, sed_f_R, sed_f_T, sed_f_B;
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
	float f; // fricton constant
	float cell_w; // Width of cell = spacing between grid cells (metres)
	float recip_cell_w; // 1 / cell_w

	float K_c;// = 0.01; // 1; // sediment capacity constant
	float K_s;// = 0.01; // 0.5; // dissolving constant.
	float K_d;// = 0.01; // 1; // deposition constant
	float K_dmax;// = 0.1f; // Maximum erosion depth: water depth at which erosion stops.
	float q_0; // Minimum unit water discharge for sediment carrying.
	float K_e; // Evaporation constant

	float K_t; // thermal erosion constant
	float K_tdep; // thermal erosion constant for deposited sediment
	float max_talus_angle;
	float tan_max_talus_angle;
	float max_deposited_talus_angle;
	float tan_max_deposited_talus_angle;
	float sea_level;
	float current_time;

	int include_water_height;
	int draw_water;
	Colour3f rock_col;
	Colour3f sediment_col;
	Colour3f vegetation_col;
} Constants;


class Simulation
{
public:
	int sim_iteration;
	int W, H;

	Array2D<TerrainState> terrain_state;
	Array2D<FlowState> flow_state;
	Array2D<ThermalErosionState> thermal_erosion_state;


	OpenCLKernelRef flowSimulationKernel;
	OpenCLKernelRef thermalErosionFluxKernel;
	OpenCLKernelRef thermalErosionDepositedFluxKernel;
	OpenCLKernelRef waterAndVelFieldUpdateKernel;
	OpenCLKernelRef erosionAndDepositionKernel;
	OpenCLKernelRef sedimentTransportationKernel;
	OpenCLKernelRef thermalErosionMovementKernel;
	OpenCLKernelRef thermalErosionDepositedMovementKernel;
	OpenCLKernelRef evaporationKernel;
	OpenCLKernelRef setHeightFieldMeshKernel;

	OpenCLBuffer terrain_state_buffer;
	OpenCLBuffer flow_state_buffer_a;
	OpenCLBuffer flow_state_buffer_b;
	OpenCLBuffer thermal_erosion_state_buffer;
	OpenCLBuffer constants_buffer;

	cl_mem heightfield_mesh_buffer;
	uint32 heightfield_mesh_offset_B;
	//cl_mem water_heightfield_mesh_buffer;
	//uint32 water_heightfield_mesh_offset_B;
	cl_mem terrain_tex_cl_mem;


	Timer timer;

	Simulation(int W_, int H_)
	{
		W = W_;
		H = H_;

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

		const int num_iters = 2; // Should be even so that we end up with cur_flow_state_buffer == flow_state_buffer_a
		for(int z=0; z<num_iters; ++z)
		{
			flowSimulationKernel->setKernelArgBuffer(0, terrain_state_buffer);
			flowSimulationKernel->setKernelArgBuffer(1, *cur_flow_state_buffer); // source
			flowSimulationKernel->setKernelArgBuffer(2, *other_flow_state_buffer); // destination
			flowSimulationKernel->setKernelArgBuffer(3, constants_buffer);
			flowSimulationKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);

			mySwap(cur_flow_state_buffer, other_flow_state_buffer); // Swap pointers

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
			
			thermalErosionFluxKernel->setKernelArgBuffer(0, terrain_state_buffer);
			thermalErosionFluxKernel->setKernelArgBuffer(1, thermal_erosion_state_buffer); // source
			thermalErosionFluxKernel->setKernelArgBuffer(2, constants_buffer);
			thermalErosionFluxKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);

			thermalErosionMovementKernel->setKernelArgBuffer(0, thermal_erosion_state_buffer);
			thermalErosionMovementKernel->setKernelArgBuffer(1, terrain_state_buffer);
			thermalErosionMovementKernel->setKernelArgBuffer(2, constants_buffer);
			thermalErosionMovementKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);

			thermalErosionDepositedFluxKernel->setKernelArgBuffer(0, terrain_state_buffer);
			thermalErosionDepositedFluxKernel->setKernelArgBuffer(1, thermal_erosion_state_buffer); // source
			thermalErosionDepositedFluxKernel->setKernelArgBuffer(2, constants_buffer);
			thermalErosionDepositedFluxKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);

			thermalErosionDepositedMovementKernel->setKernelArgBuffer(0, thermal_erosion_state_buffer);
			thermalErosionDepositedMovementKernel->setKernelArgBuffer(1, terrain_state_buffer);
			thermalErosionDepositedMovementKernel->setKernelArgBuffer(2, constants_buffer);
			thermalErosionDepositedMovementKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);

			evaporationKernel->setKernelArgBuffer(0, terrain_state_buffer);
			evaporationKernel->setKernelArgBuffer(1, constants_buffer);
			evaporationKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);
		}

		assert(cur_flow_state_buffer == &flow_state_buffer_a);

		sim_iteration += num_iters;
	}

	void updateHeightFieldMeshAndTexture(OpenCLCommandQueueRef command_queue)
	{
		const cl_mem mem_objects[] = { heightfield_mesh_buffer, /*water_heightfield_mesh_buffer, */terrain_tex_cl_mem };
		::getGlobalOpenCL()->clEnqueueAcquireGLObjects(command_queue->getCommandQueue(),
			/*num objects=*/staticArrayNumElems(mem_objects), /*mem objects=*/mem_objects, /*num objects in wait list=*/0, /*event wait list=*/NULL, /*event=*/NULL);
		
		setHeightFieldMeshKernel->setKernelArgBuffer(0, terrain_state_buffer);
		setHeightFieldMeshKernel->setKernelArgBuffer(1, constants_buffer);
		setHeightFieldMeshKernel->setKernelArgBuffer(2, heightfield_mesh_buffer);
		setHeightFieldMeshKernel->setKernelArgUInt(3, heightfield_mesh_offset_B);
		setHeightFieldMeshKernel->setKernelArgBuffer(4, terrain_tex_cl_mem);
		//setHeightFieldMeshKernel->setKernelArgBuffer(5, water_heightfield_mesh_buffer);
		//setHeightFieldMeshKernel->setKernelArgUInt(6, water_heightfield_mesh_offset_B);

		
		setHeightFieldMeshKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);

		::getGlobalOpenCL()->clEnqueueReleaseGLObjects(command_queue->getCommandQueue(),
			/*num objects=*/staticArrayNumElems(mem_objects), /*mem objects=*/mem_objects, /*num objects in wait list=*/0, /*event wait list=*/NULL, /*event=*/NULL);

		command_queue->finish(); // Make sure we have finished writing to the vertex buffer and texture before we start issuing new OpenGL commands.
		// TODO: work out a better way of syncing (e.g. with barrier).
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
	TextureShow_SuspendedSediment,
	TextureShow_DepositedSediment
};
const char* TextureShow_strings[] = 
{
	"water depth",
	"water speed",
	"suspended sediment",
	"deposited sediment"
};


enum InitialTerrainShape
{
	InitialTerrainShape_ConstantSlope,
	InitialTerrainShape_Hat,
	InitialTerrainShape_Cone,
	InitialTerrainShape_FBM,
	InitialTerrainShape_Perlin
};

const char* InitialTerrainShape_strings[] = 
{
	"constant slope",
	"hat",
	"cone",
	"FBM",
	"Perlin noise"
};


inline float totalTerrainHeight(const TerrainState& state)
{
	return state.height + state.deposited_sed;//state.sediment[0] + state.sediment[1] + state.sediment[2];
}


OpenGLMeshRenderDataRef makeTerrainMesh(const Simulation& sim, OpenGLEngine* opengl_engine, float cell_w, HeightFieldShow /*cur_heightfield_show*/) 
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

	int vert_xres = myMax(2, (int)sim.terrain_state.getWidth());
	int vert_yres = myMax(2, (int)sim.terrain_state.getHeight());
	int quad_xres = vert_xres - 1; // Number of quads in x and y directions
	int quad_yres = vert_yres - 1; // Number of quads in x and y directions
	
	float quad_w_x = cell_w;
	float quad_w_y = quad_w_x;
	if(sim.terrain_state.getHeight() <= 10)
	{
		quad_w_y *= 20.f; // For height = 1 (1-d debugging case), display strip a bit wider
	}

	//const size_t normal_size_B = 4;
	const size_t normal_size_B = sizeof(float) * 3;
	const size_t vert_size_B = sizeof(float) * (3 + 2) + normal_size_B; // position, normal, uv
	OpenGLMeshRenderDataRef mesh_data = new OpenGLMeshRenderData();
	mesh_data->vert_data.resize(vert_size_B * vert_xres * vert_yres);

	mesh_data->vert_index_buffer.resize(quad_xres * quad_yres * 6);

	OpenGLMeshRenderData& meshdata = *mesh_data;

	meshdata.has_uvs = true;
	meshdata.has_shading_normals = true;
	meshdata.batches.resize(1);
	meshdata.batches[0].material_index = 0;
	meshdata.batches[0].num_indices = (uint32)meshdata.vert_index_buffer.size();
	meshdata.batches[0].prim_start_offset_B = 0;

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

	//VertexAttrib normal_attrib;
	//normal_attrib.enabled = true;
	//normal_attrib.num_comps = 4; // 3;
	//normal_attrib.type = GL_INT_2_10_10_10_REV; // GL_FLOAT;
	//normal_attrib.normalised = true; // false;
	//normal_attrib.stride = vert_size_B;
	//normal_attrib.offset = (uint32)in_vert_offset_B;
	//meshdata.vertex_spec.attributes.push_back(normal_attrib);
	//in_vert_offset_B += normal_size_B;
	VertexAttrib normal_attrib;
	normal_attrib.enabled = true;
	normal_attrib.num_comps = 3;
	normal_attrib.type = GL_FLOAT;
	normal_attrib.normalised = false;
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

	//js::AABBox aabb_os = js::AABBox::emptyAABBox();

	uint8* const vert_data = mesh_data->vert_data.data();

	//Timer timer;

	for(int y=0; y<vert_yres; ++y)
	for(int x=0; x<vert_xres; ++x)
	{
		const float p_x = x * quad_w_x;
		const float p_y = y * quad_w_y;

		const Vec3f pos(p_x, p_y, 0);
		std::memcpy(vert_data + vert_size_B * (y * vert_xres + x), &pos, sizeof(float)*3);

		//aabb_os.enlargeToHoldPoint(pos.toVec4fPoint());

		const Vec3f normal(0,0,1);
		std::memcpy(vert_data + vert_size_B * (y * vert_xres + x) + sizeof(float) * 3, &normal, sizeof(float) * 3);

		Vec2f uv((float)x / vert_xres, (float)y / vert_yres);
		std::memcpy(vert_data + vert_size_B * (y * vert_xres + x) + uv_offset_B, &uv, sizeof(float)*2);
	}

	// Make AABB bigger in z direction than initial mesh so it encloses displaced geom updated on GPU
	meshdata.aabb_os = js::AABBox(
		Vec4f(-100000.f,-100000.f,-100000.f,1),
		Vec4f( 100000.f, 100000.f, 100000.f,1)
	);
	//meshdata.aabb_os = aabb_os;
	//meshdata.aabb_os.min_[2] = -100000.f; // Make AABB bigger so encloses displaced geom updated on GPU
	//meshdata.aabb_os.max_[2] = 100000.f;


	uint32* const indices = (uint32*)mesh_data->vert_index_buffer.data();
	for(int y=0; y<quad_yres; ++y)
	for(int x=0; x<quad_xres; ++x)
	{
		// Trianglulate the quad in this way
		// |----|
		// | \  |
		// |  \ |
		// |   \|
		// |----|--> x

		// bot left tri
		int offset = (y*quad_xres + x) * 6;
		indices[offset + 0] = y * vert_xres + x; // bot left
		indices[offset + 1] = y * vert_xres + x + 1; // bot right
		indices[offset + 2] = (y + 1) * vert_xres + x; // top left

		// top right tri
		indices[offset + 3] = y * vert_xres + x + 1; // bot right
		indices[offset + 4] = (y + 1) * vert_xres + x + 1; // top right
		indices[offset + 5] = (y + 1) * vert_xres + x; // top left
	}

	//conPrint("Creating mesh took           " + timer.elapsedStringMSWIthNSigFigs(4));

	mesh_data->indices_vbo_handle = opengl_engine->vert_buf_allocator->allocateIndexDataSpace(mesh_data->vert_index_buffer.data(), mesh_data->vert_index_buffer.dataSizeBytes());

	mesh_data->vbo_handle = opengl_engine->vert_buf_allocator->allocateVertexDataSpace(mesh_data->vertex_spec.vertStride(), mesh_data->vert_data.data(), mesh_data->vert_data.dataSizeBytes());

	opengl_engine->vert_buf_allocator->getOrCreateAndAssignVAOForMesh(*mesh_data, mesh_data->vertex_spec);

	return mesh_data;
}


struct TerrainStats
{
	float total_volume;
	float total_water_volume;
	float total_suspended_sediment_vol;
	float total_deposited_sediment_vol;
};

TerrainStats computeTerrainStats(Simulation& sim)
{
	double sum_terrain_h = 0;
	double sum_water_h = 0;
	double sum_suspended = 0;
	double sum_deposited_sediment = 0;
	for(int y=0; y<sim.H; ++y)
	for(int x=0; x<sim.W; ++x)
	{
		sum_terrain_h += sim.terrain_state.elem(x, y).height;
		sum_water_h   += sim.terrain_state.elem(x, y).water;
		sum_suspended += sim.terrain_state.elem(x, y).suspended;//[0] + sim.terrain_state.elem(x, y).suspended[1] + sim.terrain_state.elem(x, y).suspended[2];
		sum_deposited_sediment  += sim.terrain_state.elem(x, y).deposited_sed;//sediment[0] + sim.terrain_state.elem(x, y).sediment[1] + sim.terrain_state.elem(x, y).sediment[2];
	}
	TerrainStats stats;
	stats.total_volume = (float)sum_terrain_h; // TODO: take into account cell width when != 1.
	stats.total_water_volume = (float)sum_water_h; // TODO: take into account cell width when != 1.
	stats.total_suspended_sediment_vol = (float)sum_suspended; // TODO: take into account cell width when != 1.
	stats.total_deposited_sediment_vol = (float)sum_deposited_sediment; // TODO: take into account cell width when != 1.
	return stats;
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


void resetTerrain(Simulation& sim, OpenCLCommandQueueRef command_queue, InitialTerrainShape initial_terrain_shape, float cell_w, float initial_height_scale, float x_scale, float y_scale, float sea_level)
{
	const int W = (int)sim.terrain_state.getWidth();
	const int H = (int)sim.terrain_state.getHeight();

	sim.sim_iteration = 0;

	// Set initial state
	TerrainState f;
	f.height = 1.f;
	f.water = 0.0f;
	f.suspended = 0;
	f.deposited_sed = 0;
	//f.suspended[0] = f.suspended[1] = f.suspended[2] = 0.f;
	//f.sediment[0] = f.sediment[1] = f.sediment[2] = 0.f;
	f.u = f.v = 0;

	sim.terrain_state.setAllElems(f);

	FlowState flow_state;
	flow_state.f_L = flow_state.f_R = flow_state.f_T = flow_state.f_B = 0;
	flow_state.sed_f_L = flow_state.sed_f_R = flow_state.sed_f_T = flow_state.sed_f_B = 0;

	sim.flow_state.setAllElems(flow_state);

	const float total_vert_scale = cell_w * initial_height_scale;

	if(initial_terrain_shape == InitialTerrainShape::InitialTerrainShape_ConstantSlope)
	{
		for(int x=0; x<W; ++x)
		for(int y=0; y<H; ++y)
		{
			float nx = (float)x / W;
			sim.terrain_state.elem(x, y).height = nx * W / 2.0f * total_vert_scale;
		}
	}
	else if(initial_terrain_shape == InitialTerrainShape::InitialTerrainShape_Hat)
	{
		for(int x=0; x<W; ++x)
		for(int y=0; y<H; ++y)
		{
			float nx = (float)x / W;
			const float tent = (nx < 0.5) ? nx : (1.0f - nx);
			sim.terrain_state.elem(x, y).height = myMax(0.0f, tent*2 - 0.5f) * W / 2.0f * total_vert_scale;
		}
	}
	else if(initial_terrain_shape == InitialTerrainShape::InitialTerrainShape_Cone)
	{
		for(int x=0; x<W; ++x)
		for(int y=0; y<H; ++y)
		{
			float nx = (float)x / W;
			float ny = (float)y / H;
			const float r = Vec2f(nx, ny).getDist(Vec2f(0.5f));
			
			const float cone = myMax(0.0f, 1 - r*4);
			sim.terrain_state.elem(x, y).height = cone * W / 4.0f * total_vert_scale;
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

			const float perlin_factor = PerlinNoise::FBM(nx * x_scale, ny * y_scale, 12);
			//const float perlin_factor = PerlinNoise::ridgedFBM<float>(Vec4f(nx * 0.5f, ny * 0.5f, 0, 1), 1, 2, 5);
			sim.terrain_state.elem(x, y).height = myMax(-100.f, perlin_factor)/3.f * W / 5.0f * total_vert_scale;// * myMax(1 - (1.1f * r / ((float)W/2)), 0.f) * 200.f;
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

			const float perlin_factor = PerlinNoise::noise(nx * x_scale, ny * y_scale) + 1.f;
			sim.terrain_state.elem(x, y).height = perlin_factor * W / 5.0f * total_vert_scale;
		}
	}

	// Set water height based on sea level
	for(int y=0; y<H; ++y)
	for(int x=0; x<W; ++x)
	{
		const float total_terrain_h = sim.terrain_state.elem(x, y).height + sim.terrain_state.elem(x, y).deposited_sed;
		if(total_terrain_h < sea_level)
		{
			sim.terrain_state.elem(x, y).water = sea_level - total_terrain_h;
		}
	}

	// Upload to GPU
	sim.terrain_state_buffer.copyFrom(command_queue, /*src ptr=*/&sim.terrain_state.elem(0, 0), /*size=*/W * H * sizeof(TerrainState), CL_MEM_READ_WRITE);
}


void saveStructMemberToEXR(Simulation& sim, const std::string& exr_path, size_t member_offset_B)
{
	std::vector<float> data(sim.W * sim.H);
	for(int y=0; y<sim.H; ++y)
	for(int x=0; x<sim.W; ++x)
		data[x + y * sim.W] = *(float*)((uint8*)&sim.terrain_state.elem(x, y) + member_offset_B);

	EXRDecoder::SaveOptions options;
	EXRDecoder::saveImageToEXR(data.data(), sim.W, sim.H, /*num channels=*/1, /*save alpha channel=*/false, exr_path, /*layer name=*/"", options);
}


void saveHeightfieldToDisk(Simulation& sim, OpenCLCommandQueueRef command_queue)
{
	sim.readBackToCPUMem(command_queue);

	try
	{
		// Save heightfield
		saveStructMemberToEXR(sim, "heightfield.exr", offsetof(TerrainState, height));

		// Save a map of deposited sediment
		saveStructMemberToEXR(sim, "sediment_map.exr", offsetof(TerrainState, deposited_sed));

		{
			std::vector<float> data(sim.W * sim.H);
			for(int y=0; y<sim.H; ++y)
			for(int x=0; x<sim.W; ++x)
				data[x + y * sim.W] = sim.terrain_state.elem(x, y).height + sim.terrain_state.elem(x, y).deposited_sed;

			EXRDecoder::SaveOptions options;
			EXRDecoder::saveImageToEXR(data.data(), sim.W, sim.H, /*num channels=*/1, /*save alpha channel=*/false, "heightfield_with_deposited_sed.exr", /*layer name=*/"", options);
		}

		// Save a map of deposited sediment
		saveStructMemberToEXR(sim, "water.exr", offsetof(TerrainState, water));
	}
	catch(glare::Exception& e)
	{
		SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Failed to save", ("Failed to save heightfield or texture to disk: " + e.what()).c_str(), /*parent=*/NULL);
	}
}


void saveColourTextureToDisk(Simulation& sim, OpenCLCommandQueueRef command_queue, OpenGLTextureRef terrain_col_tex)
{
	command_queue->finish();
	glFinish();

	ImageMapUInt8 map(sim.W, sim.H, 4);
	terrain_col_tex->readBackTexture(/*mipmap level=*/0, ArrayRef<uint8>(map.getData(), map.getDataSize()));

	PNGDecoder::write(map, "colour.png");
}


void loadStructMemberFromEXR(Simulation& sim, const std::string& exr_path, size_t member_offset_B)
{
	Reference<Map2D> map = EXRDecoder::decode(exr_path);

	if(map.isType<ImageMapFloat>())
	{
		ImageMapFloatRef image_map = map.downcast<ImageMapFloat>();

		if(image_map->getWidth() != sim.W)
			throw glare::Exception("Width does not match");
		if(image_map->getHeight() != sim.H)
			throw glare::Exception("Height does not match");

		for(int y=0; y<sim.H; ++y)
			for(int x=0; x<sim.W; ++x)
				*((float*)((uint8*)&sim.terrain_state.elem(x, y) + member_offset_B)) = image_map->getPixel(x, y)[0];
	}
	else
		throw glare::Exception("Unhandled image type");
}


void loadHeightfieldFromDisk(Simulation& sim, OpenCLCommandQueueRef command_queue)
{
	conPrint("Loading heightfield from disk...");

	loadStructMemberFromEXR(sim, "heightfield.exr", offsetof(TerrainState, height));

	loadStructMemberFromEXR(sim, "sediment_map.exr", offsetof(TerrainState, deposited_sed));

	loadStructMemberFromEXR(sim, "water.exr", offsetof(TerrainState, water));

	// Upload to GPU
	sim.terrain_state_buffer.copyFrom(command_queue, /*src ptr=*/&sim.terrain_state.elem(0, 0), /*size=*/sim.W * sim.H * sizeof(TerrainState), CL_MEM_READ_WRITE);

	conPrint("done.");
}


int main(int, char**)
{
	Clock::init();

	try
	{
		//=========================== Init SDL and OpenGL ================================
		if(SDL_Init(SDL_INIT_VIDEO) != 0)
			throw glare::Exception("SDL_Init Error: " + std::string(SDL_GetError()));


		// Set GL attributes, needs to be done before window creation.
		setGLAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4); // We need to request a specific version for a core profile.
		setGLAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
		setGLAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

		setGLAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);


		// NOTE: OpenGL init needs to go before OpenCL init
		int W = 512;
		int H = 512;
		float cell_w = 8192.f / W;
		Simulation sim(W, H);


		int primary_W = 1920;
		int primary_H = 1080;

		SDL_Window* win = SDL_CreateWindow("TerrainGen", 100, 100, primary_W, primary_H, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
		if(win == nullptr)
			throw glare::Exception("SDL_CreateWindow Error: " + std::string(SDL_GetError()));


		SDL_GLContext gl_context = SDL_GL_CreateContext(win);
		if(!gl_context)
			throw glare::Exception("OpenGL context could not be created! SDL Error: " + std::string(SDL_GetError()));

		if(SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1) != 0)
			throw glare::Exception("SDL_GL_SetAttribute Error: " + std::string(SDL_GetError()));


		gl3wInit();


		
		//=========================== Init OpenCL================================
		OpenCL* opencl = getGlobalOpenCL();
		if(!opencl)
			throw glare::Exception("Failed to open OpenCL: " + getGlobalOpenCLLastErrorMsg());


		const std::vector<OpenCLDeviceRef> devices = opencl->getOpenCLDevices();

		if(devices.empty())
			throw glare::Exception("No OpenCL devices found");

		// Use first GPU device for now
		OpenCLDeviceRef opencl_device;
		for(size_t i=0; i<devices.size(); ++i)
		{
			if(devices[i]->opencl_device_type == CL_DEVICE_TYPE_GPU)
			{
				opencl_device = devices[i];
				break;
			}
		}

		if(opencl_device.isNull())
			throw glare::Exception("No OpenCL GPU devices found");
			
		OpenCLContextRef opencl_context = new OpenCLContext(opencl_device, /*enable opengl interop=*/true);

		std::vector<OpenCLDeviceRef> devices_to_build_for(1, opencl_device);

		const bool profile = false;

		OpenCLCommandQueueRef command_queue = new OpenCLCommandQueue(opencl_context, opencl_device->opencl_device_id, profile);

		const std::string base_src_dir(BASE_SOURCE_DIR);

		std::string build_log;
		OpenCLProgramRef program;
		try
		{
			// Prepend some definitions to the source code
			std::string src = FileUtils::readEntireFile(base_src_dir + "/erosion_kernel.cl");
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

			conPrint("Build log: " + build_log);
		}
		catch(glare::Exception& e)
		{
			conPrint("Build log: " + build_log);
			throw e;
		}

		Constants constants;
		constants.delta_t = 0.08f;//0.01f; // time step
		constants.r = 0.0025f; // 0.012f; // rainfall rate
		constants.A = 1; // cross-sectional 'pipe' area
		constants.g = 9.81f; // gravity accel.  NOTE: should be negative?
		constants.l = 1.0; // l = pipe length
		constants.f = 1.f; // 0.05f; // friction constant
		constants.cell_w = cell_w;
		constants.recip_cell_w = 1.f / cell_w;

		constants.K_c = 0.5f; // sediment capacity constant
		constants.K_s = 3; // 0.5f; // dissolving constant.
		constants.K_d = 0.5f; // deposition constant
		constants.K_dmax = 1.f;
		constants.q_0 = 0.2f;
		constants.K_e = 0.001f; // 0.005f; // Evaporation constant
		constants.K_t = 0.03f; // Thermal erosion constant
		constants.K_tdep = 0.03f; // thermal erosion constant for deposited sediment
		constants.max_talus_angle = Maths::pi<float>()/4;
		constants.tan_max_talus_angle = std::tan(constants.max_talus_angle);
		constants.max_deposited_talus_angle = 0.55f;
		constants.tan_max_deposited_talus_angle = std::tan(constants.max_deposited_talus_angle);
		constants.sea_level = -4.f;
		constants.current_time = 0;

		constants.include_water_height = 1;
		constants.draw_water = 1;
		constants.rock_col       = toLinearSRGB(Colour3f(63 / 255.f, 56 / 255.f, 51 / 255.f));
		constants.sediment_col   = toLinearSRGB(Colour3f(105 / 255.f, 97 / 255.f, 88 / 255.f));
		constants.vegetation_col = toLinearSRGB(Colour3f(27 / 255.f, 58 / 255.f, 37 / 255.f));

		sim.terrain_state_buffer.alloc(opencl_context, /*size=*/W * H * sizeof(TerrainState), CL_MEM_READ_WRITE);
		sim.flow_state_buffer_a.alloc(opencl_context, W * H * sizeof(FlowState), CL_MEM_READ_WRITE);
		sim.flow_state_buffer_b.alloc(opencl_context, W * H * sizeof(FlowState), CL_MEM_READ_WRITE);
		sim.thermal_erosion_state_buffer.alloc(opencl_context, W * H * sizeof(ThermalErosionState), CL_MEM_READ_WRITE);
		sim.constants_buffer.allocFrom(opencl_context, &constants, sizeof(Constants), CL_MEM_READ_ONLY);

		sim.flowSimulationKernel = new OpenCLKernel(program, "flowSimulationKernel", opencl_device->opencl_device_id, profile);
		sim.thermalErosionFluxKernel = new OpenCLKernel(program, "thermalErosionFluxKernel", opencl_device->opencl_device_id, profile);
		sim.thermalErosionDepositedFluxKernel = new OpenCLKernel(program, "thermalErosionDepositedFluxKernel", opencl_device->opencl_device_id, profile);
		sim.waterAndVelFieldUpdateKernel = new OpenCLKernel(program, "waterAndVelFieldUpdateKernel", opencl_device->opencl_device_id, profile);
		sim.erosionAndDepositionKernel = new OpenCLKernel(program, "erosionAndDepositionKernel", opencl_device->opencl_device_id, profile);
		sim.sedimentTransportationKernel = new OpenCLKernel(program, "sedimentTransportationKernel", opencl_device->opencl_device_id, profile);
		sim.thermalErosionMovementKernel = new OpenCLKernel(program, "thermalErosionMovementKernel", opencl_device->opencl_device_id, profile);
		sim.thermalErosionDepositedMovementKernel = new OpenCLKernel(program, "thermalErosionDepositedMovementKernel", opencl_device->opencl_device_id, profile);
		sim.evaporationKernel = new OpenCLKernel(program, "evaporationKernel", opencl_device->opencl_device_id, profile);
		sim.setHeightFieldMeshKernel = new OpenCLKernel(program, "setHeightFieldMeshKernel", opencl_device->opencl_device_id, profile);
	

		
		// Initialise ImGUI
		ImGui::CreateContext();

		ImGui_ImplSDL2_InitForOpenGL(win, gl_context);
		ImGui_ImplOpenGL3_Init();


		// Create OpenGL engine
		OpenGLEngineSettings settings;
		settings.compress_textures = true;
		settings.shadow_mapping = true;
		settings.depth_fog = true;
		Reference<OpenGLEngine> opengl_engine = new OpenGLEngine(settings);

		TextureServer* texture_server = new TextureServer(/*use_canonical_path_keys=*/false);

		// opengl_data_dir should have 'shaders' and 'gl_data' dirs in it.
		const std::string opengl_data_dir = PlatformUtils::getCurrentWorkingDirPath();
		// const std::string opengl_data_dir = FileUtils::getDirectory(PlatformUtils::getFullPathToCurrentExecutable()); 
		// const std::string opengl_data_dir = PlatformUtils::getEnvironmentVariable("GLARE_CORE_TRUNK_DIR") + "/opengl";

		StandardPrintOutput print_output;
		glare::TaskManager main_task_manager(1);
		glare::TaskManager high_priority_task_manager(1);
		Reference<glare::Allocator> malloc_mem_allocator = new glare::MallocAllocator();
		opengl_engine->initialise(opengl_data_dir, texture_server, &print_output, &main_task_manager, &high_priority_task_manager, malloc_mem_allocator);
		if(!opengl_engine->initSucceeded())
			throw glare::Exception("OpenGL init failed: " + opengl_engine->getInitialisationErrorMsg());
		opengl_engine->setViewportDims(primary_W, primary_H);
		opengl_engine->setMainViewportDims(primary_W, primary_H);

		const std::string base_dir = ".";


		const float sun_phi = 1.f;
		const float sun_theta = Maths::pi<float>() / 4;
		opengl_engine->setSunDir(normalise(Vec4f(std::cos(sun_phi) * sin(sun_theta), std::sin(sun_phi) * sin(sun_theta), cos(sun_theta), 0)));
		opengl_engine->setEnvMapTransform(Matrix3f::rotationMatrix(Vec3f(0,0,1), sun_phi));

		/*
		Set env material
		*/
		{
			OpenGLMaterial env_mat;
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

		// NOTE: OpenGLTexture::Format_SRGBA_Uint8 doesn't seem to work on AMD Gpus (rx480 in particular), gives CL_INVALID_IMAGE_FORMAT_DESCRIPTOR.
		OpenGLTextureRef terrain_col_tex = new OpenGLTexture(W, H, opengl_engine.ptr(), ArrayRef<uint8>(NULL, 0), /*OpenGLTexture::Format_SRGBA_Uint8*/OpenGLTexture::Format_RGBA_Linear_Uint8, OpenGLTexture::Filtering_Bilinear);


		InitialTerrainShape initial_terrain_shape = InitialTerrainShape::InitialTerrainShape_FBM;
		HeightFieldShow cur_heightfield_show = HeightFieldShow::HeightFieldShow_TerrainOnly;
		TextureShow cur_texture_show = TextureShow::TextureShow_DepositedSediment;
		float tex_display_max_val = 1;
		bool display_water = true; // Erosion water
		bool display_sea = true;

		float initial_height_scale = 0.8f;
		float noise_x_scale = 3;
		float noise_y_scale = 3;
		resetTerrain(sim, command_queue, initial_terrain_shape, cell_w, initial_height_scale, noise_x_scale, noise_y_scale, constants.sea_level);

		//TEMP:
		//loadHeightfieldFromDisk(sim, command_queue);


		// Add terrain object
		GLObjectRef terrain_gl_ob = new GLObject();
		terrain_gl_ob->ob_to_world_matrix = Matrix4f::uniformScaleMatrix(1.f);// Matrix4f::uniformScaleMatrix(0.002f);
		terrain_gl_ob->mesh_data = makeTerrainMesh(sim, opengl_engine.ptr(), cell_w, cur_heightfield_show);




		// Add water mesh object
		//GLObjectRef water_gl_ob = new GLObject();
		//water_gl_ob->ob_to_world_matrix = Matrix4f::uniformScaleMatrix(1.f);// Matrix4f::uniformScaleMatrix(0.002f);
		//water_gl_ob->mesh_data = makeTerrainMesh(sim, opengl_engine.ptr(), cell_w, cur_heightfield_show);








		const float large_water_quad_w = 1000000;
		
		// Add water plane
		std::vector<GLObjectRef> sea_water_obs;
		if(true)
		{
			OpenGLMaterial water_mat;
			water_mat.water = true;

			
			Reference<OpenGLMeshRenderData> quad_meshdata = MeshPrimitiveBuilding::makeQuadMesh(*opengl_engine->vert_buf_allocator, Vec4f(1,0,0,0), Vec4f(0,1,0,0), /*res=*/2);
			/*for(int y=0; y<16; ++y)
				for(int x=0; x<16; ++x)
				{
					const int offset_x = x - 8;
					const int offset_y = y - 8;
					if(!(offset_x == 0 && offset_y == 0))
					{
						// Tessellate ground mesh, to avoid texture shimmer due to large quads.
						GLObjectRef gl_ob = new GLObject();
						gl_ob->ob_to_world_matrix = Matrix4f::translationMatrix(0, 0, water_level) * Matrix4f::uniformScaleMatrix(large_water_quad_w) * Matrix4f::translationMatrix(-0.5f + offset_x, -0.5f + offset_y, 0);
						gl_ob->mesh_data = quad_meshdata;

						gl_ob->materials.resize(1);
						gl_ob->materials[0].albedo_linear_rgb = Colour3f(1,0,0);
						gl_ob->materials[0] = water_mat;
						opengl_engine->addObject(gl_ob);
						sea_water_obs.push_back(gl_ob);
					}
				}
				*/
			{
				// Tessellate ground mesh, to avoid texture shimmer due to large quads.
				GLObjectRef gl_ob = new GLObject();
				gl_ob->ob_to_world_matrix = Matrix4f::translationMatrix(0, 0, constants.sea_level) * Matrix4f::uniformScaleMatrix(large_water_quad_w) * Matrix4f::translationMatrix(-0.5f, -0.5f, 0);
				gl_ob->mesh_data = MeshPrimitiveBuilding::makeQuadMesh(*opengl_engine->vert_buf_allocator, Vec4f(1,0,0,0), Vec4f(0,1,0,0), /*res=*/64);

				gl_ob->materials.resize(1);
				//gl_ob->materials[0].albedo_linear_rgb = Colour3f(0,0,1);
				gl_ob->materials[0] = water_mat;
				if(display_sea)
					opengl_engine->addObject(gl_ob);
				sea_water_obs.push_back(gl_ob);
			}

			/*GLObjectRef ground_plane = new GLObject();
			ground_plane->mesh_data = opengl_engine->getUnitQuadMeshData();
			ground_plane->ob_to_world_matrix = Matrix4f::uniformScaleMatrix(10) * Matrix4f::translationMatrix(-0.5f, -0.5f, 0);
			ground_plane->materials.resize(1);
			ground_plane->materials[0].albedo_texture = opengl_engine->getTexture(base_dir + "/resources/obstacle.png");
			ground_plane->materials[0].tex_matrix = Matrix2f(10.f, 0, 0, 10.f);*/

			//opengl_engine->addObject(ground_plane);
		}

		glFinish();

		//----------------------------------------------
		// Get OpenCL buffer for OpenGL terrain texture
		cl_int retcode = 0;
		const cl_mem terrain_tex_cl_mem = getGlobalOpenCL()->clCreateFromGLTexture(opencl_context->getContext(), CL_MEM_WRITE_ONLY, /*texture target=*/GL_TEXTURE_2D, /*miplevel=*/0, /*texture=*/terrain_col_tex->texture_handle, &retcode);
		if(retcode != CL_SUCCESS)
			throw glare::Exception("Failed to create OpenCL buffer for GL terrain texture: " + OpenCL::errorString(retcode));

		sim.terrain_tex_cl_mem = terrain_tex_cl_mem;

		// Get OpenCL buffer for OpenGL terrain mesh vertex buffer
		{
			const GLuint buffer_name = terrain_gl_ob->mesh_data->vbo_handle.vbo->bufferName();
			const cl_mem mesh_vert_buffer_cl_mem = getGlobalOpenCL()->clCreateFromGLBuffer(opencl_context->getContext(), CL_MEM_WRITE_ONLY, buffer_name, &retcode);
			if(retcode != CL_SUCCESS)
				throw glare::Exception("Failed to create OpenCL buffer for GL buffer: " + OpenCL::errorString(retcode));

			sim.heightfield_mesh_buffer = mesh_vert_buffer_cl_mem;
			sim.heightfield_mesh_offset_B = (uint32)terrain_gl_ob->mesh_data->vbo_handle.offset;
		}

		
		// Get OpenCL buffer for OpenGL water mesh vertex buffer
		/*{
			const GLuint buffer_name = water_gl_ob->mesh_data->vbo_handle.vbo->bufferName();
			const cl_mem mesh_vert_buffer_cl_mem = getGlobalOpenCL()->clCreateFromGLBuffer(opencl_context->getContext(), CL_MEM_WRITE_ONLY, buffer_name, &retcode);
			if(retcode != CL_SUCCESS)
				throw glare::Exception("Failed to create OpenCL buffer for GL buffer: " + OpenCL::errorString(retcode));

			sim.water_heightfield_mesh_buffer = mesh_vert_buffer_cl_mem;
			sim.water_heightfield_mesh_offset_B = (uint32)water_gl_ob->mesh_data->vbo_handle.offset;
		}*/
		//----------------------------------------------
		

		//UpdateTexResults results = updateTerrainTexture(sim, terrain_col_tex, cur_texture_show, tex_display_max_val);

		terrain_gl_ob->materials.resize(1);
		terrain_gl_ob->materials[0].albedo_texture = terrain_col_tex;
		terrain_gl_ob->materials[0].fresnel_scale = 0.3f;
		terrain_gl_ob->materials[0].roughness = 0.8f;

		opengl_engine->addObject(terrain_gl_ob);


		//water_gl_ob->materials.resize(1);
		//water_gl_ob->materials[0].water = true;
		//opengl_engine->addObject(water_gl_ob);

		

		Timer timer;
		Timer time_since_mesh_update;

		TerrainStats stats = computeTerrainStats(sim);

		float cam_phi = 0.0;
		float cam_theta = 1.f;
		//Vec4f cam_target_pos = Vec4f(W * cell_w / 2.f, H * cell_w / 2.f, 100, 1);
		Vec4f cam_pos = Vec4f(W * cell_w / 2.f, H * cell_w / 2.f, 500, 1);
		//float cam_dist = 500;
		bool orbit_camera = false;

		bool sim_running = true;
		
		Timer time_since_last_frame;
		Timer stats_timer;
		int stats_last_num_iters = 0;
		double stats_last_iters_per_sec = 0;
		bool reset = false;

		bool quit = false;
		while(!quit)
		{
			//const double cur_time = timer.elapsed();

			if(orbit_camera)
				cam_phi = (float)(timer.elapsed() * 0.1);
			

			if(SDL_GL_MakeCurrent(win, gl_context) != 0)
				conPrint("SDL_GL_MakeCurrent failed.");


			//const Matrix4f T = Matrix4f::translationMatrix(0.f, cam_dist, 0.f);
			const Matrix4f z_rot = Matrix4f::rotationAroundZAxis(cam_phi);
			const Matrix4f x_rot = Matrix4f::rotationAroundXAxis(-(cam_theta - Maths::pi_2<float>()));
			const Matrix4f rot = x_rot * z_rot;
			//const Matrix4f world_to_camera_space_matrix = T * rot * Matrix4f::translationMatrix(-cam_target_pos);

			const Matrix4f world_to_camera_space_matrix = rot * Matrix4f::translationMatrix(-cam_pos);

			const float sensor_width = 0.035f;
			const float lens_sensor_dist = 0.025f;//0.03f;
			const float render_aspect_ratio = opengl_engine->getViewPortAspectRatio();


			int gl_w, gl_h;
			SDL_GL_GetDrawableSize(win, &gl_w, &gl_h);

			opengl_engine->setViewportDims(gl_w, gl_h);
			opengl_engine->setMainViewportDims(gl_w, gl_h);
			opengl_engine->setMaxDrawDistance(1000000.f);
			opengl_engine->setPerspectiveCameraTransform(world_to_camera_space_matrix, sensor_width, lens_sensor_dist, render_aspect_ratio, /*lens shift up=*/0.f, /*lens shift right=*/0.f);
			opengl_engine->setCurrentTime((float)timer.elapsed());
			opengl_engine->draw();


			ImGuiIO& imgui_io = ImGui::GetIO();

			// Draw ImGUI GUI controls
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplSDL2_NewFrame();
			ImGui::NewFrame();

			//ImGui::ShowDemoWindow();

			ImGui::SetNextWindowSize(ImVec2(600, 1100));
			ImGui::Begin("TerrainGen");

			ImGui::TextColored(ImVec4(1,1,0,1), "Simulation parameters");
			bool param_changed = false;
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"delta_t (s)", /*val=*/&constants.delta_t, /*min=*/0.0f, /*max=*/0.3f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"rainfall rate (m/s)", /*val=*/&constants.r, /*min=*/0.0f, /*max=*/0.01f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"friction constant (f)", /*val=*/&constants.f, /*min=*/0.0f, /*max=*/5.f, "%.5f");
			//param_changed = param_changed || ImGui::SliderFloat(/*label=*/"cross-sectional 'pipe' area (m)", /*val=*/&constants.A, /*min=*/0.0f, /*max=*/100.f, "%.5f");
			//param_changed = param_changed || ImGui::SliderFloat(/*label=*/"gravity mag (m/s^2)", /*val=*/&constants.g, /*min=*/0.0f, /*max=*/100.f, "%.5f");
			//param_changed = param_changed || ImGui::SliderFloat(/*label=*/"virtual pipe length (m)", /*val=*/&constants.l, /*min=*/0.0f, /*max=*/100.f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"sediment capacity constant (K_c) ", /*val=*/&constants.K_c, /*min=*/0.0f, /*max=*/4.f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"dissolving constant (K_s) ", /*val=*/&constants.K_s, /*min=*/0.0f, /*max=*/20.f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"deposition constant (K_d) ", /*val=*/&constants.K_d, /*min=*/0.0f, /*max=*/4.f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"erosion depth (K_dmax) ", /*val=*/&constants.K_dmax, /*min=*/0.0f, /*max=*/1.f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"min unit water dischage (q_0) ", /*val=*/&constants.q_0, /*min=*/0.0f, /*max=*/1.f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"evaporation constant (K_e) ", /*val=*/&constants.K_e, /*min=*/0.0f, /*max=*/0.1f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"Thermal erosion constant (K_t) ", /*val=*/&constants.K_t, /*min=*/0.0f, /*max=*/0.5f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"Thermal erosion const, deposited (K_tdep) ", /*val=*/&constants.K_tdep, /*min=*/0.0f, /*max=*/0.5f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"Max talus angle (rad) ", /*val=*/&constants.max_talus_angle, /*min=*/0.0f, /*max=*/1.5f, "%.5f");
			param_changed = param_changed || ImGui::SliderFloat(/*label=*/"Max talus angle, deposited (rad) ", /*val=*/&constants.max_deposited_talus_angle, /*min=*/0.0f, /*max=*/1.5f, "%.5f");
			

			
			
			constants.tan_max_talus_angle = std::tan(constants.max_talus_angle);
			constants.tan_max_deposited_talus_angle = std::tan(constants.max_deposited_talus_angle);

			ImGui::Dummy(ImVec2(100, 40));
			ImGui::TextColored(ImVec4(1,1,0,1), "Visualisation");

			param_changed = param_changed || ImGui::ColorEdit3("rock colour", &constants.rock_col.r);
			param_changed = param_changed || ImGui::ColorEdit3("sediment colour", &constants.sediment_col.r);
			param_changed = param_changed || ImGui::ColorEdit3("vegetaton colour", &constants.vegetation_col.r);

			/*if(ImGui::BeginCombo("heightfield showing", HeightFieldShow_strings[cur_heightfield_show]))
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
			}*/
			param_changed = param_changed || ImGui::Checkbox("Display water", &display_water);
			if(ImGui::Checkbox("Display sea surface", &display_sea))
			{
				if(display_sea)
				{
					for(size_t i=0; i<sea_water_obs.size(); ++i)
						opengl_engine->addObject(sea_water_obs[i]);
				}
				else
				{
					for(size_t i=0; i<sea_water_obs.size(); ++i)
						opengl_engine->removeObject(sea_water_obs[i]);
				}
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
			
			if(ImGui::InputFloat(/*label=*/"Sea level (m)", /*val=*/&constants.sea_level, /*step=*/1.f, /*step fast=*/10.f, "%.3f"))
			{
				// Sea height changed, move water plane
				param_changed = true;

				for(size_t i=0; i<sea_water_obs.size(); ++i)
				{
					sea_water_obs[i]->ob_to_world_matrix = Matrix4f::translationMatrix(0, 0, constants.sea_level) * Matrix4f::uniformScaleMatrix(large_water_quad_w) * Matrix4f::translationMatrix(-0.5f, -0.5f, 0);
					opengl_engine->updateObjectTransformData(*sea_water_obs[i]);
				}
			}

			ImGui::Checkbox("orbit camera", &orbit_camera);

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

			ImGui::InputFloat(/*label=*/"grid cell width", /*val=*/&cell_w, /*step=*/0.1f, /*step fast=*/1.f);
			ImGui::SliderFloat(/*label=*/"Initial height scale", /*val=*/&initial_height_scale, /*min=*/0.0f, /*max=*/10.f, "%.5f");
			ImGui::SliderFloat(/*label=*/"Noise x scale", /*val=*/&noise_x_scale, /*min=*/0.0f, /*max=*/10.f, "%.5f");
			ImGui::SliderFloat(/*label=*/"Noise y scale", /*val=*/&noise_y_scale, /*min=*/0.0f, /*max=*/10.f, "%.5f");

			if(ImGui::Button("Save heightfield to disk"))
			{
				saveHeightfieldToDisk(sim, command_queue);
			}

			if(ImGui::Button("Save colour texture to disk"))
			{
				saveColourTextureToDisk(sim, command_queue, terrain_col_tex);
			}


			ImGui::Dummy(ImVec2(100, 20));
			reset = ImGui::Button("Reset terrain") || reset;

			ImGui::Dummy(ImVec2(100, 40));
			ImGui::TextColored(ImVec4(1,1,0,1), "Info");
			ImGui::Text((std::string(sim_running ? "Sim running" : "Sim paused") + ", iteration: " + toString(sim.sim_iteration)).c_str());
			ImGui::Text(("Speed: " + toString((int)stats_last_iters_per_sec) + " iters/s").c_str());

			
			
			ImGui::Text(("Total terrain volume: " + doubleToStringScientific(stats.total_volume, 4) + "m^3").c_str());
			ImGui::Text(("Total terrain and deposited sed volume: " + doubleToStringScientific(stats.total_volume + stats.total_deposited_sediment_vol, 4) + "m^3").c_str());
			ImGui::Text(("Total water volume: " + doubleToStringMaxNDecimalPlaces(stats.total_water_volume, 4) + "m^3").c_str());
			ImGui::Text(("Total suspended sediment volume: " + doubleToStringMaxNDecimalPlaces(stats.total_suspended_sediment_vol, 4) + "m^3").c_str());
			ImGui::Text(("Total deposited sediment volume: " + doubleToStringMaxNDecimalPlaces(stats.total_deposited_sediment_vol, 4) + "m^3").c_str());
			ImGui::Text(("cam position: " + cam_pos.toStringMaxNDecimalPlaces(1)).c_str());
			//ImGui::Text(("max texture value: " + toString(results.max_value)).c_str());
			ImGui::End(); 

			//if(param_changed || reset)
			{
				constants.draw_water = display_water;
				constants.include_water_height = display_water;
				constants.cell_w = cell_w;
				constants.recip_cell_w = 1.f / cell_w;
				constants.current_time = sim.sim_iteration * constants.delta_t;
				sim.constants_buffer.copyFrom(command_queue, &constants, sizeof(Constants), /*blocking write=*/true);
			}

			if(reset)
			{
				resetTerrain(sim, command_queue, initial_terrain_shape, cell_w, initial_height_scale, noise_x_scale, noise_y_scale, constants.sea_level);
				stats_last_num_iters = 0;
				reset = false;
			}
			

			

			// Update terrain OpenGL mesh and texture from the sim
			//if(time_since_mesh_update.elapsed() > 0.1)
			{
				glFinish();
				sim.updateHeightFieldMeshAndTexture(command_queue); // Do with OpenGL - OpenCL interop

				// Code to do via CPU:
				
				//opengl_engine->removeObject(terrain_gl_ob);
				//terrain_gl_ob = NULL;

				//results = updateTerrainTexture(sim, terrain_col_tex, cur_texture_show, tex_display_max_val);

				//terrain_gl_ob = new GLObject();
				//terrain_gl_ob->ob_to_world_matrix = Matrix4f::translationMatrix(0, 0, 0.01f) * Matrix4f::uniformScaleMatrix(0.002f);
				//terrain_gl_ob->mesh_data = makeTerrainMesh(sim, opengl_engine.ptr(), cur_heightfield_show);

				//terrain_gl_ob->materials.resize(1);
				//terrain_gl_ob->materials[0].albedo_texture = terrain_col_tex;

				//opengl_engine->addObject(terrain_gl_ob);
				
				time_since_mesh_update.reset();
			}


			if(stats_timer.elapsed() > 5.0)
			{
				// Update statistics
				stats = computeTerrainStats(sim);
				
				stats_last_iters_per_sec = (sim.sim_iteration - stats_last_num_iters) / stats_timer.elapsed();

				stats_timer.reset();
				stats_last_num_iters = sim.sim_iteration;
			}

			
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

			// Display
			SDL_GL_SwapWindow(win);

			const float dt = (float)time_since_last_frame.elapsed();
			time_since_last_frame.reset();

			const Vec4f forwards = GeometrySampling::dirForSphericalCoords(-cam_phi + Maths::pi_2<float>(), Maths::pi<float>() - cam_theta);
			const Vec4f right = normalise(crossProduct(forwards, Vec4f(0,0,1,0)));
			const Vec4f up = crossProduct(right, forwards);

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
						
						opengl_engine->setViewportDims(w, h);
						opengl_engine->setMainViewportDims(w, h);
					}
				}
				else if(e.type == SDL_KEYDOWN)
				{
					if(e.key.keysym.sym == SDLK_r)
						reset = true;
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

						//const float move_scale = 1.f;
						//cam_target_pos += right * -(float)e.motion.xrel * move_scale + up * (float)e.motion.yrel * move_scale;
					}
				}
				else if(e.type == SDL_MOUSEWHEEL)
				{
					//conPrint("SDL_MOUSEWHEEL");
					//cam_dist = myClamp<float>(cam_dist - cam_dist * e.wheel.y * 0.2f, 0.01f, 10000.f);
					const float move_speed = 30.f * cell_w;
					cam_pos += forwards * (float)e.wheel.y * move_speed;
				}
			}

			SDL_PumpEvents();
			const uint8* keystate = SDL_GetKeyboardState(NULL);
			const float shift_factor = (keystate[SDL_SCANCODE_LSHIFT] != 0) ? 3.f : 1.f;
			if(keystate[SDL_SCANCODE_LEFT])
				cam_phi -= dt * 0.25f * shift_factor;
			if(keystate[SDL_SCANCODE_RIGHT])
				cam_phi += dt * 0.25f * shift_factor;

			const float move_speed = 140.f * cell_w * shift_factor;
			if(keystate[SDL_SCANCODE_W])
				cam_pos += forwards * dt * move_speed;
			if(keystate[SDL_SCANCODE_S])
				cam_pos -= forwards * dt * move_speed;
			if(keystate[SDL_SCANCODE_A])
				cam_pos -= right * dt * move_speed;
			if(keystate[SDL_SCANCODE_D])
				cam_pos += right * dt * move_speed;
		}
		SDL_Quit();
		return 0;
	}
	catch(glare::Exception& e)
	{
		stdErrPrint(e.what());
		SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Error", e.what().c_str(), /*parent=*/NULL);
		return 1;
	}
}
