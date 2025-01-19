/*=====================================================================
terraingen.cpp
--------------
Copyright Nicholas Chapman 2025 -
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
#include <utils/FileDialogs.h>
#include <utils/XMLWriteUtils.h>
#include <utils/XMLParseUtils.h>
#include <utils/IndigoXMLDoc.h>
#include <GL/gl3w.h>
#include <SDL_opengl.h>
#include <SDL.h>
#include <imgui.h>
#include <backends/imgui_impl_opengl3.h>
#include <backends/imgui_impl_sdl2.h>
#include <fstream>
#include <string>
#ifdef _WIN32
#include <Objbase.h>
#endif
#include <ArgumentParser.h>


typedef struct
{
	float height; // terrain height ('b') (m)
	//float water; // water height (depth) above terrain ('d') (m)
	float suspended_vol; // Volume of suspended sediment. ('s') (m^3)
	float deposited_sed_h; // Height of deposited sediment (m)

	float water_mass;  // water_mass = water_depth * cell_w^2 * water_density
	Vec2f water_vel; // average velocity of water in cell
	
	float new_water_mass;
	float new_suspended_vol;
	Vec2f new_water_vel;

	Vec2f water_vel_laplacian;
	Vec2f duv_dx;
	Vec2f duv_dy;

	Vec2f thermal_vel; // horizontal velocity of thermally eroded solid
	float thermal_move_vol; // volume of thermally eroded solid
	float height_laplacian;

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
	int W; // grid width
	int H; // grid height
	float cell_w; // Width of cell = spacing between grid cells (metres)
	float recip_cell_w; // 1 / cell_w

	float delta_t; // time step
	float r; // rainfall rate
	float A; // cross-sectional 'pipe' area
	float g; // gravity accel magnitude. positive.
	float l; // virtual pipe length
	float f; // friction constant
	float k; // viscous drag coefficient (see https://en.wikipedia.org/wiki/Shallow_water_equations)
	float nu; // kinematic viscosity
	
	float K_c;// = 0.01; // 1; // sediment capacity constant
	float K_s;// = 0.01; // 0.5; // dissolving constant.
	float K_d;// = 0.01; // 1; // deposition constant
	float K_dmax;// = 0.1f; // Maximum erosion depth: water depth at which erosion stops.
	float q_0; // Minimum unit water discharge for sediment carrying.
	float K_e; // Evaporation constant

	float K_smooth; // smoothing constant
	float laplacian_threshold;
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

	int debug_draw_channel; // From TextureShow enum
	float debug_display_max_val;

	float water_z_bias;
} Constants;



enum InitialTerrainShape
{
	InitialTerrainShape_ConstantSlope,
	InitialTerrainShape_Hat,
	InitialTerrainShape_Cone,
	InitialTerrainShape_FBM,
	InitialTerrainShape_Perlin
};

const char* InitialTerrainShape_display_strings[] = 
{
	"constant slope",
	"hat",
	"cone",
	"FBM",
	"Perlin noise"
};

// For writing to xml
const char* InitialTerrainShape_storage_strings[] = 
{
	"constant_slope",
	"hat",
	"cone",
	"FBM",
	"Perlin"
};



struct TerrainParams
{
	InitialTerrainShape terrain_shape;
	float height_scale;
	float fine_roughness_vert_scale;
	float x_scale;
	float y_scale;
	float initial_water_depth;
};



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
	cl_mem water_heightfield_mesh_buffer;
	uint32 water_heightfield_mesh_offset_B;
	cl_mem terrain_tex_cl_mem;

	bool use_water_mesh;

	Timer timer;

	Simulation(int W_, int H_, OpenCLContextRef opencl_context, OpenCLProgramRef program, OpenCLDeviceRef opencl_device, bool profile, const Constants& constants)
	{
		W = W_;
		H = H_;

		sim_iteration = 0;

		terrain_state.resize(W, H);
		flow_state.resize(W, H);

		use_water_mesh = true;

		terrain_state_buffer.alloc(opencl_context, /*size=*/constants.W * constants.H * sizeof(TerrainState), CL_MEM_READ_WRITE);
		flow_state_buffer_a.alloc(opencl_context, constants.W * constants.H * sizeof(FlowState), CL_MEM_READ_WRITE);
		flow_state_buffer_b.alloc(opencl_context, constants.W * constants.H * sizeof(FlowState), CL_MEM_READ_WRITE);
		thermal_erosion_state_buffer.alloc(opencl_context, constants.W * constants.H * sizeof(ThermalErosionState), CL_MEM_READ_WRITE);
		constants_buffer.allocFrom(opencl_context, &constants, sizeof(Constants), CL_MEM_READ_ONLY);

		flowSimulationKernel = new OpenCLKernel(program, "flowSimulationKernel", opencl_device->opencl_device_id, profile);
		thermalErosionFluxKernel = new OpenCLKernel(program, "thermalErosionFluxKernel", opencl_device->opencl_device_id, profile);
		thermalErosionDepositedFluxKernel = new OpenCLKernel(program, "thermalErosionDepositedFluxKernel", opencl_device->opencl_device_id, profile);
		waterAndVelFieldUpdateKernel = new OpenCLKernel(program, "waterAndVelFieldUpdateKernel", opencl_device->opencl_device_id, profile);
		erosionAndDepositionKernel = new OpenCLKernel(program, "erosionAndDepositionKernel", opencl_device->opencl_device_id, profile);
		sedimentTransportationKernel = new OpenCLKernel(program, "sedimentTransportationKernel", opencl_device->opencl_device_id, profile);
		thermalErosionMovementKernel = new OpenCLKernel(program, "thermalErosionMovementKernel", opencl_device->opencl_device_id, profile);
		thermalErosionDepositedMovementKernel = new OpenCLKernel(program, "thermalErosionDepositedMovementKernel", opencl_device->opencl_device_id, profile);
		evaporationKernel = new OpenCLKernel(program, "evaporationKernel", opencl_device->opencl_device_id, profile);
		setHeightFieldMeshKernel = new OpenCLKernel(program, "setHeightFieldMeshKernel", opencl_device->opencl_device_id, profile);
	}


	void readBackToCPUMem(OpenCLCommandQueueRef command_queue)
	{
		// Read back terrain state buffer to CPU mem
		terrain_state_buffer.readTo(command_queue, /*dest ptr=*/&terrain_state.elem(0, 0), /*size=*/W * H * sizeof(TerrainState), /*blocking read=*/true);

		// TEMP: just for debugging (showing/reading flux)
		//flow_state_buffer_a.readTo(command_queue, /*dest ptr=*/&flow_state.elem(0, 0), /*size=*/W * H * sizeof(FlowState), /*blocking read=*/true);
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

			//waterAndVelFieldUpdateKernel->setKernelArgBuffer(0, *cur_flow_state_buffer);
			waterAndVelFieldUpdateKernel->setKernelArgBuffer(0, terrain_state_buffer);
			waterAndVelFieldUpdateKernel->setKernelArgBuffer(1, constants_buffer);
			waterAndVelFieldUpdateKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);
			
		
			// NEW: transports both water and sediment
			sedimentTransportationKernel->setKernelArgBuffer(0, terrain_state_buffer);
			sedimentTransportationKernel->setKernelArgBuffer(1, constants_buffer);
			sedimentTransportationKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);

			// NEW: updates height, suspended_vol, deposited_sed_h, also assigns new_water_mass -> water_mass etc.
			erosionAndDepositionKernel->setKernelArgBuffer(0, terrain_state_buffer);
			erosionAndDepositionKernel->setKernelArgBuffer(1, constants_buffer);
			erosionAndDepositionKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);


			for(int deposited_sed=0; deposited_sed<2; ++deposited_sed)
			{
				thermalErosionFluxKernel->setKernelArgBuffer(0, terrain_state_buffer);
				thermalErosionFluxKernel->setKernelArgBuffer(1, thermal_erosion_state_buffer); // source
				thermalErosionFluxKernel->setKernelArgBuffer(2, constants_buffer);
				thermalErosionFluxKernel->setKernelArgInt(3, deposited_sed); // erosion of deposited sediment?
				thermalErosionFluxKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);

				thermalErosionMovementKernel->setKernelArgBuffer(0, thermal_erosion_state_buffer);
				thermalErosionMovementKernel->setKernelArgBuffer(1, terrain_state_buffer);
				thermalErosionMovementKernel->setKernelArgBuffer(2, constants_buffer);
				thermalErosionMovementKernel->setKernelArgInt(3, deposited_sed); // erosion of deposited sediment?
				thermalErosionMovementKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);
			}

			//thermalErosionFluxKernel->setKernelArgBuffer(0, terrain_state_buffer);
			//thermalErosionFluxKernel->setKernelArgBuffer(1, thermal_erosion_state_buffer); // source
			//thermalErosionFluxKernel->setKernelArgBuffer(2, constants_buffer);
			//thermalErosionFluxKernel->setKernelArgInt(3, 1); // erosion of deposited sediment?
			//thermalErosionFluxKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);

			//thermalErosionMovementKernel->setKernelArgBuffer(0, thermal_erosion_state_buffer);
			//thermalErosionMovementKernel->setKernelArgBuffer(1, terrain_state_buffer);
			//thermalErosionMovementKernel->setKernelArgBuffer(2, constants_buffer);
			//thermalErosionMovementKernel->setKernelArgInt(3, 1); // erosion of deposited sediment?
			//thermalErosionMovementKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);

			//thermalErosionDepositedFluxKernel->setKernelArgBuffer(0, terrain_state_buffer);
			//thermalErosionDepositedFluxKernel->setKernelArgBuffer(1, thermal_erosion_state_buffer); // source
			//thermalErosionDepositedFluxKernel->setKernelArgBuffer(2, constants_buffer);
			//thermalErosionDepositedFluxKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);

			//thermalErosionDepositedMovementKernel->setKernelArgBuffer(0, thermal_erosion_state_buffer);
			//thermalErosionDepositedMovementKernel->setKernelArgBuffer(1, terrain_state_buffer);
			//thermalErosionDepositedMovementKernel->setKernelArgBuffer(2, constants_buffer);
			//thermalErosionDepositedMovementKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);

			evaporationKernel->setKernelArgBuffer(0, terrain_state_buffer);
			evaporationKernel->setKernelArgBuffer(1, constants_buffer);
			evaporationKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);
		}

		assert(cur_flow_state_buffer == &flow_state_buffer_a);

		sim_iteration += num_iters;
	}

	void updateHeightFieldMeshAndTexture(OpenCLCommandQueueRef command_queue)
	{
		SmallVector<cl_mem, 8> mem_objects;
		mem_objects.push_back(heightfield_mesh_buffer);
		if(use_water_mesh)
			mem_objects.push_back(water_heightfield_mesh_buffer);
		mem_objects.push_back(terrain_tex_cl_mem);
		::getGlobalOpenCL()->clEnqueueAcquireGLObjects(command_queue->getCommandQueue(),
			/*num objects=*/(cl_uint)mem_objects.size(), /*mem objects=*/mem_objects.data(), /*num objects in wait list=*/0, /*event wait list=*/NULL, /*event=*/NULL);
		
		setHeightFieldMeshKernel->setKernelArgBuffer(0, terrain_state_buffer);
		setHeightFieldMeshKernel->setKernelArgBuffer(1, constants_buffer);
		setHeightFieldMeshKernel->setKernelArgBuffer(2, heightfield_mesh_buffer);
		setHeightFieldMeshKernel->setKernelArgUInt(3, heightfield_mesh_offset_B);
		setHeightFieldMeshKernel->setKernelArgBuffer(4, terrain_tex_cl_mem);
		if(use_water_mesh)
		{
			setHeightFieldMeshKernel->setKernelArgBuffer(5, water_heightfield_mesh_buffer);
			setHeightFieldMeshKernel->setKernelArgUInt(6, water_heightfield_mesh_offset_B);
		}

		
		setHeightFieldMeshKernel->launchKernel2D(command_queue->getCommandQueue(), W, H);

		::getGlobalOpenCL()->clEnqueueReleaseGLObjects(command_queue->getCommandQueue(),
			/*num objects=*/(cl_uint)mem_objects.size(), /*mem objects=*/mem_objects.data(), /*num objects in wait list=*/0, /*event wait list=*/NULL, /*event=*/NULL);

		command_queue->finish(); // Make sure we have finished writing to the vertex buffer and texture before we start issuing new OpenGL commands.
		// TODO: work out a better way of syncing (e.g. with barrier).
	}
};


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


// Should match TextureShow_Default etc. defines in erosion_kernel.cl
const char* TextureShow_strings[] = 
{
	"none",
	"water speed",
	"water depth",
	"suspended sediment vol",
	"deposited sediment height"
};


inline float totalTerrainHeight(const TerrainState& state)
{
	return state.height + state.deposited_sed_h;//state.sediment[0] + state.sediment[1] + state.sediment[2];
}


OpenGLMeshRenderDataRef makeTerrainMesh(const Simulation& sim, OpenGLEngine* opengl_engine, float cell_w) 
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
	float total_uneroded_volume;
	float total_water_mass;
	float total_suspended_sediment_vol;
	float total_deposited_sediment_vol;

	float total_solid_volume; // Uneroded volume + deposited volume + suspended volume
};

TerrainStats computeTerrainStats(Simulation& sim, const Constants& constants)
{
	double sum_uneroded_terrain_h = 0;
	double sum_water_mass = 0;
	double sum_suspended_vol = 0;
	double sum_deposited_sediment_h = 0;
	for(int y=0; y<sim.H; ++y)
	for(int x=0; x<sim.W; ++x)
	{
		sum_uneroded_terrain_h += sim.terrain_state.elem(x, y).height;
		sum_water_mass   += sim.terrain_state.elem(x, y).water_mass;
		sum_suspended_vol += sim.terrain_state.elem(x, y).suspended_vol;
		sum_deposited_sediment_h  += sim.terrain_state.elem(x, y).deposited_sed_h;
	}
	const float cell_A = constants.cell_w * constants.cell_w;
	TerrainStats stats;
	stats.total_uneroded_volume = (float)sum_uneroded_terrain_h * cell_A;
	stats.total_water_mass = (float)sum_water_mass;
	stats.total_suspended_sediment_vol = (float)sum_suspended_vol;
	stats.total_deposited_sediment_vol = (float)sum_deposited_sediment_h * cell_A;
	
	stats.total_solid_volume = stats.total_uneroded_volume + stats.total_deposited_sediment_vol + stats.total_suspended_sediment_vol;

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


struct NotificationInfo
{
	std::string notification;
	Timer notification_start_display_timer;
};


void showNotification(NotificationInfo& info, const std::string& new_notification)
{
	info.notification_start_display_timer.reset();
	info.notification = new_notification;
}


float waterMassForHeight(float water_height, float cell_w)
{
	float water_density = 1000.f;
	return water_height * (cell_w * cell_w * water_density);
}


void resetTerrain(Simulation& sim, OpenCLCommandQueueRef command_queue, const TerrainParams& terrain_params, float cell_w, float sea_level)
{
	const int W = (int)sim.terrain_state.getWidth();
	const int H = (int)sim.terrain_state.getHeight();

	sim.sim_iteration = 0;

	const float initial_water_mass = waterMassForHeight(terrain_params.initial_water_depth, cell_w);

	// Set initial state
	TerrainState f;
	f.height = 1.f;
	f.suspended_vol = 0.f;
	f.deposited_sed_h = 0.f;
	f.water_mass = initial_water_mass;
	f.water_vel = Vec2f(0.f);

	sim.terrain_state.setAllElems(f);

	FlowState flow_state;
	flow_state.f_L = flow_state.f_R = flow_state.f_T = flow_state.f_B = 0;
	flow_state.sed_f_L = flow_state.sed_f_R = flow_state.sed_f_T = flow_state.sed_f_B = 0;

	sim.flow_state.setAllElems(flow_state);

	const float total_vert_scale = cell_w * terrain_params.height_scale;
	const float fine_rough_xy_scale = 8.f;

	if(terrain_params.terrain_shape == InitialTerrainShape::InitialTerrainShape_ConstantSlope)
	{
		for(int x=0; x<W; ++x)
		for(int y=0; y<H; ++y)
		{
			float nx = (float)x / W;
			float ny = (float)y / H;

			sim.terrain_state.elem(x, y).height = nx * W / 2.0f * total_vert_scale + 
				(terrain_params.fine_roughness_vert_scale > 0.f ? PerlinNoise::FBM(nx * fine_rough_xy_scale, ny * fine_rough_xy_scale, 12) * terrain_params.fine_roughness_vert_scale : 0.f);
			
		}
	}
	else if(terrain_params.terrain_shape == InitialTerrainShape::InitialTerrainShape_Hat)
	{
		for(int x=0; x<W; ++x)
		for(int y=0; y<H; ++y)
		{
			float nx = (float)x / W;
			float ny = (float)y / H;

			float dx = nx - 0.5f;
			const float tent = myMax(0.f, (1.f - fabs(dx) / terrain_params.x_scale));
			sim.terrain_state.elem(x, y).height = tent * W / 2.0f * total_vert_scale + 
				(terrain_params.fine_roughness_vert_scale > 0.f ? PerlinNoise::FBM(nx * fine_rough_xy_scale, ny * fine_rough_xy_scale, 12) * terrain_params.fine_roughness_vert_scale : 0.f);
		}
	}
	else if(terrain_params.terrain_shape == InitialTerrainShape::InitialTerrainShape_Cone)
	{
		for(int x=0; x<W; ++x)
		for(int y=0; y<H; ++y)
		{
			float nx = (float)x / W;
			float ny = (float)y / H;

			float dx = nx - 0.5f;
			float dy = ny - 0.5f;
			dx /= terrain_params.x_scale;
			dy /= terrain_params.y_scale;
			const float r = sqrt(dx*dx + dy*dy);
			
			const float cone = myMax(0.0f, 1 - r*4);
			sim.terrain_state.elem(x, y).height = cone * W / 4.0f * total_vert_scale + 
				(terrain_params.fine_roughness_vert_scale > 0.f ? PerlinNoise::FBM(nx * fine_rough_xy_scale, ny * fine_rough_xy_scale, 12) * terrain_params.fine_roughness_vert_scale : 0.f);
		}
	}
	else if(terrain_params.terrain_shape == InitialTerrainShape::InitialTerrainShape_FBM)
	{
		for(int x=0; x<W; ++x)
		for(int y=0; y<H; ++y)
		{
			float nx = (float)x / W;
			float ny = (float)y / H;

			//sim.terrain_state.elem(x, y).height = (-nx*nx + nx) * 200.0f;
			//const float r = Vec2f((float)x, (float)y).getDist(Vec2f((float)W/2, (float)H/2));

			//sim.terrain_state.elem(x, y).height = r < (W/4.0) ? 100.f : 0.f;

			const float perlin_factor = PerlinNoise::FBM(nx * terrain_params.x_scale, ny * terrain_params.y_scale, 12);
			//const float perlin_factor = PerlinNoise::ridgedFBM<float>(Vec4f(nx * 0.5f, ny * 0.5f, 0, 1), 1, 2, 5);
			sim.terrain_state.elem(x, y).height = myMax(-100.f, perlin_factor)/3.f * W / 5.0f * total_vert_scale;// * myMax(1 - (1.1f * r / ((float)W/2)), 0.f) * 200.f;
			//sim.terrain_state.elem(x, y).height = nx < 0.25 ? 0 : (nx < 0.5 ? (nx - 0.25) : (1.0f - (nx-0.25)) * 200.f;
			//const float tent = (nx < 0.5) ? nx : (1.0f - nx);
			//sim.terrain_state.elem(x, y).height = myMax(0.0f, tent*2 - 0.5f) * 200.f;
		}
	}
	else if(terrain_params.terrain_shape == InitialTerrainShape::InitialTerrainShape_Perlin)
	{
		for(int x=0; x<W; ++x)
		for(int y=0; y<H; ++y)
		{
			float nx = (float)x / W;
			float ny = (float)y / H;

			const float perlin_factor = PerlinNoise::noise(nx * terrain_params.x_scale, ny * terrain_params.y_scale) + 1.f;
			sim.terrain_state.elem(x, y).height = perlin_factor * W / 5.0f * total_vert_scale + 
				(terrain_params.fine_roughness_vert_scale > 0.f ? PerlinNoise::FBM(nx * fine_rough_xy_scale, ny * fine_rough_xy_scale, 12) * terrain_params.fine_roughness_vert_scale : 0.f);
		}
	}

	// Set water height based on sea level
	for(int y=0; y<H; ++y)
	for(int x=0; x<W; ++x)
	{
		const float total_terrain_h = sim.terrain_state.elem(x, y).height + sim.terrain_state.elem(x, y).deposited_sed_h;
		if(total_terrain_h < sea_level)
		{
			//TEMP sim.terrain_state.elem(x, y).water = sea_level - total_terrain_h;
		}
	}

	// Upload to GPU
	sim.terrain_state_buffer.copyFrom(command_queue, /*src ptr=*/&sim.terrain_state.elem(0, 0), /*size=*/W * H * sizeof(TerrainState), /*blocking_write=*/true);
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


void saveHeightfieldToDisk(Simulation& sim, OpenCLCommandQueueRef command_queue, NotificationInfo& info)
{
	try
	{
		FileDialogs::Options options;
		options.dialog_title = "Save Heightfield";
		options.file_types.push_back(FileDialogs::FileTypeInfo("EXR", "*.exr", "exr"));
		options.file_types.push_back(FileDialogs::FileTypeInfo("PNG", "*.png", "png"));
		const std::string path = FileDialogs::showSaveFileDialog(options);
		if(!path.empty())
		{
			sim.readBackToCPUMem(command_queue);

			// Save heightfield
			//saveStructMemberToEXR(sim, path, offsetof(TerrainState, height));

			// Save a map of deposited sediment
			//saveStructMemberToEXR(sim, "sediment_map.exr", offsetof(TerrainState, deposited_sed_h));

			//saveStructMemberToEXR(sim, "water.exr", offsetof(TerrainState, water));

			if(hasExtension(path, "exr"))
			{
				std::vector<float> data(sim.W * sim.H);
				for(int y=0; y<sim.H; ++y)
				for(int x=0; x<sim.W; ++x)
					data[x + y * sim.W] = sim.terrain_state.elem(x, sim.H - y - 1).height + sim.terrain_state.elem(x, sim.H - y - 1).deposited_sed_h; // flip upside down (so that y=0 is at the top as EXR and PNG expects)

				EXRDecoder::SaveOptions exr_options;
				EXRDecoder::saveImageToEXR(data.data(), sim.W, sim.H, /*num channels=*/1, /*save alpha channel=*/false, /*"heightfield_with_deposited_sed.exr"*/path, /*layer name=*/"", exr_options);
			}
			else
			{
				ImageMapUInt16 map(sim.W, sim.H, /*N=*/1);
				map.setAsNotIndependentlyHeapAllocated();

				// Compute min and max terrain heights, map terrain heights to png values such that min terrain height = 0 and max terrain height = 2^16 - 1.
				// We need to do something like this since PNG files can't contain negative values.
				float min_val = 1.0e10f;
				float max_val = 1.0e-10f;
				for(int y=0; y<sim.H; ++y)
				for(int x=0; x<sim.W; ++x)
				{
					const float val = sim.terrain_state.elem(x, y).height + sim.terrain_state.elem(x, y).deposited_sed_h;
					min_val = myMin(min_val, val);
					max_val = myMax(max_val, val);
				}

				for(int y=0; y<sim.H; ++y)
				for(int x=0; x<sim.W; ++x)
				{
					const float val = sim.terrain_state.elem(x, sim.H - y - 1).height + sim.terrain_state.elem(x, sim.H - y - 1).deposited_sed_h; // flip upside down (so that y=0 is at the top as EXR and PNG expects)
					map.getPixel(x, y)[0] = (uint16)((val - min_val) / (max_val - min_val) * 65535.9f);
				}

				PNGDecoder::write(map, path);
			}

			showNotification(info, "Saved image to '" + path + "'.");
		}
	}
	catch(glare::Exception& e)
	{
		SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Failed to save", ("Failed to save heightfield to disk: " + e.what()).c_str(), /*parent=*/NULL);
	}
}


void saveColourTextureToDisk(Simulation& sim, OpenCLCommandQueueRef command_queue, OpenGLTextureRef terrain_col_tex, NotificationInfo& info)
{
	command_queue->finish();
	glFinish();

	try
	{
		FileDialogs::Options options;
		options.dialog_title = "Save Heightfield";
		options.file_types.push_back(FileDialogs::FileTypeInfo("PNG", "*.png", "png"));
		const std::string path = FileDialogs::showSaveFileDialog(options);
		if(!path.empty())
		{
			ImageMapUInt8 map(sim.W, sim.H, 4);
			map.setAsNotIndependentlyHeapAllocated();

			terrain_col_tex->readBackTexture(/*mipmap level=*/0, ArrayRef<uint8>(map.getData(), map.getDataSize()));

			// Convert to 3-component map and flip upside down (so that y=0 is at the top as PNG expects)
			ImageMapUInt8Ref rgb_map = new ImageMapUInt8(sim.W, sim.H, 3); // map.extract3ChannelImage();
			for(int y=0; y<sim.H; ++y)
			for(int x=0; x<sim.W; ++x)
			{
				for(int c=0; c<3; ++c)
					rgb_map->getPixel(x, y)[c] = map.getPixel(x, sim.H - y - 1)[c];
			}
			PNGDecoder::write(*rgb_map, path);

			showNotification(info, "Saved image to '" + path + "'.");
		}
	}
	catch(glare::Exception& e)
	{
		SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Failed to save", ("Failed to save heightfield to disk: " + e.what()).c_str(), /*parent=*/NULL);
	}
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

	loadStructMemberFromEXR(sim, "sediment_map.exr", offsetof(TerrainState, deposited_sed_h));

	//TEMP loadStructMemberFromEXR(sim, "water.exr", offsetof(TerrainState, water));

	// Upload to GPU
	sim.terrain_state_buffer.copyFrom(command_queue, /*src ptr=*/&sim.terrain_state.elem(0, 0), /*size=*/sim.W * sim.H * sizeof(TerrainState), CL_MEM_READ_WRITE);

	conPrint("done.");
}


#define WRITE_FLOAT_PARAM(param)   XMLWriteUtils::writeFloatToXML(xml, #param, constants.param, tab_depth);


void saveParametersToFile(const Constants& constants, const TerrainParams& terrain_params, const std::string& path)
{
	std::string xml = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>\n";
	
	xml += "<sim_parameters>\n";
	
	const int tab_depth = 1;

	XMLWriteUtils::writeInt32ToXML(xml, "version", 1, tab_depth);

	XMLWriteUtils::writeInt32ToXML(xml, "W", constants.W, tab_depth);
	XMLWriteUtils::writeInt32ToXML(xml, "H", constants.H, tab_depth);
	WRITE_FLOAT_PARAM(cell_w);

	WRITE_FLOAT_PARAM(delta_t);
	WRITE_FLOAT_PARAM(r);
	WRITE_FLOAT_PARAM(A);
	WRITE_FLOAT_PARAM(g);
	WRITE_FLOAT_PARAM(l);
	WRITE_FLOAT_PARAM(f);
	WRITE_FLOAT_PARAM(k);
	WRITE_FLOAT_PARAM(nu);

	WRITE_FLOAT_PARAM(K_c);
	WRITE_FLOAT_PARAM(K_s);
	WRITE_FLOAT_PARAM(K_d);
	WRITE_FLOAT_PARAM(K_dmax);
	WRITE_FLOAT_PARAM(q_0);
	WRITE_FLOAT_PARAM(K_e);

	WRITE_FLOAT_PARAM(K_smooth);
	WRITE_FLOAT_PARAM(laplacian_threshold);
	WRITE_FLOAT_PARAM(K_t);
	WRITE_FLOAT_PARAM(K_tdep);
	WRITE_FLOAT_PARAM(max_talus_angle);
	WRITE_FLOAT_PARAM(max_deposited_talus_angle);
	WRITE_FLOAT_PARAM(sea_level);

	XMLWriteUtils::writeInt32ToXML(xml, "include_water_height", constants.include_water_height, tab_depth);
	XMLWriteUtils::writeInt32ToXML(xml, "draw_water", constants.draw_water, tab_depth);
	XMLWriteUtils::writeColour3fToXML(xml, "rock_col", constants.rock_col, tab_depth);
	XMLWriteUtils::writeColour3fToXML(xml, "sediment_col", constants.sediment_col, tab_depth);
	XMLWriteUtils::writeColour3fToXML(xml, "vegetation_col", constants.vegetation_col, tab_depth);

	// Write terrain params
	xml += "\t<!-- terrain -->\n";
	XMLWriteUtils::writeStringElemToXML(xml, "terrain_shape", InitialTerrainShape_storage_strings[terrain_params.terrain_shape], tab_depth);
	XMLWriteUtils::writeFloatToXML(xml, "height_scale", terrain_params.height_scale, tab_depth);
	XMLWriteUtils::writeFloatToXML(xml, "fine_roughness_vert_scale", terrain_params.fine_roughness_vert_scale, tab_depth);
	XMLWriteUtils::writeFloatToXML(xml, "x_scale", terrain_params.x_scale, tab_depth);
	XMLWriteUtils::writeFloatToXML(xml, "y_scale", terrain_params.y_scale, tab_depth);
	XMLWriteUtils::writeFloatToXML(xml, "initial_water_depth", terrain_params.initial_water_depth, tab_depth);

	xml += "</sim_parameters>\n";

	FileUtils::writeEntireFileTextMode(path, xml);
}


#define PARSE_FLOAT_PARAM(param)   constants.param = XMLParseUtils::parseFloatWithDefault(sim_params_node, #param, /*default val=*/constants.param);

void loadParametersFromFile(const std::string& path, Constants& constants, TerrainParams& terrain_params)
{
	std::string xml = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>\n";
	
	xml += "<sim_parameters>\n";

	IndigoXMLDoc doc(path);

	pugi::xml_node sim_params_node = doc.getRootElement();

	constants.W = XMLParseUtils::parseIntWithDefault(sim_params_node, "W", /*default=*/1024);
	constants.H = XMLParseUtils::parseIntWithDefault(sim_params_node, "H", /*default=*/1024);
	PARSE_FLOAT_PARAM(cell_w);

	PARSE_FLOAT_PARAM(delta_t);
	PARSE_FLOAT_PARAM(r);
	PARSE_FLOAT_PARAM(A);
	PARSE_FLOAT_PARAM(g);
	PARSE_FLOAT_PARAM(l);
	PARSE_FLOAT_PARAM(f);
	PARSE_FLOAT_PARAM(k);
	PARSE_FLOAT_PARAM(nu);

	PARSE_FLOAT_PARAM(K_c);
	PARSE_FLOAT_PARAM(K_s);
	PARSE_FLOAT_PARAM(K_d);
	PARSE_FLOAT_PARAM(K_dmax);
	PARSE_FLOAT_PARAM(q_0);
	PARSE_FLOAT_PARAM(K_e);

	PARSE_FLOAT_PARAM(K_smooth);
	PARSE_FLOAT_PARAM(laplacian_threshold);
	PARSE_FLOAT_PARAM(K_t);
	PARSE_FLOAT_PARAM(K_tdep);
	PARSE_FLOAT_PARAM(max_talus_angle);
	PARSE_FLOAT_PARAM(max_deposited_talus_angle);
	PARSE_FLOAT_PARAM(sea_level);

	constants.include_water_height = XMLParseUtils::parseIntWithDefault(sim_params_node, "include_water_height", constants.include_water_height);
	constants.draw_water = XMLParseUtils::parseIntWithDefault(sim_params_node, "draw_water", constants.draw_water);
	constants.rock_col = XMLParseUtils::parseColour3fWithDefault(sim_params_node, "rock_col", constants.rock_col);
	constants.rock_col = XMLParseUtils::parseColour3fWithDefault(sim_params_node, "rock_col", constants.rock_col);
	constants.sediment_col = XMLParseUtils::parseColour3fWithDefault(sim_params_node, "sediment_col", constants.sediment_col);
	constants.vegetation_col = XMLParseUtils::parseColour3fWithDefault(sim_params_node, "vegetation_col", constants.vegetation_col);

	// Read terrain params
	const std::string terrain_shape = XMLParseUtils::parseStringWithDefault(sim_params_node, "terrain_shape", InitialTerrainShape_storage_strings[0]);
	for(size_t i=0; i<staticArrayNumElems(InitialTerrainShape_storage_strings); ++i)
		if(terrain_shape == InitialTerrainShape_storage_strings[i])
			terrain_params.terrain_shape = (InitialTerrainShape)i;

	terrain_params.height_scale = XMLParseUtils::parseFloatWithDefault(sim_params_node, "height_scale", terrain_params.height_scale);
	terrain_params.fine_roughness_vert_scale = XMLParseUtils::parseFloatWithDefault(sim_params_node, "fine_roughness_vert_scale", /*default val=*/0.f);
	terrain_params.x_scale = XMLParseUtils::parseFloatWithDefault(sim_params_node, "x_scale", terrain_params.x_scale);
	terrain_params.y_scale = XMLParseUtils::parseFloatWithDefault(sim_params_node, "y_scale", terrain_params.y_scale);
	terrain_params.initial_water_depth = XMLParseUtils::parseFloatWithDefault(sim_params_node, "initial_water_depth", /*default val=*/0.f);
}


void shareOpenGLBuffersWithOpenCLSim(Simulation* sim, OpenCLContextRef opencl_context, OpenGLTextureRef terrain_col_tex, OpenGLMeshRenderDataRef terrain_gl_ob_mesh_data, OpenGLMeshRenderDataRef water_gl_ob_mesh_data)
{
	cl_int retcode = 0;

	// Get OpenCL buffer for OpenGL terrain texture
	{
		const cl_mem terrain_tex_cl_mem = getGlobalOpenCL()->clCreateFromGLTexture(opencl_context->getContext(), CL_MEM_WRITE_ONLY, /*texture target=*/GL_TEXTURE_2D, /*miplevel=*/0, /*texture=*/terrain_col_tex->texture_handle, &retcode);
		if(retcode != CL_SUCCESS)
			throw glare::Exception("Failed to create OpenCL buffer for GL terrain texture: " + OpenCL::errorString(retcode));

		sim->terrain_tex_cl_mem = terrain_tex_cl_mem;
	}

	// Get OpenCL buffer for OpenGL terrain mesh vertex buffer
	{
		const GLuint buffer_name = terrain_gl_ob_mesh_data->vbo_handle.vbo->bufferName();
		const cl_mem mesh_vert_buffer_cl_mem = getGlobalOpenCL()->clCreateFromGLBuffer(opencl_context->getContext(), CL_MEM_WRITE_ONLY, buffer_name, &retcode);
		if(retcode != CL_SUCCESS)
			throw glare::Exception("Failed to create OpenCL buffer for GL buffer: " + OpenCL::errorString(retcode));

		sim->heightfield_mesh_buffer = mesh_vert_buffer_cl_mem;
		sim->heightfield_mesh_offset_B = (uint32)terrain_gl_ob_mesh_data->vbo_handle.offset;
	}

		
	// Get OpenCL buffer for OpenGL water mesh vertex buffer
	if(water_gl_ob_mesh_data)
	{
		const GLuint buffer_name = water_gl_ob_mesh_data->vbo_handle.vbo->bufferName();
		const cl_mem mesh_vert_buffer_cl_mem = getGlobalOpenCL()->clCreateFromGLBuffer(opencl_context->getContext(), CL_MEM_WRITE_ONLY, buffer_name, &retcode);
		if(retcode != CL_SUCCESS)
			throw glare::Exception("Failed to create OpenCL buffer for GL buffer: " + OpenCL::errorString(retcode));

		sim->water_heightfield_mesh_buffer = mesh_vert_buffer_cl_mem;
		sim->water_heightfield_mesh_offset_B = (uint32)water_gl_ob_mesh_data->vbo_handle.offset;
	}
}


void unshareOpenGLBuffersFromOpenCLSim(Simulation* sim)
{
	cl_int result = getGlobalOpenCL()->clReleaseMemObject(sim->terrain_tex_cl_mem);
	if(result != CL_SUCCESS)
		throw glare::Exception("clReleaseMemObject failed: " + OpenCL::errorString(result));

	result = getGlobalOpenCL()->clReleaseMemObject(sim->heightfield_mesh_buffer);
	if(result != CL_SUCCESS)
		throw glare::Exception("clReleaseMemObject failed: " + OpenCL::errorString(result));

	result = getGlobalOpenCL()->clReleaseMemObject(sim->water_heightfield_mesh_buffer);
	if(result != CL_SUCCESS)
		throw glare::Exception("clReleaseMemObject failed: " + OpenCL::errorString(result));
}


void createAndAddTerrainTextureAndMeshes(Constants& constants, Simulation* sim, Reference<OpenGLEngine> opengl_engine, OpenCLContextRef opencl_context, bool use_water_mesh, OpenGLTextureRef& terrain_col_tex, GLObjectRef& terrain_gl_ob, GLObjectRef& water_gl_ob)
{
	// Create terrain colour texture.
	terrain_col_tex = new OpenGLTexture(constants.W, constants.H, opengl_engine.ptr(), ArrayRef<uint8>(NULL, 0), /*OpenGLTexture::Format_SRGBA_Uint8*/OpenGLTextureFormat::Format_RGBA_Linear_Uint8, OpenGLTexture::Filtering_Bilinear);
		
	// Add terrain mesh object
	terrain_gl_ob = new GLObject();
	terrain_gl_ob->ob_to_world_matrix = Matrix4f::uniformScaleMatrix(1.f);
	terrain_gl_ob->mesh_data = makeTerrainMesh(*sim, opengl_engine.ptr(), constants.cell_w);
	terrain_gl_ob->materials.resize(1);
	terrain_gl_ob->materials[0].albedo_texture = terrain_col_tex;
	terrain_gl_ob->materials[0].fresnel_scale = 0.3f;
	terrain_gl_ob->materials[0].roughness = 0.8f;
	terrain_gl_ob->materials[0].convert_albedo_from_srgb = true; // Need this as the texture uses a linear colour space, but we want to treat it as non-linear sRGB.
				
	// Add water mesh object
	if(use_water_mesh)
	{
		water_gl_ob = new GLObject();
		water_gl_ob->ob_to_world_matrix = Matrix4f::uniformScaleMatrix(1.f);
		water_gl_ob->mesh_data = makeTerrainMesh(*sim, opengl_engine.ptr(), constants.cell_w);
		water_gl_ob->materials.resize(1);
		water_gl_ob->materials[0].water = true;
	}

	glFinish();

	shareOpenGLBuffersWithOpenCLSim(sim, opencl_context, terrain_col_tex, terrain_gl_ob->mesh_data, use_water_mesh ? water_gl_ob->mesh_data : OpenGLMeshRenderDataRef());

	opengl_engine->addObject(terrain_gl_ob);
	opengl_engine->addObject(water_gl_ob);
}


int main(int argc, char** argv)
{
	Clock::init();

#ifdef _WIN32
	// Init COM (used by file open dialog etc.)
	HRESULT res = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
	if(FAILED(res))
	{
		conPrint("Failed to init COM");
		return 1;
	}
#endif

	try
	{
		std::vector<std::string> args(argc);
		for(int i=0; i<argc; ++i)
			args[i] = std::string(argv[i]);

		std::map<std::string, std::vector<ArgumentParser::ArgumentType> > syntax;
		syntax["--params"] = std::vector<ArgumentParser::ArgumentType>(1, ArgumentParser::ArgumentType_string); // One string arg

		ArgumentParser arg_parser(args, syntax, /*allow unnamed arg=*/false);



		//=========================== Init SDL and OpenGL ================================
		if(SDL_Init(SDL_INIT_VIDEO) != 0)
			throw glare::Exception("SDL_Init Error: " + std::string(SDL_GetError()));


		// Set GL attributes, needs to be done before window creation.
		setGLAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4); // We need to request a specific version for a core profile.
		setGLAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
		setGLAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

		setGLAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);


		// NOTE: OpenGL init needs to go before OpenCL init
		const int primary_window_W = 1920;
		const int primary_window_H = 1080;

		SDL_Window* win = SDL_CreateWindow("TerrainGen", 100, 100, primary_window_W, primary_window_H, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
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
			const std::string src = FileUtils::readEntireFile(base_src_dir + "/erosion_kernel.cl");

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
		constants.W = 1024;
		constants.H = 1024;
		constants.cell_w = 8.f;
		constants.recip_cell_w = 1.f / constants.cell_w;
		constants.delta_t = 0.08f; // time step
		constants.r = 0.0025f; // 0.012f; // rainfall rate
		constants.A = 1; // cross-sectional 'pipe' area
		constants.g = 9.81f; // gravity accel.  NOTE: should be negative?
		constants.l = 1.0; // l = pipe length
		constants.f = 1.f; // 0.05f; // friction constant
		constants.k = 0.001f; // viscous drag coefficient
		constants.nu = 0.001f; // kinematic viscosity

		constants.K_c = 0.5f; // sediment capacity constant
		constants.K_s = 3; // 0.5f; // dissolving constant.
		constants.K_d = 0.5f; // deposition constant
		constants.K_dmax = 1.f;
		constants.q_0 = 0.2f;
		constants.K_e = 0.001f; // 0.005f; // Evaporation constant
		constants.K_smooth = 0.f; // Smoothing constant
		constants.laplacian_threshold = 0.f;
		constants.K_t = 0.f; // TEMP 0.03f; // Thermal erosion constant
		constants.K_tdep = 0.f; // TEMP 0.03f; // thermal erosion constant for deposited sediment
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
		constants.water_z_bias = -0.1f;

		constants.debug_draw_channel = 0;
		constants.debug_display_max_val = 1.f;

		TerrainParams terrain_params;
		terrain_params.terrain_shape = InitialTerrainShape::InitialTerrainShape_FBM;
		terrain_params.height_scale = 0.8f;
		terrain_params.fine_roughness_vert_scale = 0.01f;
		terrain_params.x_scale = 3;
		terrain_params.y_scale = 3;
		terrain_params.initial_water_depth = 0;

		//HeightFieldShow cur_heightfield_show = HeightFieldShow::HeightFieldShow_TerrainOnly;
		//TextureShow cur_texture_show = TextureShow::TextureShow_DepositedSediment;
		
		//float tex_display_max_val = 1;
		bool display_water = true; // Show flowing water?
		bool display_sea = false; // Display a flat sea surface around the terrain?
		const bool use_water_mesh = true; // Show water as a mesh instead of just as a colour tint on the terrain?

		if(arg_parser.isArgPresent("--params"))
		{
			const std::string param_path = arg_parser.getArgStringValue("--params");
			loadParametersFromFile(param_path, constants, terrain_params);
		}


		Simulation* sim = new Simulation(constants.W, constants.H, opencl_context, program, opencl_device, profile, constants);
		sim->use_water_mesh = use_water_mesh;

		resetTerrain(*sim, command_queue, terrain_params, constants.cell_w, constants.sea_level);
		

		// Initialise ImGUI
		ImGui::CreateContext();

		ImGui_ImplSDL2_InitForOpenGL(win, gl_context);
		ImGui_ImplOpenGL3_Init();


		// Create OpenGL engine
		OpenGLEngineSettings settings;
		settings.compress_textures = true;
		settings.shadow_mapping = true;
		settings.depth_fog = true;
		settings.render_sun_and_clouds = false;
		settings.render_water_caustics = false;
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
		opengl_engine->setViewportDims(primary_window_W, primary_window_H);
		opengl_engine->setMainViewportDims(primary_window_W, primary_window_H);

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

		OpenGLTextureRef terrain_col_tex;
		GLObjectRef terrain_gl_ob;
		GLObjectRef water_gl_ob;

		createAndAddTerrainTextureAndMeshes(constants, sim, opengl_engine, opencl_context, use_water_mesh, terrain_col_tex, terrain_gl_ob, water_gl_ob);

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

		//glFinish();

		//shareOpenGLBuffersWithOpenCLSim(sim, opencl_context, terrain_col_tex, terrain_gl_ob->mesh_data, use_water_mesh ? water_gl_ob->mesh_data : OpenGLMeshRenderDataRef());

		//opengl_engine->addObject(terrain_gl_ob);
		//opengl_engine->addObject(water_gl_ob);

		Timer timer;
		Timer time_since_mesh_update;

		TerrainStats stats = computeTerrainStats(*sim, constants);

		float cam_phi = 0.0;
		float cam_theta = 2.1f; // Maths::pi_2<float>();//1.f;
		//Vec4f cam_target_pos = Vec4f(W * cell_w / 2.f, H * cell_w / 2.f, 100, 1);

		const float terrain_w = constants.W * constants.cell_w;
		const float terrain_h = constants.H * constants.cell_w;
		Vec4f cam_pos;
		{
		const Vec4f terrain_centre = Vec4f(terrain_w / 2.f, terrain_h / 2.f, sim->terrain_state.elem(constants.W/2, constants.H/2).height, 1);
		const Vec4f cam_forwards = GeometrySampling::dirForSphericalCoords(cam_phi + Maths::pi_2<float>(), cam_theta); // cam_phi = 0   =>  cam forwards is (0,1,0).
		cam_pos = terrain_centre - cam_forwards * (terrain_h * 1.f);// - Vec4f(cos(cam_phi), sin(cam_phi), 
		}
		//float cam_dist = 500;
		bool orbit_camera = false;
		float orbit_angular_vel = 0.1f;
		float orbit_dist = terrain_w;

		bool sim_running = true;
		
		Timer time_since_last_frame;
		Timer stats_timer;
		int stats_last_num_iters = 0;
		double stats_last_iters_per_sec = 0;
		bool reset = false;

		
		NotificationInfo notification_info;

		// Keep these around, are applied by 'apply' button.
		int new_W = constants.W;
		int new_H = constants.H;
		float new_cell_w = constants.cell_w;

		bool quit = false;
		while(!quit)
		{
			//const double cur_time = timer.elapsed();

			if(orbit_camera)
			{
				cam_phi = (float)(timer.elapsed() * orbit_angular_vel);
				const Vec4f terrain_centre = Vec4f(terrain_w / 2.f, terrain_h / 2.f, sim->terrain_state.elem(constants.W/2, constants.H/2).height, 1);
				const Vec4f cam_forwards = GeometrySampling::dirForSphericalCoords(cam_phi + Maths::pi_2<float>(), cam_theta); // cam_phi = 0   =>  cam forwards is (0,1,0).
				cam_pos = terrain_centre - cam_forwards * orbit_dist;
			}

				
			

			if(SDL_GL_MakeCurrent(win, gl_context) != 0)
				conPrint("SDL_GL_MakeCurrent failed.");


			//const Matrix4f T = Matrix4f::translationMatrix(0.f, cam_dist, 0.f);
			/*const Matrix4f z_rot = Matrix4f::rotationAroundZAxis(cam_phi);
			const Matrix4f x_rot = Matrix4f::rotationAroundXAxis(-(cam_theta - Maths::pi_2<float>()));*/
			const Matrix4f z_rot = Matrix4f::rotationAroundZAxis(-cam_phi);
			const Matrix4f x_rot = Matrix4f::rotationAroundXAxis(cam_theta - Maths::pi_2<float>());
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


			// Draw ImGUI GUI controls
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplSDL2_NewFrame();
			ImGui::NewFrame();

			//ImGui::ShowDemoWindow();

			
			const float spacing_vert_pixels = 15;

			ImGui::SetNextWindowSize(ImVec2(600, 1200));
			ImGui::Begin("TerrainGen");

			//-------------------------------------- Render Simulation parameters section --------------------------------------
			ImGui::TextColored(ImVec4(1,1,0,1), "Simulation parameters");
			ImGui::BeginGroup(); // Begin Simulation parameters group

			ImGui::Text("Grid");
			ImGui::InputInt(/*label=*/"grid x res", /*val=*/&new_W, /*step=*/1, /*step fast=*/32);
			ImGui::InputInt(/*label=*/"grid Y res", /*val=*/&new_H, /*step=*/1, /*step fast=*/32);
			ImGui::SliderFloat(/*label=*/"cell width (s)", /*val=*/&new_cell_w, /*min=*/0.0001f, /*max=*/100.f, "%.5f");
			bool grid_changed = ImGui::Button("Apply");

			ImGui::SliderFloat(/*label=*/"delta_t (s)", /*val=*/&constants.delta_t, /*min=*/0.0f, /*max=*/0.3f, "%.5f");

			ImGui::Text("Water");
			ImGui::SliderFloat(/*label=*/"rainfall rate (m/s)", /*val=*/&constants.r, /*min=*/0.0f, /*max=*/0.01f, "%.5f");
			ImGui::SliderFloat(/*label=*/"evaporation constant (K_e) ", /*val=*/&constants.K_e, /*min=*/0.0f, /*max=*/0.1f, "%.5f");
			ImGui::SliderFloat(/*label=*/"friction constant (f)", /*val=*/&constants.f, /*min=*/0.0f, /*max=*/5.f, "%.5f");
			ImGui::SliderFloat(/*label=*/"viscous drag coeff (k)", /*val=*/&constants.k, /*min=*/0.0f, /*max=*/1.f, "%.5f");
			ImGui::SliderFloat(/*label=*/"kinematic viscosity (nu)", /*val=*/&constants.nu, /*min=*/0.0f, /*max=*/1000.f, "%.5f");
			//param_changed = param_changed || ImGui::SliderFloat(/*label=*/"cross-sectional 'pipe' area (m)", /*val=*/&constants.A, /*min=*/0.0f, /*max=*/100.f, "%.5f");
			//param_changed = param_changed || ImGui::SliderFloat(/*label=*/"gravity mag (m/s^2)", /*val=*/&constants.g, /*min=*/0.0f, /*max=*/100.f, "%.5f");
			//param_changed = param_changed || ImGui::SliderFloat(/*label=*/"virtual pipe length (m)", /*val=*/&constants.l, /*min=*/0.0f, /*max=*/100.f, "%.5f");

			ImGui::Text("Sediment");
			ImGui::SliderFloat(/*label=*/"sediment capacity constant (K_c) ", /*val=*/&constants.K_c, /*min=*/0.0f, /*max=*/4.f, "%.5f");
			ImGui::SliderFloat(/*label=*/"dissolving constant (K_s) ", /*val=*/&constants.K_s, /*min=*/0.0f, /*max=*/20.f, "%.5f");
			ImGui::SliderFloat(/*label=*/"deposition constant (K_d) ", /*val=*/&constants.K_d, /*min=*/0.0f, /*max=*/4.f, "%.5f");
			ImGui::SliderFloat(/*label=*/"erosion depth (K_dmax) ", /*val=*/&constants.K_dmax, /*min=*/0.0f, /*max=*/1.f, "%.5f");
			ImGui::SliderFloat(/*label=*/"min unit water dischage (q_0) ", /*val=*/&constants.q_0, /*min=*/0.0f, /*max=*/1.f, "%.5f");
			
			ImGui::Text("Smoothing");
			ImGui::SliderFloat(/*label=*/"Smoothing constant (K_smooth)", /*val=*/&constants.K_smooth,            /*min=*/0.0f, /*max=*/10.f, "%.5f");
			ImGui::SliderFloat(/*label=*/"Smoothing laplacian threshold", /*val=*/&constants.laplacian_threshold, /*min=*/0.0f, /*max=*/1.f, "%.5f");

			ImGui::Text("Thermal erosion");
			ImGui::SliderFloat(/*label=*/"Thermal erosion constant (K_t)",             /*val=*/&constants.K_t,    /*min=*/0.0f, /*max=*/100.f, "%.5f");
			ImGui::SliderFloat(/*label=*/"Thermal erosion const, deposited (K_tdep) ", /*val=*/&constants.K_tdep, /*min=*/0.0f, /*max=*/100.f, "%.5f");
			ImGui::SliderFloat(/*label=*/"Max talus angle (rad)",            /*val=*/&constants.max_talus_angle,           /*min=*/0.0f, /*max=*/1.5f, "%.5f");
			ImGui::SliderFloat(/*label=*/"Max talus angle, deposited (rad)", /*val=*/&constants.max_deposited_talus_angle, /*min=*/0.0f, /*max=*/1.5f, "%.5f");
			
			//ImGui::Dummy(ImVec2(30, 20));
			ImGui::Spacing();
			ImGui::Text("Terrain initialisation");
			if(ImGui::BeginCombo("heightfield shape", InitialTerrainShape_display_strings[terrain_params.terrain_shape]))
			{
				for(int i=0; i<staticArrayNumElems(InitialTerrainShape_display_strings); ++i)
				{
					const bool selected = terrain_params.terrain_shape == i;
					if(ImGui::Selectable(InitialTerrainShape_display_strings[i], selected))
						terrain_params.terrain_shape = (InitialTerrainShape)i;
					if(selected)
						ImGui::SetItemDefaultFocus();
				}

				ImGui::EndCombo();
			}

			ImGui::InputFloat(/*label=*/"grid cell width", /*val=*/&constants.cell_w, /*step=*/0.1f, /*step fast=*/1.f);
			ImGui::SliderFloat(/*label=*/"height scale", /*val=*/&terrain_params.height_scale, /*min=*/0.0f, /*max=*/10.f, "%.5f");
			ImGui::SliderFloat(/*label=*/"fine roughness height scale", /*val=*/&terrain_params.fine_roughness_vert_scale, /*min=*/0.0f, /*max=*/100.f, "%.5f");
			ImGui::SliderFloat(/*label=*/"x scale", /*val=*/&terrain_params.x_scale, /*min=*/0.0f, /*max=*/10.f, "%.5f");
			ImGui::SliderFloat(/*label=*/"y scale", /*val=*/&terrain_params.y_scale, /*min=*/0.0f, /*max=*/10.f, "%.5f");
			ImGui::SliderFloat(/*label=*/"initial water depth", /*val=*/&terrain_params.initial_water_depth, /*min=*/0.0f, /*max=*/10.f, "%.5f");

			ImGui::Spacing();

			if(ImGui::Button("save parameters"))
			{
				try
				{
					FileDialogs::Options options;
					options.dialog_title = "Save parameters";
					options.file_types.push_back(FileDialogs::FileTypeInfo("Parameter file", "*.tgparams", "tgparams"));
					const std::string path = FileDialogs::showSaveFileDialog(options);
					if(!path.empty())
						 saveParametersToFile(constants, terrain_params, path); // Save
				}
				catch(glare::Exception& e)
				{
					conPrint("Failed to choose path or to save parameters to path: " + e.what());
				}
			}
			ImGui::SameLine(); // Put buttons on same lin
			if(ImGui::Button("load parameters"))
			{
				try
				{
					FileDialogs::Options options;
					options.dialog_title = "Load parameters";
					options.file_types.push_back(FileDialogs::FileTypeInfo("Parameter file", "*.tgparams", "tgparams"));
					const std::string path = FileDialogs::showOpenFileDialog(options);
					if(!path.empty())
					{
						const Constants old_constants = constants;
						loadParametersFromFile(path, constants, terrain_params); // Load

						new_W = constants.W;
						new_H = constants.H;
						new_cell_w = constants.cell_w;

						grid_changed = constants.W != old_constants.W || constants.H != old_constants.H || constants.cell_w != old_constants.cell_w;
						reset = true;
					}
				}
				catch(glare::Exception& e)
				{
					conPrint("Failed to choose path or to load parameters from path: " + e.what());
				}
			}

			ImGui::EndGroup(); // End Simulation parameters group

			//-------------------------------------- Render Visualisation section --------------------------------------
			ImGui::Dummy(ImVec2(60, spacing_vert_pixels));
			ImGui::TextColored(ImVec4(1,1,0,1), "Visualisation");

			ImGui::ColorEdit3("rock colour", &constants.rock_col.r);
			ImGui::ColorEdit3("sediment colour", &constants.sediment_col.r);
			ImGui::ColorEdit3("vegetation colour", &constants.vegetation_col.r);

			//debug_draw_channel
			
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
			if(ImGui::Checkbox("Display water", &display_water))
			{
				if(use_water_mesh)
				{
					if(display_water)
						opengl_engine->addObject(water_gl_ob);
					else
						opengl_engine->removeObject(water_gl_ob);
				}
			}
			if(display_water)
				ImGui::SliderFloat(/*label=*/"water height bias", /*val=*/&constants.water_z_bias, /*min=*/-1.f, /*max=*/0.f, "%.5f");

			
			
			
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
			if(display_sea)
			{
				if(ImGui::InputFloat(/*label=*/"Sea level (m)", /*val=*/&constants.sea_level, /*step=*/1.f, /*step fast=*/10.f, "%.3f"))
				{
					// Sea height changed, move water plane
					for(size_t i=0; i<sea_water_obs.size(); ++i)
					{
						sea_water_obs[i]->ob_to_world_matrix = Matrix4f::translationMatrix(0, 0, constants.sea_level) * Matrix4f::uniformScaleMatrix(large_water_quad_w) * Matrix4f::translationMatrix(-0.5f, -0.5f, 0);
						if(display_sea)
							opengl_engine->updateObjectTransformData(*sea_water_obs[i]);
					}
				}
			}

			ImGui::Checkbox("orbit camera", &orbit_camera);
			if(orbit_camera)
			{
				ImGui::SliderFloat(/*label=*/"orbit speed", /*val=*/&orbit_angular_vel, /*min=*/0.f, /*max=*/1.f, "%.3f");
				ImGui::SliderFloat(/*label=*/"orbit dist", /*val=*/&orbit_dist, /*min=*/0.f, /*max=*/myMax(terrain_w, terrain_h) * 3.f, "%.3f");
			}


			ImGui::Text("Debug visualisation");
			if(ImGui::BeginCombo("texture showing", TextureShow_strings[constants.debug_draw_channel]))
			{
				for(int i=0; i<staticArrayNumElems(TextureShow_strings); ++i)
				{
					const bool selected = constants.debug_draw_channel == i;
					if(ImGui::Selectable(TextureShow_strings[i], selected))
						constants.debug_draw_channel = i;
					if(selected)
						ImGui::SetItemDefaultFocus();
				}

				ImGui::EndCombo();
			}

			ImGui::DragFloat(/*label=*/"Texture display max val", /*val=*/&constants.debug_display_max_val, /*value speed=*/0.1f, /*min val=*/0.f, /*max val=*/10000.f);


			ImGui::Dummy(ImVec2(60, spacing_vert_pixels));
			ImGui::TextColored(ImVec4(1,1,0,1), "Simulation control");
			
			// Upload constants
			{
				constants.draw_water = display_water;
				constants.include_water_height = use_water_mesh ? false : display_water;
				constants.cell_w = constants.cell_w;
				constants.recip_cell_w = 1.f / constants.cell_w;
				constants.tan_max_talus_angle = std::tan(constants.max_talus_angle);
				constants.tan_max_deposited_talus_angle = std::tan(constants.max_deposited_talus_angle);
				constants.current_time = sim->sim_iteration * constants.delta_t;

				sim->constants_buffer.copyFrom(command_queue, /*src ptr=*/&constants, sizeof(Constants), /*blocking write=*/false);
			}

			bool do_advance = false;
			if(sim_running)
			{
				do_advance = true;
				if(ImGui::Button("pause"))
				{
					sim_running = false;
					showNotification(notification_info, "Paused");
				}
			}
			else // Else if paused:
			{
				if(ImGui::Button("resume"))
				{
					sim_running = true;
					showNotification(notification_info, "Resumed");
				}

				ImGui::SameLine(); // Put buttons on same line
				const bool single_step = ImGui::Button("single step");
				if(single_step)
					do_advance = true;
			}
			
			// Advance simulation (if not paused)
			if(do_advance)
				sim->doSimIteration(command_queue);

			ImGui::SameLine(); // Put buttons on same line
			reset = ImGui::Button("Reset") || reset;


			ImGui::Dummy(ImVec2(60, spacing_vert_pixels));
			if(ImGui::Button("Save heightfield to disk"))
			{
				saveHeightfieldToDisk(*sim, command_queue, notification_info);
			}

			if(ImGui::Button("Save colour texture to disk"))
			{
				saveColourTextureToDisk(*sim, command_queue, terrain_col_tex, notification_info);
			}

			//-------------------------------------- Render Info section --------------------------------------
			ImGui::Dummy(ImVec2(60, spacing_vert_pixels));
			ImGui::TextColored(ImVec4(1,1,0,1), "Info");
			ImGui::Text((std::string(sim_running ? "Sim running" : "Sim paused") + ", iteration: " + toString(sim->sim_iteration)).c_str());
			ImGui::Text(("Speed: " + toString((int)stats_last_iters_per_sec) + " iters/s").c_str());

			ImGui::Text(("Total water mass: " + doubleToStringMaxNDecimalPlaces(stats.total_water_mass, 4) + " kg").c_str());
			ImGui::Text(("Total uneroded terrain volume: " + doubleToStringScientific(stats.total_uneroded_volume, 4) + " m^3").c_str());
			ImGui::Text(("Total suspended sediment volume: " + doubleToStringMaxNDecimalPlaces(stats.total_suspended_sediment_vol, 4) + " m^3").c_str());
			ImGui::Text(("Total deposited sediment volume: " + doubleToStringMaxNDecimalPlaces(stats.total_deposited_sediment_vol, 4) + " m^3").c_str());
			ImGui::Text(("Total solid volume: " + doubleToStringScientific(stats.total_solid_volume, 4) + " m^3").c_str());
			ImGui::Text(("cam position: " + cam_pos.toStringMaxNDecimalPlaces(1)).c_str());
			//ImGui::Text(("cam theta, phi: " + doubleToStringMaxNDecimalPlaces(cam_theta, 2) + ", " + doubleToStringMaxNDecimalPlaces(cam_phi, 2)).c_str());
			//ImGui::Text(("max texture value: " + toString(results.max_value)).c_str());
			ImGui::End(); 



			// Show notification widget
			if(!notification_info.notification.empty() && notification_info.notification_start_display_timer.elapsed() < 3.0)
			{
				ImVec2 tex_dims = ImGui::CalcTextSize(notification_info.notification.c_str());

				const int notification_window_w = tex_dims.x + 20;
				ImGui::SetNextWindowSize(ImVec2(notification_window_w, tex_dims.y + 10));
				ImGui::SetNextWindowPos(ImVec2(gl_w/2 - notification_window_w/2, 25));
				ImGui::Begin("Notification", NULL, ImGuiWindowFlags_NoDecoration);
				ImGui::Text(notification_info.notification.c_str());
				ImGui::End();
			}


			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			
			// Update terrain OpenGL mesh and texture from the sim
			//if(time_since_mesh_update.elapsed() > 0.1)
			{
				glFinish();
				sim->updateHeightFieldMeshAndTexture(command_queue); // Do with OpenGL - OpenCL interop

				//time_since_mesh_update.reset();
			}


			if(stats_timer.elapsed() > 1.0)
			{
				// Update statistics
				sim->readBackToCPUMem(command_queue);

				stats = computeTerrainStats(*sim, constants);
				
				stats_last_iters_per_sec = (sim->sim_iteration - stats_last_num_iters) / stats_timer.elapsed();

				stats_timer.reset();
				stats_last_num_iters = sim->sim_iteration;
			}


			// Display
			SDL_GL_SwapWindow(win);

			if(grid_changed)
			{
				glFinish();

				unshareOpenGLBuffersFromOpenCLSim(sim);

				delete sim;

				// Remove old meshes from OpenGL engine
				//terrain_gl_ob->materials[0].albedo_texture = NULL;
				terrain_col_tex = NULL;

				opengl_engine->removeObject(terrain_gl_ob);
				terrain_gl_ob = NULL;

				if(water_gl_ob)
				{
					opengl_engine->removeObject(water_gl_ob);
					water_gl_ob = NULL;
				}

				constants.W = new_W;
				constants.H = new_H;
				constants.cell_w = new_cell_w;

				sim = new Simulation(constants.W, constants.H, opencl_context, program, opencl_device, profile, constants);

				resetTerrain(*sim, command_queue, terrain_params, constants.cell_w, constants.sea_level);

				createAndAddTerrainTextureAndMeshes(constants, sim, opengl_engine, opencl_context, use_water_mesh, terrain_col_tex, terrain_gl_ob, water_gl_ob);
			}
			else if(reset)
			{
				resetTerrain(*sim, command_queue, terrain_params, constants.cell_w, constants.sea_level);
				stats_last_num_iters = 0;
				reset = false;
			}

			const float dt = (float)time_since_last_frame.elapsed();
			time_since_last_frame.reset();

			const Vec4f forwards = GeometrySampling::dirForSphericalCoords(cam_phi + Maths::pi_2<float>(), cam_theta); // cam_phi = 0   =>  cam forwards is (0,1,0).
			const Vec4f right = normalise(crossProduct(forwards, Vec4f(0,0,1,0)));
			const Vec4f up = crossProduct(right, forwards);

			// Handle any events
			SDL_Event e;
			while(SDL_PollEvent(&e))
			{
				if(ImGui::GetIO().WantCaptureMouse)
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
					else if(e.key.keysym.sym == SDLK_PAUSE)
					{
						sim_running = !sim_running;
						showNotification(notification_info, sim_running ? "Resumed" : "Paused");
					}
				}
				else if(e.type == SDL_MOUSEMOTION)
				{
					//conPrint("SDL_MOUSEMOTION");
					if(e.motion.state & SDL_BUTTON_LMASK)
					{
						//conPrint("SDL_BUTTON_LMASK down");

						const float move_scale = 0.005f;
						cam_phi -= e.motion.xrel * move_scale;
						cam_theta = myClamp<float>(cam_theta + (float)e.motion.yrel * move_scale, 0.01f, Maths::pi<float>() - 0.01f);
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
					const float move_speed = 30.f * constants.cell_w;
					cam_pos += forwards * (float)e.wheel.y * move_speed;
				}
			}

			SDL_PumpEvents();
			const uint8* keystate = SDL_GetKeyboardState(NULL);
			const float shift_factor = (keystate[SDL_SCANCODE_LSHIFT] != 0) ? 3.f : 1.f;
			if(keystate[SDL_SCANCODE_LEFT])
				cam_phi += dt * 0.25f * shift_factor;
			if(keystate[SDL_SCANCODE_RIGHT])
				cam_phi -= dt * 0.25f * shift_factor;

			const float move_speed = 140.f * constants.cell_w * shift_factor;
			if(keystate[SDL_SCANCODE_W])
				cam_pos += forwards * dt * move_speed;
			if(keystate[SDL_SCANCODE_S])
				cam_pos -= forwards * dt * move_speed;
			if(keystate[SDL_SCANCODE_A])
				cam_pos -= right * dt * move_speed;
			if(keystate[SDL_SCANCODE_D])
				cam_pos += right * dt * move_speed;
			if(keystate[SDL_SCANCODE_SPACE])
				cam_pos += up * dt * move_speed;
			if(keystate[SDL_SCANCODE_C])
				cam_pos -= up * dt * move_speed;
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
