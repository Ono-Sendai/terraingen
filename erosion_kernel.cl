/*=====================================================================
erosion_kernel.cl
-----------------
Copyright Nicholas Chapman 2025 -
=====================================================================*/

// See "Fast Hydraulic Erosion Simulation and Visualization on GPU"
// Also
// "Fast Hydraulic and Thermal Erosion on the GPU"
// http://www.cescg.org/CESCG-2011/papers/TUBudapest-Jako-Balazs.pdf


#define DO_SEMILAGRANGIAN_ADVECTION 0

#define TextureShow_Default						0
#define TextureShow_WaterSpeed					1
#define TextureShow_WaterDepth					2
#define TextureShow_SuspendedSedimentVol		3
#define TextureShow_DepositedSedimentH			4


inline float square(float x)
{
	return x*x;
}


typedef struct
{
	float height; // height of uneroded terrain ('b') (m) (deposited sediement sits on top of this) 
	//float water; // water height (depth) above terrain ('d') (m)
	float suspended_vol; // Volume of suspended sediment. ('s') (m^3)
	float deposited_sed_h; // Height of deposited sediment (m)

	//float u, v; // currently storing water flux (m^3/s) in x and y directions. OLD: velocity
	//float water_vel;
	//float sed_flux;

	float water_mass;  // water_mass = water_depth * cell_w^2 * water_density,            water_depth = water_mass / (cell_w^2 * water_density)
	//float2 water_position; // average position of water in cell, in pixel coordinates
	float2 water_vel; // average velocity of water in cell

	float new_water_mass;
	float new_suspended_vol;
	float2 new_water_vel;

	float2 water_vel_laplacian;
	float2 duv_dx;
	float2 duv_dy;

	float2 thermal_vel; // in pixel coordinates
	float thermal_move_vol;

	float height_laplacian;
} TerrainState;


typedef struct
{
	float f_L, f_R, f_T, f_B; // outflow flux.  (m^3 s^-1)
	float sed_f_L, sed_f_R, sed_f_T, sed_f_B; // outflow sediment flux.  (m^3 s^-1)

} FlowState;

typedef struct
{
	float flux[8]; // outflow flux per unit area of cell.  m^3 s^-1 / m^2 = m s^-1

} ThermalErosionState;


//typedef struct
//{
//	float u, v; // velocity
//
//} WaterVelState;


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

	// Draw options:
	int include_water_height;
	int draw_water;

	float rock_col[3];
	float sediment_col[3];
	float vegetation_col[3];

	int debug_draw_channel;
	float debug_display_max_val;

	float water_z_bias;
} Constants;


float rainfallFactorForCoords(int x, int y)
{
	const float px = (float)x;
	const float py = (float)y;

	//return length((float2)(px, py) - (float2)(W - 70, H/2.f)) < 50.f ? 1.f : 0.f;
	return 1.f;
}

// NEW: sets water_vel_laplacian, water_vel_partial_derivs
// Sets f_L, f_T, f_R, f_B in new_flow_state
__kernel void flowSimulationKernel(
	__global       TerrainState* restrict const terrain_state, 
	__global const FlowState* restrict const flow_state, 
	__global       FlowState* restrict const new_flow_state, 
	__constant Constants* restrict const constants
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int x_minus_1 = max(x-1, 0);
	const int x_plus_1  = min(x+1, constants->W-1);
	const int y_minus_1 = max(y-1, 0);
	const int y_plus_1  = min(y+1, constants->W-1);

	__global const TerrainState* const state_left     = &terrain_state[x_minus_1 + y         * constants->W];
	__global const TerrainState* const state_right    = &terrain_state[x_plus_1  + y         * constants->W];
	__global const TerrainState* const state_top      = &terrain_state[x         + y_plus_1  * constants->W];
	__global const TerrainState* const state_bot      = &terrain_state[x         + y_minus_1 * constants->W];
	__global       TerrainState* const state_middle   = &terrain_state[x         + y          *constants->W];

	const float2 uv_m = state_middle ->water_vel;
	const float2 uv_L = state_left   ->water_vel;
	const float2 uv_R = state_right  ->water_vel;
	const float2 uv_T = state_top    ->water_vel;
	const float2 uv_B = state_bot    ->water_vel;

	const float2 duv_dx = (uv_R - uv_L) / (2 * constants->cell_w);
	const float2 duv_dy = (uv_T - uv_B) / (2 * constants->cell_w);

	/*const float d2_u_dx2 = (uv_R.x - 2*uv_m + uv_L) / square(constants->cell_w);
	const float d2_u_dy2 = (uv_T.x - 2*uv_m + uv_B) / square(constants->cell_w);
	const float d2_v_dx2 = (uv_R.y - 2*uv_m + uv_L) / square(constants->cell_w);
	const float d2_v_dy2 = (uv_T.y - 2*uv_m + uv_B) / square(constants->cell_w);*/
	const float2 d2_uv_dx2 = (uv_R - 2*uv_m + uv_L) / square(constants->cell_w); // (d^2u/dx^2, d^2v/dx^2)
	const float2 d2_uv_dy2 = (uv_T - 2*uv_m + uv_B) / square(constants->cell_w); // (d^2u/dy^2, d^2v/dy^2)

	const float2 uv_curv = d2_uv_dx2 + d2_uv_dy2; // (d^2u/dx^2 + d^2u/dy^2, d^2v/dx^2 + d^2v/dy^2)

	state_middle->water_vel_laplacian = uv_curv;

	state_middle->duv_dx = duv_dx;
	state_middle->duv_dy = duv_dy;


#if 0
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int x_minus_1 = max(x-1, 0);
	const int x_plus_1  = min(x+1, W-1);
	const int y_minus_1 = max(y-1, 0);
	const int y_plus_1  = min(y+1, H-1);

	__global const TerrainState* const state_left     = &terrain_state[x_minus_1 + y         * W];
	__global const TerrainState* const state_right    = &terrain_state[x_plus_1  + y         * W];
	__global const TerrainState* const state_top      = &terrain_state[x         + y_plus_1  * W];
	__global const TerrainState* const state_bot      = &terrain_state[x         + y_minus_1 * W];
	__global const TerrainState* const state_middle   = &terrain_state[x         + y          *W];

	__global const FlowState* const flow_state_middle     = &flow_state    [x         + y          *W];
	__global       FlowState* const new_flow_state_middle = &new_flow_state[x         + y          *W];

	// Step 1: water increment

	//// Compute intermediate water height (eqn. 1)
	//const float d_1   = state_middle->water + constants->delta_t * constants->r * rainfallFactorForCoords(x, y);
	//const float d_L_1 = state_left  ->water + constants->delta_t * constants->r * rainfallFactorForCoords(x_minus_1, y);
	//const float d_T_1 = state_top   ->water + constants->delta_t * constants->r * rainfallFactorForCoords(x, y_plus_1);
	//const float d_R_1 = state_right ->water + constants->delta_t * constants->r * rainfallFactorForCoords(x_plus_1, y);
	//const float d_B_1 = state_bot   ->water + constants->delta_t * constants->r * rainfallFactorForCoords(x, y_minus_1);

	//// Step 2: Flow simulation

	//// Eqn. 3: Compute total height difference between this cell and adjacent cells 
	//// NOTE: since rainfall is constant for all cells, it cancels out, so ignore when computing height differences.
	///*const float middle_total_h = state_middle->height + state_middle->deposited_sed + d_1;
	//const float delta_h_L = middle_total_h - (state_left ->height + state_left ->deposited_sed + d_L_1);
	//const float delta_h_T = middle_total_h - (state_top  ->height + state_top  ->deposited_sed + d_T_1);
	//const float delta_h_R = middle_total_h - (state_right->height + state_right->deposited_sed + d_R_1);
	//const float delta_h_B = middle_total_h - (state_bot  ->height + state_bot  ->deposited_sed + d_B_1);*/

	//const float h_L = state_left ->height + state_left ->deposited_sed + d_L_1;
	//const float h_R = state_right->height + state_right->deposited_sed + d_R_1;
	//const float h_T = state_top  ->height + state_top  ->deposited_sed + d_T_1;
	//const float h_B = state_bot  ->height + state_bot  ->deposited_sed + d_B_1;


	//// NEW:
	//// Compute gradient of resulting water surface
	//float2 grad = (float2)(
	//	(h_R - h_L) / (2.0 * constants->cell_w), 
	//	(h_T - h_B) / (2.0 * constants->cell_w)
	//);

	//// our force is proportional to the negative of this gradient.

	//// F = -dE/dx = -d(mgh)/dx = -mg dh/dx
	//// F = ma  => a = F/m   =>    a = -mg dh/dx / m = -g dh/dx
	//float2 accel = -constants->g * grad;

	//// Add to velocity
	//new_flow_state_middle







	

	// Eqn. 2: Compute outflow flux to adjacent cells
	const float h_p = state_middle->water;

	const float friction_factor = (1.0f - constants->f * constants->delta_t / h_p);

	// Assume pipe width = pipe length (w = l), so w/l factor in flux deriv = 1.
	const float flux_factor = constants->delta_t * min(5.0f, h_p) * constants->g; // TEMP HACK use max value for pipe height.  Suppresses water oscillations in deep water.
	// TODO: Work out the correct behaviour here.
	float f_L_next = max(0.f, flow_state_middle->f_L * friction_factor  +  flux_factor * delta_h_L); // If this cell is higher than left cell, delta_h_L is positive
	float f_T_next = max(0.f, flow_state_middle->f_T * friction_factor  +  flux_factor * delta_h_T);
	float f_R_next = max(0.f, flow_state_middle->f_R * friction_factor  +  flux_factor * delta_h_R);
	float f_B_next = max(0.f, flow_state_middle->f_B * friction_factor  +  flux_factor * delta_h_B);

	// fluid speed = flow flux volume / area.
	// area = d_1 * 1 * 1 = d_1
/*	float s_L = f_L_next / d_1;
	float s_R = f_R_next / d_1;
	float s_B = f_B_next / d_1;
	float s_T = f_T_next / d_1;
	
	// If any fluid flow speeds exceed max_speed, reduce flux so that the speed = max_speed.
	const float max_speed = 8.0;
	if(s_L > max_speed) f_L_next *= max_speed / s_L;
	if(s_R > max_speed) f_R_next *= max_speed / s_R;
	if(s_B > max_speed) f_B_next *= max_speed / s_B;
	if(s_T > max_speed) f_T_next *= max_speed / s_T;*/

	// Enforce boundary conditions: no flux over boundary
	if(x == 0)
		f_L_next = 0;
	else if(x == W-1)
		f_R_next = 0;
	if(y == 0)
		f_B_next = 0;
	else if(y == H-1)
		f_T_next = 0;

	

	// d_1 * cell_w * cell_w = current water volume in cell
	// (f_L_next + f_T_next + f_R_next + f_B_next) * delta_t = volume of water to be removed next timestep.  (m^3 s^-1  .  s = m^3)
	// If the volume of water to be removed is > current volume, we scale down the volume of water to be removed.
	const float cur_vol = d_1 * square(constants->cell_w);
	const float delta_vol = (f_L_next + f_T_next + f_R_next + f_B_next) * constants->delta_t;
	float K = min(1.f, cur_vol / delta_vol); // Eqn. 4

	f_L_next *= K;
	f_T_next *= K;
	f_R_next *= K;
	f_B_next *= K;

	new_flow_state_middle->f_L = f_L_next;
	new_flow_state_middle->f_R = f_R_next;
	new_flow_state_middle->f_T = f_T_next;
	new_flow_state_middle->f_B = f_B_next;


	// Set out sediment flux
	// fraction of water volume moved next timestep is
	// out_frac = f_L_next * delta_t / cur_vol         (m^3 s^1  .  s   /   m^3)
	// Volume of sediment moved next timestep:
	// delta_sed_vol = cur_suspended_vol * out_frac
	// sed flux (vol/s) of sediment:
	// sed flux = delta_sed_vol / delta_t
	//          = (cur_suspended_vol * out_frac) / delta_t
	//          = cur_suspended_vol * (f_L_next * delta_t / cur_vol) / delta_t
	//          = cur_suspended_vol * f_L_next / cur_vol
	//
	// Example: 100 m^3 of water in cell, 10 m^3 s^-1 water flux.   (1/10 of water removed each second)
	// 50 m^3 of suspended sediment in cell.
	// sed flux
	// = (50 m^3 / 100 m^3) * 10 m^3 s^-1 = 5 m^3 s^-1   (1/10 of the suspended sediment removed each second)

	const float cur_suspended_vol = state_middle->suspended_vol;
	new_flow_state_middle->sed_f_L = cur_suspended_vol / cur_vol * f_L_next;
	new_flow_state_middle->sed_f_R = cur_suspended_vol / cur_vol * f_R_next;
	new_flow_state_middle->sed_f_T = cur_suspended_vol / cur_vol * f_T_next;
	new_flow_state_middle->sed_f_B = cur_suspended_vol / cur_vol * f_B_next;

#endif
}



float waterHeightForMass(float water_mass, __constant Constants* restrict const constants)
{
	float water_density = 1000.f;
	return water_mass / (square(constants->cell_w) * water_density);
}

float waterMassForHeight(float water_height, __constant Constants* restrict const constants)
{
	float water_density = 1000.f;
	return water_height * (square(constants->cell_w) * water_density);
}



// NEW: Updates water_vel, updates water_mass from rainfall
// Updates water, u, v in terrain_state
__kernel void waterAndVelFieldUpdateKernel(
	//__global const FlowState* restrict const flow_state, 
	__global       TerrainState* restrict const terrain_state, 
	__constant Constants* restrict const constants
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int x_minus_1 = max(x-1, 0);
	const int x_plus_1  = min(x+1, constants->W-1);
	const int y_minus_1 = max(y-1, 0);
	const int y_plus_1  = min(y+1, constants->H-1);


	/*__global const FlowState* const state_left     = &flow_state[x_minus_1 + y         * W];
	__global const FlowState* const state_right    = &flow_state[x_plus_1  + y         * W];
	__global const FlowState* const state_top      = &flow_state[x         + y_plus_1  * W];
	__global const FlowState* const state_bot      = &flow_state[x         + y_minus_1 * W];
	__global const FlowState* const state_middle   = &flow_state[x         + y          *W];

	__global TerrainState* const terrain_state_middle   = &terrain_state[x         + y          *W];*/

	__global const TerrainState* const state_left     = &terrain_state[x_minus_1 + y         * constants->W];
	__global const TerrainState* const state_right    = &terrain_state[x_plus_1  + y         * constants->W];
	__global const TerrainState* const state_top      = &terrain_state[x         + y_plus_1  * constants->W];
	__global const TerrainState* const state_bot      = &terrain_state[x         + y_minus_1 * constants->W];
	__global       TerrainState* const state_middle   = &terrain_state[x         + y         * constants->W];




	//NEW:
	// Compute intermediate water height (water depth plus rainfall depth) (eqn. 1)
	const float d_m = waterHeightForMass(state_middle->water_mass, constants) + constants->delta_t * constants->r * rainfallFactorForCoords(x, y);
	const float d_L = waterHeightForMass(state_left  ->water_mass, constants) + constants->delta_t * constants->r * rainfallFactorForCoords(x_minus_1, y);
	const float d_T = waterHeightForMass(state_top   ->water_mass, constants) + constants->delta_t * constants->r * rainfallFactorForCoords(x, y_plus_1);
	const float d_R = waterHeightForMass(state_right ->water_mass, constants) + constants->delta_t * constants->r * rainfallFactorForCoords(x_plus_1, y);
	const float d_B = waterHeightForMass(state_bot   ->water_mass, constants) + constants->delta_t * constants->r * rainfallFactorForCoords(x, y_minus_1);

	// Compute total water surface height: (terrain height + water depth)
	const float h_m = state_middle->height + state_middle ->deposited_sed_h + d_m;
	const float h_L = state_left  ->height + state_left   ->deposited_sed_h + d_L;
	const float h_R = state_right ->height + state_right  ->deposited_sed_h + d_R;
	const float h_T = state_top   ->height + state_top    ->deposited_sed_h + d_T;
	const float h_B = state_bot   ->height + state_bot    ->deposited_sed_h + d_B;


	// NEW:
	// Compute gradient of resulting water surface height
	float2 grad = (float2)(
		(h_R - h_L) / (2.0 * constants->cell_w), 
		(h_T - h_B) / (2.0 * constants->cell_w)
	);

	//const float2 uv_m = state_middle ->water_vel;
	//const float2 uv_L = state_left   ->water_vel;
	//const float2 uv_R = state_right  ->water_vel;
	//const float2 uv_T = state_top    ->water_vel;
	//const float2 uv_B = state_bot    ->water_vel;

	///*const float d2_u_dx2 = (uv_R.x - 2*uv_m + uv_L) / square(constants->cell_w);
	//const float d2_u_dy2 = (uv_T.x - 2*uv_m + uv_B) / square(constants->cell_w);
	//const float d2_v_dx2 = (uv_R.y - 2*uv_m + uv_L) / square(constants->cell_w);
	//const float d2_v_dy2 = (uv_T.y - 2*uv_m + uv_B) / square(constants->cell_w);*/
	//const float2 d2_uv_dx2 = (uv_R - 2*uv_m + uv_L) / square(constants->cell_w); // (d^2u/dx^2, d^2v/dx^2)
	//const float2 d2_uv_dy2 = (uv_T - 2*uv_m + uv_B) / square(constants->cell_w); // (d^2u/dy^2, d^2v/dy^2)

	//const float2 uv_curv = d2_uv_dx2 + d2_uv_dy2; // (d^2u/dx^2 + d^2u/dy^2, d^2v/dx^2 + d^2v/dy^2)


	// our force is proportional to the negative of this gradient.

	// F = -dE/dx = -d(mgh)/dx = -mg dh/dx
	// F = ma  => a = F/m   =>    a = -mg dh/dx / m = -g dh/dx

	
	// k is viscous drag coefficient: https://en.wikipedia.org/wiki/Shallow_water_equations

	//const float friction_factor = (1.0f - constants->f * constants->delta_t / d_m_1);

	// TEMP HACK 1.0 grav g
	float2 accel = -constants->g * grad - constants->k * state_middle->water_vel + constants->nu * state_middle->water_vel_laplacian 
		- (constants->f / max(0.1f, d_m)) * state_middle->water_vel
		;// -
	//	/*u=*/state_middle->water_vel.x * state_middle->duv_dx - 
	//	/*v=*/state_middle->water_vel.y * state_middle->duv_dy;


	//float mov_frac = length(grad) / h_m;

	// Integrate acceleration, adding to velocity
	state_middle->water_vel += constants->delta_t * accel/* * mov_frac*/;



	// Apply friction approx
	//const float friction_factor = (1.0f - constants->f * constants->delta_t/* / h_p*/);
	//state_middle->water_vel *= friction_factor;


	// Add rainfall to water_mass
	const float delta_water_h = constants->delta_t * constants->r * rainfallFactorForCoords(x, y);
	const float delta_water_mass = waterMassForHeight(delta_water_h, constants);

	//if(x == 100 && y == 100)
	//{
	//	printf("initial state_middle->water_mass: %f \n", state_middle->water_mass);
	//	printf("initial state_middle->water_vel: %f, %f \n", state_middle->water_vel.x, state_middle->water_vel.y);
	//	printf("initial state_middle->water_position: %f, %f \n", state_middle->water_position.x, state_middle->water_position.y);
	//}


	// Update water_position based on weighted mass of old water and new rainfall water
	const float new_total_mass = state_middle->water_mass + delta_water_mass;
	if(new_total_mass > 0.0)
	{
		const float orig_mass_frac = state_middle->water_mass / new_total_mass;

		//state_middle->water_position = state_middle->water_position * orig_mass_frac + (float2)(x + 0.5f, y + 0.5f) * (delta_water_mass / new_total_mass);
		state_middle->water_mass = new_total_mass;

		state_middle->water_vel *= orig_mass_frac; // Rainfall has zero lateral velocity, adjust cell water vel accordingly.
	}


	//TEMP:
	 // Limit water speed so that water can't move more than 1 grid cell per time step, otherwise the reintegration procedure will 'lose' the water.
	float v = length(state_middle->water_vel);
	float max_v = constants->cell_w;
	if(v > max_v)
		state_middle->water_vel *= max_v / v;
	//state_middle->water_vel *= (v > max_v) ? (max_v/v) : 1.0f;

	/*if(x == 100 && y == 100)
	{
		printf("state_middle->water_mass: %f \n", state_middle->water_mass);
		printf("state_middle->water_vel: %f, %f \n", state_middle->water_vel.x, state_middle->water_vel.y);
		printf("state_middle->water_position: %f, %f \n", state_middle->water_position.x, state_middle->water_position.y);
	}*/

	// Boundary conditions: force zero velocity out of boundaries:
	if(x == 0)
		state_middle->water_vel.x = max(state_middle->water_vel.x, 0.f);
	else if(x == constants->W - 1)
		state_middle->water_vel.x = min(state_middle->water_vel.x, 0.f);

	if(y == 0)
		state_middle->water_vel.y = max(state_middle->water_vel.y, 0.f);
	else if(y == constants->H - 1)
		state_middle->water_vel.y = min(state_middle->water_vel.y, 0.f);


#if 0


	// Step 3: Water surface and velocity field update

	//const float d_1 = state_middle->water; // Current water height of middle cell
	// Compute intermediate water height (eqn. 1)
	const float d_1 = terrain_state_middle->water + constants->delta_t * constants->r * rainfallFactorForCoords(x, y);

	// Get water fluxes (m^3 s^-1)
	float in_left_R  = (x > 0)   ? state_left ->f_R : 0; // If this cell is on the left border, inwards flux from left is zero.  Otherwise get from left cell.
	float in_right_L = (x < W-1) ? state_right->f_L : 0;
	float in_bot_T   = (y > 0)   ? state_bot  ->f_T : 0;
	float in_top_B   = (y < H-1) ? state_top  ->f_B : 0;

	float in_sed_left_R  = (x > 0)   ? state_left ->sed_f_R : 0; // If this cell is on the left border, inwards flux from left is zero.  Otherwise get from left cell.
	float in_sed_right_L = (x < W-1) ? state_right->sed_f_L : 0;
	float in_sed_bot_T   = (y > 0)   ? state_bot  ->sed_f_T : 0;
	float in_sed_top_B   = (y < H-1) ? state_top  ->sed_f_B : 0;

	// Compute net volume change for the water (eqn 6):
	const float delta_V = constants->delta_t *
		((in_left_R + in_right_L + in_top_B + in_bot_T) - // inwards flow
		 (state_middle->f_L + state_middle->f_R + state_middle->f_T + state_middle->f_B)); // outwards flow
	// m^3 = s * (m^3 s^-1)

	//    m   = m   + m^3     / (m               * m)
	float d_2 = max(0.f, d_1 + delta_V / square(constants->cell_w)); // Eqn. 7: new water height for middle cell: change in height = change in volume / cell area.  
	// Also make sure water level doesn't become negative.

	// Eqn 8.  Compute average amount of water passing through cell (x, y) in the x direction:
	// m^3 s^-1     = m^3 s^-1
	float delta_W_x = (in_left_R - state_middle->f_L + state_middle->f_R - in_right_L) * 0.5f;

	//if(x == 0 || x == W-1)
	//	delta_W_x = 0;

	// Compute average amount of water passing through cell (x, y) in the y direction:
	float delta_W_y = (in_bot_T  - state_middle->f_B + state_middle->f_T - in_top_B) * 0.5f;

	//if(y == 0 || y == H-1)
	//	delta_W_y = 0;

	const float d_bar = (d_1 + d_2) * 0.5f; // Average water height


	// Compute average amount of sediment passing through cell (x, y) in the x direction (per unit time):  (m^3 s^-1)
	/*float delta_sed_x = (in_sed_left_R - state_middle->sed_f_L + state_middle->sed_f_R - in_sed_right_L) * 0.5f;
	float delta_sed_y = (in_sed_bot_T  - state_middle->sed_f_B + state_middle->sed_f_T - in_sed_top_B) * 0.5f;*/
	float delta_sed_x = (fabs(in_sed_left_R - state_middle->sed_f_L) + fabs(state_middle->sed_f_R - in_sed_right_L)) * 0.5f;
	float delta_sed_y = (fabs(in_sed_bot_T  - state_middle->sed_f_B) + fabs(state_middle->sed_f_T - in_sed_top_B)) * 0.5f;


	const float delta_sed_V = constants->delta_t *
		((in_sed_left_R + in_sed_right_L + in_sed_top_B + in_sed_bot_T) - // inwards flow
		 (state_middle->sed_f_L + state_middle->sed_f_R + state_middle->sed_f_T + state_middle->sed_f_B)); // outwards sediment flow
	// m^3 = s * (m^3 s^-1)

	// Compute new amount of sediment
	const float new_suspended_vol = max(0.f, terrain_state_middle->suspended_vol + delta_sed_V);


	//terrain_state_middle->sed_flux = sqrt(square(delta_sed_x) + square(delta_sed_y));

	//TEMP: store unit discharge in u, v
//	float u = delta_W_x;
//	float v = delta_W_y;

	//float max_speed_comp = 1.f;

//	float u, v;
//	if(d_bar <= 1.0e-4f) // If the water height is ~= 0, then avoid divide by zero below and consider the water velocity to be zero.
//	{
//		u = v = 0;
//	}
//	else
//	{
//		// From eqn. 9:
//		//m^s-1 = m^3 s^-1  / (m     * m)
//		float new_u = delta_W_x / (d_bar * constants->l_x); // u_{t+delta_t}
//		float new_v = delta_W_y / (d_bar * constants->l_y); // v_{t+delta_t}
//
//		//const float old_u = terrain_state_middle->u;
//		//const float old_v = terrain_state_middle->v;
//
//		// TEMP HACK:
//		//u = old_u * 0.8f + new_u * 0.1f;
//		//v = old_v * 0.8f + new_v * 0.1f;
//		u = new_u;
//		v = new_v;
//	}
	 
	// water vel in x direction 
	// = water flux / cross sectional area  
	//   m^3 s^-1   / m^2           =   m s^1
	// 
	float water_u = delta_W_x / (d_bar * constants->cell_w); // u_{t+delta_t}
	float water_v = delta_W_y / (d_bar * constants->cell_w); // v_{t+delta_t}

	float water_vel = sqrt(square(water_u) + square(water_v));

	//const float old_u = terrain_state_middle->u;
	//const float old_v = terrain_state_middle->v;
	//
	//float u = old_u * 0.4f + new_u * 0.4f;
	//float v = old_v * 0.4f + new_v * 0.4f;

	//if(d_2 < 0.001)
	//{
	//	u = v = 0; // TEMP HACK
	//}

	//if(d_2 < 0.01f) // TEMP: force water depth to 0 if too small
	//{
	//	d_2 = 0;
	//	u = 0;
	//	v = 0;
	//}

	//const float v_len = sqrt(u*u + v*v);

	//if(x == 200 && y == 200)
	//	printf("v_len: %f  \n", v_len);
	//if(v_len > 40.0f)
	//{
	//	const float scale = 40.0 / v_len;
	//	u *= scale;
	//	v *= scale;
	//}

	// Sea boundary conditions:
	// If this is an edge cell, and if terrain level is below sea level, set water height so that the total terrain + water height = sea level.
	if((x == 0) || (x == W-1) || (y == 0) || (y == H-1))
	{
		float sea_level = constants->sea_level;
		//if(x == 0)
		//	sea_level = constants->sea_level + sin(constants->current_time) * 6.0; // incoming water waves!
		const float total_terrain_h = terrain_state_middle->height + terrain_state_middle->deposited_sed_h;
		if(total_terrain_h < sea_level)
			d_2 = sea_level - total_terrain_h;
	}

	terrain_state_middle->water = d_2;
#if !DO_SEMILAGRANGIAN_ADVECTION
	terrain_state_middle->suspended_vol = new_suspended_vol;
#endif
	terrain_state_middle->u = water_u;
	terrain_state_middle->v = water_v;
	terrain_state_middle->water_vel = water_vel;

#endif
}


float3 overlap(float2 x, float2 p, float diffusion_radius)
{
    float4 aabb0 = (float4)(p - (float2)(0.5), p + (float2)(0.5)); //cell box
    float4 aabb1 = (float4)(x - diffusion_radius, x + diffusion_radius); //particle box
    float4 aabbX = (float4)(max(aabb0.xy, aabb1.xy), min(aabb0.zw, aabb1.zw)); //overlap box
    float2 center = 0.5f*(aabbX.xy + aabbX.zw); //center of mass 
    float2 size = max(aabbX.zw - aabbX.xy, (float2)(0.0)); //only positive
    float m = size.x*size.y/(4.0*diffusion_radius*diffusion_radius); //relative area
    //if any of the dimensions are 0 then the mass ratio is 0
    return (float3)(center, m);
}


// NEW: updates height, suspended_vol, deposited_sed_h
// Updates 'height', 'suspended', 'sediment' in terrain_state
__kernel void erosionAndDepositionKernel(
	__global       TerrainState* restrict const terrain_state, 
	__constant Constants* restrict const constants
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int x_minus_1 = max(x-1, 0);
	const int x_plus_1  = min(x+1, constants->W-1);
	const int y_minus_1 = max(y-1, 0);
	const int y_plus_1  = min(y+1, constants->H-1);

	//__global const TerrainState* const state_left     = &terrain_state[x_minus_1 + y         * constants->W];
	//__global const TerrainState* const state_right    = &terrain_state[x_plus_1  + y         * constants->W];
	//__global const TerrainState* const state_top      = &terrain_state[x         + y_plus_1  * constants->W];
	//__global const TerrainState* const state_bot      = &terrain_state[x         + y_minus_1 * constants->W];
	__global       TerrainState* const state_middle   = &terrain_state[x         + y         * constants->W];


#if 1

	// NEW: assign
	state_middle->water_mass    = state_middle->new_water_mass;
	state_middle->water_vel     = state_middle->new_water_vel;
	state_middle->suspended_vol = state_middle->new_suspended_vol;

	
	//const float L_h = state_left ->height + state_left ->deposited_sed_h; // state_left ->sediment[0] + state_left ->sediment[1] + state_left ->sediment[2];// + state_left ->water;
	//const float R_h = state_right->height + state_right->deposited_sed_h; // state_right->sediment[0] + state_right->sediment[1] + state_right->sediment[2];// + state_right->water;
	//const float B_h = state_bot  ->height + state_bot  ->deposited_sed_h; // state_bot  ->sediment[0] + state_bot  ->sediment[1] + state_bot  ->sediment[2];// + state_bot  ->water;
	//const float T_h = state_top  ->height + state_top  ->deposited_sed_h; // state_top  ->sediment[0] + state_top  ->sediment[1] + state_top  ->sediment[2];// + state_top  ->water;

	//const float dh_dx = (R_h - L_h) * 0.5f * constants->recip_cell_w; // dh/dx = (R_h - L_h) / (2*cell_w) = (R_h - L_h) * 0.5 * (1/cell_w)
	//const float dh_dy = (T_h - B_h) * 0.5f * constants->recip_cell_w;

	//const float3 normal = normalize((float3)(-dh_dx, -dh_dy, 1));

	//const float cos_alpha = normal.z;
	//const float sin_alpha = sqrt(1 - min(1.0f, cos_alpha*cos_alpha));
	//const float use_sin_alpha = sin_alpha;//max(0.1f, sin_alpha); // NOTE: min sin alpha

//	float water_flux = sqrt(square(state_middle->u) + square(state_middle->v)); // Volume of water passing through cell per unit time (m^3 s^-1)

	// Compute l_max as a function of water height (d)  (eqn. 10 from 'Fast Hydraulic and Thermal Erosion on the GPU')

//	const float3 water_flux_vec = (float3)(state_middle->u, state_middle->v, (state_middle->u * dh_dx + state_middle->v * dh_dy) * constants->cell_w);
//	const float3 unit_water_vel = normalize(water_flux_vec);

//	const float hit_dot = max(0.05f, -dot(unit_water_vel, normal));
			
	const float water_d = waterHeightForMass(state_middle->water_mass, constants);
	/*float l_max;
	if(d <= 0)
		l_max = 0;
	else if(d >= constants->K_dmax)
		l_max = 1;
	else
		l_max = 1 - (constants->K_dmax - d) / constants->K_dmax;*/

	//const float water_factor = min(0.01f, state_middle->water * 10.0f);
	//const float water_factor = min(1.0f, state_middle->water * 1.0f);

	//const float water_depth_factor = 1.f;//min(water_d, constants->K_dmax);

	// Compute Sediment transport capacity (eq 10)



	//const float C = 0.001f * constants->K_c * v_len;
	//const float q = v_len * max(0.f, min(water_d, 1.0f));
	//const float q = min(square(constants->cell_w), water_flux); //fabs(state_middle->u) + fabs(state_middle->v)); // unit water discharge (water flux per unit width of stream?)  (m^3 s^-1 / m = m^2 s^-1)
	const float q = length(state_middle->water_vel) * min(water_d, constants->K_dmax);//min(10.0f, water_flux / constants->cell_w);
	float q_to_gamma = q;//square(q);

	q_to_gamma = max(0.f, q_to_gamma - constants->q_0);


	const float current_vol = state_middle->suspended_vol;// / (water_d * square(constants->cell_w)); // current vol
	const float max_vol = constants->K_c * length(state_middle->water_vel) * water_d * square(constants->cell_w); // max vol   TEMP HACK

	//const float S = use_sin_alpha;
	//const float S_to_beta = pow(S, 1.5f);
	const float unit_C = constants->K_c /** S_to_beta*/ * q_to_gamma; // m^2 s^-1
	const float C = unit_C;// * constants->cell_w; // m^3 s^-1
			
	float height = state_middle->height;
	float suspended_vol = state_middle->suspended_vol;
	float deposited_sed_h = state_middle->deposited_sed_h;

	float cur_suspended_rate = suspended_vol;//TEMP / constants->cell_w/* * state_middle->water_vel*/;// / constants->delta_t;//state_middle->sed_flux; // TEMP // suspended_vol / constants->cell_w * state_middle->water_vel;
	// m^3 s^-1              = m^3           / m                 * m s^-1
	
	//const float suspended_sum = suspended; // suspended[0] + suspended[1] + suspended[2];
	//if(C > cur_suspended_rate) // suspended amount is smaller than transport capacity, dissolve soil into water:
	
	if(x == 200 && y == 256)
	{
		printf("water_d:  %f   \n", water_d);
		printf("current_vol:  %f   \n", current_vol);
		printf("max_vol:  %f   \n", max_vol);
	}
	
	if(max_vol > current_vol)
	{
		float sed_change_vol = /*hit_dot * */constants->delta_t * constants->K_s * constants->K_c * length(state_middle->water_vel)/* * (max_vol - current_vol)*/; // (C - cur_suspended_rate); // s   .   m^3 s^-1  = m^3
		float sed_change_rock_vol = sed_change_vol * 0.3f; //delta_t * K_s * (C - s_t);
		float sed_change_dep_vol  = sed_change_vol * 0.7f; //delta_t * K_s * (C - s_t);

		// Dissolve any deposited sediment into the water
		const float sed_change_dep_h = sed_change_dep_vol / square(constants->cell_w); // m = m^3 / m^2
		const float deposited_sed_delta_h = min(sed_change_dep_h, deposited_sed_h); // Dissolve <= the amount of deposited sediment here.
		deposited_sed_h -= deposited_sed_delta_h;
		suspended_vol   += deposited_sed_delta_h * square(constants->cell_w);

		//sed_change_dep -= deposited_sed_delta;
		

		//if(sed_change > 0) // If we have dissolved all deposited sediment, and there is still dissolving to be done:
		{
			// Dissolve underlying rock

			height -= sed_change_rock_vol / square(constants->cell_w);// Reduce terrain height
			suspended_vol += sed_change_rock_vol; // Add to suspended height
		}
	}
	else // else suspended amount exceeds transport capacity, so deposit sediment:
	{
		float sed_change_vol = constants->delta_t * constants->K_d * current_vol/* * (current_vol - max_vol)*//*(cur_suspended_rate - C)*/;
		sed_change_vol = min(sed_change_vol, suspended_vol); // Don't exceed current suspended volume

		suspended_vol   -= sed_change_vol;
		deposited_sed_h += sed_change_vol / square(constants->cell_w);
	}
		
	//if(x == 200 && y == 256)
	//	printf("s_t:  %1.15f   , C: %1.15f   \n", s_t, C);

	// Write
	state_middle->height = height;
	state_middle->suspended_vol   = suspended_vol;
	state_middle->deposited_sed_h = deposited_sed_h;

#endif
}




inline float biLerp(float a, float b, float c, float d, float t_x, float t_y)
{
	const float one_t_x = 1 - t_x;
	const float one_t_y = 1 - t_y;
	return 
		one_t_x * one_t_y * a + 
		t_x     * one_t_y * b + 
		one_t_x * t_y     * c + 
		t_x     * t_y     * d;
}


inline float mitchellNetravaliEval(float x)
{
	float B = 0.5f;
	float C = 0.25f;

	float region_0_a = ((12)  - B*9  - C*6) * (1.f/6);
	float region_0_b = ((-18) + B*12 + C*6) * (1.f/6);
	float region_0_d = ((6)   - B*2       ) * (1.f/6);

	float region_1_a = (-B - C*6)                * (1.f/6);
	float region_1_b = (B*6 + C*30)              * (1.f/6);
	float region_1_c = (B*-12 - C*48)            * (1.f/6);
	float region_1_d = (B*8 + C*24)              * (1.f/6);

	float region_0_f = region_0_a * (x*x*x) + region_0_b * (x*x) + region_0_d;
	float region_1_f = region_1_a * (x*x*x) + region_1_b * (x*x) + region_1_c * x + region_1_d;
	if(x < 1.0f)
		return region_0_f;
	else if(x < 2.f)
		return region_1_f;
	else
		return 0;
}


#if 0
inline float mitchellNetravaliCubic(float px, float py, __global       TerrainState* restrict const terrain_state)
{
	int ut_minus_1 = clamp((int)px - 1, 0, W);
	int ut         = clamp((int)px    , 0, W);
	int ut_1       = clamp((int)px + 1, 0, W);
	int ut_2       = clamp((int)px + 2, 0, W);

	int vt_minus_1 = clamp((int)py - 1, 0, H);
	int vt         = clamp((int)py    , 0, H);
	int vt_1       = clamp((int)py + 1, 0, H);
	int vt_2       = clamp((int)py + 2, 0, H);

	float sq_dx_minus_1 = square(px - (float)ut_minus_1);
	float sq_dx         = square(px - (float)ut);
	float sq_dx_1       = square(px - (float)ut_1);
	float sq_dx_2       = square(px - (float)ut_2);

	float sq_dy_minus_1 = square(py - (float)vt_minus_1);
	float sq_dy         = square(py - (float)vt);
	float sq_dy_1       = square(py - (float)vt_1);
	float sq_dy_2       = square(py - (float)vt_2);

	const float v0 = terrain_state[(ut_minus_1   + W * vt_minus_1  )].suspended;
	const float v1 = terrain_state[(ut           + W * vt_minus_1  )].suspended;
	const float v2 = terrain_state[(ut_1         + W * vt_minus_1  )].suspended;
	const float v3 = terrain_state[(ut_2         + W * vt_minus_1  )].suspended;

	const float v4 = terrain_state[(ut_minus_1   + W * vt          )].suspended;
	const float v5 = terrain_state[(ut           + W * vt          )].suspended;
	const float v6 = terrain_state[(ut_1         + W * vt          )].suspended;
	const float v7 = terrain_state[(ut_2         + W * vt          )].suspended;

	const float  v8 = terrain_state[(ut_minus_1   + W * vt_1        )].suspended;
	const float  v9 = terrain_state[(ut           + W * vt_1        )].suspended;
	const float v10 = terrain_state[(ut_1         + W * vt_1        )].suspended;
	const float v11 = terrain_state[(ut_2         + W * vt_1        )].suspended;

	const float v12 = terrain_state[(ut_minus_1   + W * vt_2        )].suspended;
	const float v13 = terrain_state[(ut           + W * vt_2        )].suspended;
	const float v14 = terrain_state[(ut_1         + W * vt_2        )].suspended;
	const float v15 = terrain_state[(ut_2         + W * vt_2        )].suspended;

	float w0  = mitchellNetravaliEval(sqrt(sq_dx_minus_1 + sq_dy_minus_1));
	float w1  = mitchellNetravaliEval(sqrt(sq_dx         + sq_dy_minus_1));
	float w2  = mitchellNetravaliEval(sqrt(sq_dx_1       + sq_dy_minus_1));
	float w3  = mitchellNetravaliEval(sqrt(sq_dx_2       + sq_dy_minus_1));
			  
	float w4  = mitchellNetravaliEval(sqrt(sq_dx_minus_1 + sq_dy));
	float w5  = mitchellNetravaliEval(sqrt(sq_dx         + sq_dy));
	float w6  = mitchellNetravaliEval(sqrt(sq_dx_1       + sq_dy));
	float w7  = mitchellNetravaliEval(sqrt(sq_dx_2       + sq_dy));

	float w8  = mitchellNetravaliEval(sqrt(sq_dx_minus_1 + sq_dy_1));
	float w9  = mitchellNetravaliEval(sqrt(sq_dx         + sq_dy_1));
	float w10 = mitchellNetravaliEval(sqrt(sq_dx_1       + sq_dy_1));
	float w11 = mitchellNetravaliEval(sqrt(sq_dx_2       + sq_dy_1));

	float w12 = mitchellNetravaliEval(sqrt(sq_dx_minus_1 + sq_dy_2));
	float w13 = mitchellNetravaliEval(sqrt(sq_dx         + sq_dy_2));
	float w14 = mitchellNetravaliEval(sqrt(sq_dx_1       + sq_dy_2));
	float w15 = mitchellNetravaliEval(sqrt(sq_dx_2       + sq_dy_2));

	const float filter_sum = 
		((w0 + w1 + w2 + w3) + 
		(w4 + w5 + w6 + w7)) + 
		((w8 + w9 + w10 + w11) + 
		(w12 + w13 + w14 + w15));

	const float sum = 
		(((v0  * w0 +
		   v1  * w1) +
		  (v2  * w2 +
		   v3  * w3)) +

  		 ((v4  * w4 +
		   v5  * w5) +
		  (v6  * w6 +
		   v7  * w7))) +

		(((v8  * w8 +
		   v9  * w9) +
		  (v10 * w10 +
		   v11 * w11)) +

		 ((v12 * w12 +
		   v13 * w13) +
		  (v14 * w14 +
		   v15 * w15)));

	return sum / filter_sum;
}
#endif

// NEW: transports both water and sediment
// sediment transportation kernel.  Updates 'suspended' in terrain_state
__kernel void sedimentTransportationKernel(
	__global       TerrainState* restrict const terrain_state, 
	__constant Constants* restrict const constants
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	__global       TerrainState* const state_middle   = &terrain_state[x         + y          *constants->W];

#if 1
	// Loop over neighbouring cells
	float2 total_water_momentum_in = (float2)(0.f, 0.f); // Aka total water momentum
	float total_mass_in = 0.f;
	float total_suspended_vol_in = 0.f;
	for(int ny = y-1; ny <= y+1; ny++)
	for(int nx = x-1; nx <= x+1; nx++)
	{
		if(nx >= 0 && nx < constants->W && ny >= 0 && ny < constants->H)
		{
			__global const TerrainState* const n_state = &terrain_state[nx + ny * constants->W];
			float2 vel_px_coords = n_state->water_vel * constants->recip_cell_w;  
			float2 new_pos = (float2)(nx, ny) + vel_px_coords * constants->delta_t;

			/*if(x == 100 && y == 100)
			{
				printf("nx, ny: %f, %f \n", (float)nx, (float)ny);
				printf("n_state->water_position: %f, %f \n", n_state->water_position.x, n_state->water_position.y);
				printf("n_state->water_vel: %f, %f \n", n_state->water_vel.x, n_state->water_vel.y);
				printf("n_state->water_mass: %f \n", n_state->water_mass);
				printf("new_pos: %f, %f \n", new_pos.x, new_pos.y);
			}*/

			const float x_diff = fabs((float)x - new_pos.x);
			const float y_diff = fabs((float)y - new_pos.y);
			const float weight = max(0.f, (1 - x_diff)) * max(0.f, (1 - y_diff));
			const float weighted_mass = n_state->water_mass * weight;
			total_mass_in            += weighted_mass;
			total_water_momentum_in  += n_state->water_vel * weighted_mass;
			total_suspended_vol_in   += n_state->suspended_vol * weight;
		}
	}

	if(total_mass_in > 0)
		total_water_momentum_in /= total_mass_in; // Convert from momentum to velocity

	state_middle->new_water_vel     = total_water_momentum_in;
	state_middle->new_water_mass    = total_mass_in;
	state_middle->new_suspended_vol = total_suspended_vol_in;

#else
	// Loop over neighbouring cells
	float2 water_pos = (float2)(0.f, 0.f);
	float2 water_vel = (float2)(0.f, 0.f);
	float water_mass = 0.f;
	float sediment_vol = 0.f;
	for(int ny = y-1; ny <= y+1; ny++)
	for(int nx = x-1; nx <= x+1; nx++)
	{
		if(nx >= 0 && nx < W && ny >= 0 && ny < H)
		{
			__global const TerrainState* const n_state = &terrain_state[nx + ny * W];
			float2 vel_px_coords = n_state->water_vel * constants->recip_cell_w;  
			float2 new_pos = (float2)(nx, ny)/*n_state->water_position */ + vel_px_coords * constants->delta_t; // TEMP not using water pos

			/*if(x == 100 && y == 100)
			{
				printf("nx, ny: %f, %f \n", (float)nx, (float)ny);
				printf("n_state->water_position: %f, %f \n", n_state->water_position.x, n_state->water_position.y);
				printf("n_state->water_vel: %f, %f \n", n_state->water_vel.x, n_state->water_vel.y);
				printf("n_state->water_mass: %f \n", n_state->water_mass);
				printf("new_pos: %f, %f \n", new_pos.x, new_pos.y);
			}*/
			
			float3 ovrlp = overlap(new_pos, (float2)((float)x, (float)y), /*diffusion_radius=*/0.5f);
			float overlapRelativeArea = ovrlp.z;
			float2 overlapCenterOfMass = ovrlp.xy;
			float overlapMass = overlapRelativeArea * n_state->water_mass;

			water_mass   += overlapMass;
			water_pos    += overlapCenterOfMass    * overlapMass; //add the overlap center weighted by mass
			water_vel    += n_state->water_vel     * overlapMass; //add the particle velocity weighted by overlap mass(momentum)
			sediment_vol += n_state->suspended_vol * overlapRelativeArea;//overlapMass; // Add the particle sediment_vol weighted by overlap mass
		}
	}

	//normalize
	if(water_mass > 0.0) //if not vacuum
	{
	  water_pos    /= water_mass; //center of mass
	  water_vel    /= water_mass; //average velocity
	  //sediment_vol /= water_mass;
	}

	// Write to state array
	state_middle->water_position = water_pos;
	state_middle->water_vel = water_vel;
	state_middle->water_mass = water_mass;
	state_middle->suspended_vol = sediment_vol;
#endif


#if 0 // DO_SEMILAGRANGIAN_ADVECTION
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	__global       TerrainState* const state_middle   = &terrain_state[x         + y          *W];

	float u = state_middle->u;
	float v = state_middle->v;

	// If x = grid cell coords, p_x = position coords:
	// old_p_x = p_x - (d p_x / dt) delta_t
	// old_p_x = p_x - u delta_t
	// old_x = old_p_x / cell_w = (p_x - u delta_t) / cell_w
	// = (x * cell_w - u delta_t) / cell_w
	// = (x - u delta_t / cell_w) 
	const float old_x = clamp((float)x - /*state_middle->*/u * constants->delta_t * constants->recip_cell_w, 0.0f, (float)(W-1));
	const float old_y = clamp((float)y - /*state_middle->*/v * constants->delta_t * constants->recip_cell_w, 0.0f, (float)(H-1));

	//printf("old_x: %f \n", old_x);
	//printf("old_y: %f \n", old_y);

	const float floor_old_x = floor(old_x);
	const float floor_old_y = floor(old_y);
	const float t_x = old_x - floor_old_x;//(float)(int)floor_old_x;
	const float t_y = old_y - floor_old_y;//(float)(int)floor_old_y;

	//printf("t_x: %f    t_y: %f \n", t_x, t_y);

	const int old_xi = clamp((int)floor_old_x, 0, W-1);
	const int old_yi = clamp((int)floor_old_y, 0, H-1);
	const int old_xi1 = clamp((int)floor_old_x + 1, 0, W-1);
	const int old_yi1 = clamp((int)floor_old_y + 1, 0, H-1);

	// Read sedimentation value at (old_x, old_y)
	/*const float old_s = biLerp(
		terrain_state[old_xi  + old_yi  * W].suspended,
		terrain_state[old_xi1 + old_yi  * W].suspended,
		terrain_state[old_xi  + old_yi1 * W].suspended,
		terrain_state[old_xi1 + old_yi1 * W].suspended,
		t_x, t_y);*/

	const float one_t_x = 1 - t_x;
	const float one_t_y = 1 - t_y;

	const float old_suspended = // terrain_state[old_xi  + old_yi  * W].suspended;
		terrain_state[old_xi  + old_yi  * W].suspended_vol * one_t_x * one_t_y +
		terrain_state[old_xi1 + old_yi  * W].suspended_vol * t_x     * one_t_y +
		terrain_state[old_xi  + old_yi1 * W].suspended_vol * one_t_x * t_y     +
		terrain_state[old_xi1 + old_yi1 * W].suspended_vol * t_x     * t_y     ;

	state_middle->suspended_vol = old_suspended;
#endif
}


// Sets flux in thermal_erosion_state
__kernel void thermalErosionFluxKernel(
	__global TerrainState* restrict const terrain_state, 
	__global ThermalErosionState* restrict const thermal_erosion_state, 
	__constant Constants* restrict const constants,
	int process_deposited_sed
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int x_minus_1 = max(x-1, 0);
	const int x_plus_1  = min(x+1, constants->W-1);
	const int y_minus_1 = max(y-1, 0);
	const int y_plus_1  = min(y+1, constants->H-1);

	
#if 1
	__global const TerrainState* const state_left     = &terrain_state[x_minus_1 + y         * constants->W];
	__global const TerrainState* const state_right    = &terrain_state[x_plus_1  + y         * constants->W];
	__global const TerrainState* const state_top      = &terrain_state[x         + y_plus_1  * constants->W];
	__global const TerrainState* const state_bot      = &terrain_state[x         + y_minus_1 * constants->W];
	__global       TerrainState* const state_middle   = &terrain_state[x         + y         * constants->W];

	const float L_h = state_left  ->height + ((process_deposited_sed != 0) ? state_left  ->deposited_sed_h : 0.0f);   // state_left ->sediment[0] + state_left ->sediment[1] + state_left ->sediment[2];// + state_left ->water;
	const float R_h = state_right ->height + ((process_deposited_sed != 0) ? state_right ->deposited_sed_h : 0.0f);   // state_right->sediment[0] + state_right->sediment[1] + state_right->sediment[2];// + state_right->water;
	const float B_h = state_bot   ->height + ((process_deposited_sed != 0) ? state_bot   ->deposited_sed_h : 0.0f);   // state_bot  ->sediment[0] + state_bot  ->sediment[1] + state_bot  ->sediment[2];// + state_bot  ->water;
	const float T_h = state_top   ->height + ((process_deposited_sed != 0) ? state_top   ->deposited_sed_h : 0.0f);   // state_top  ->sediment[0] + state_top  ->sediment[1] + state_top  ->sediment[2];// + state_top  ->water;
	const float   h = state_middle->height + ((process_deposited_sed != 0) ? state_middle->deposited_sed_h : 0.0f); // state_top  ->sediment[0] + state_top  ->sediment[1] + state_top  ->sediment[2];// + state_top  ->water;

	const float dh_dx = (R_h - L_h) * (0.5f * constants->recip_cell_w); // dh/dx = (R_h - L_h) / (2*cell_w) = (R_h - L_h) * 0.5 * (1/cell_w)
	const float dh_dy = (T_h - B_h) * (0.5f * constants->recip_cell_w);

	// Compute curvature (second deriv)
	const float d2_h_dx2 = (R_h - 2*h + L_h) / square(constants->cell_w); // (d^2u/dx^2, d^2v/dx^2)
	const float d2_h_dy2 = (T_h - 2*h + B_h) / square(constants->cell_w); // (d^2u/dy^2, d^2v/dy^2)

	state_middle->height_laplacian = d2_h_dx2 + d2_h_dy2;

	float2 thermal_vel = (float2)(0.0, 0.0);
	float thermal_move_vol = 0.0;

	float2 grad_h = (float2)(dh_dx, dh_dy);
	float grad_h_len = length(grad_h);
	if(grad_h_len > 1.0e-4f)
	{
		float2 unit_step_vec = -grad_h / grad_h_len;
		

		//float2 step_dir = -grad_h / grad_h_len; // normalised step vector

		// let step_dir = -grad / ||grad||
		// step_h = dot(step_dir, grad) = dot(-grad / ||grad||, grad) = -||grad||^2 / ||grad = -||grad||
		// with second derivs:
		// step_h = dot(step_dir, (d^2h/dx^2, d^2h/dy^2)) = (-dh/dx d^2h/dx^2 + -dh/dy d^2h/dy^2) / ||grad|| = 
		//float step_h = dot(step_dir, grad) + dot(step_dir, (float2)(d2_h_dx2, d2_h_dy2));
		//float step_delta_h = dot(step_dir, grad_h + (float2)(d2_h_dx2, d2_h_dy2));

		//float step_delta_h = dot(step_vec, grad_h) + (d2_h_dx2 * square(step_vec.x)
		float2 dv = unit_step_vec;// * constants->cell_w;
		float2 downhill_p = (float2)((float)x, (float)y) + dv;
	//	float step_delta_h = dv.x * dh_dx + dv.y * dh_dy + 0.5f * (square(dv.x) * d2_h_dx2 + square(dv.y) * d2_h_dy2);

		// Read sedimentation value at (old_x, old_y)
		const int old_xi = clamp((int)downhill_p.x, 0, constants->W-1);
		const int old_yi = clamp((int)downhill_p.y, 0, constants->H-1);
		const int old_xi1 = clamp((int)downhill_p.x + 1, 0, constants->W-1);
		const int old_yi1 = clamp((int)downhill_p.y + 1, 0, constants->H-1);

		const float t_x = downhill_p.x - (int)downhill_p.x;//(float)(int)floor_old_x;
		const float t_y = downhill_p.y - (int)downhill_p.y;//(float)(int)floor_old_y;

		 float downhill_h = biLerp(
			terrain_state[old_xi  + old_yi  * constants->W].height + ((process_deposited_sed != 0) ? terrain_state[old_xi  + old_yi  * constants->W].deposited_sed_h : 0.0f),
			terrain_state[old_xi1 + old_yi  * constants->W].height + ((process_deposited_sed != 0) ? terrain_state[old_xi1 + old_yi  * constants->W].deposited_sed_h : 0.0f),
			terrain_state[old_xi  + old_yi1 * constants->W].height + ((process_deposited_sed != 0) ? terrain_state[old_xi  + old_yi1 * constants->W].deposited_sed_h : 0.0f),
			terrain_state[old_xi1 + old_yi1 * constants->W].height + ((process_deposited_sed != 0) ? terrain_state[old_xi1 + old_yi1 * constants->W].deposited_sed_h : 0.0f),
			t_x, t_y);

		//const float3 normal = normalize((float3)(-dh_dx, -dh_dy, 1));

		//const float tan_slope_angle = sqrt(square(normal.x) + square(normal.y)) / normal.z;
		//const float tan_slope_angle = -step_delta_h;
		 const float tan_slope_angle = (h - downhill_h) / constants->cell_w;

		/*if(x == W/2 + 10 && y == W/2 + 10)
		{
			printf("----------------------\n");
			printf("(x, y): %f %f \n", (float)x, (float)y);
			printf("downhill_p: %f %f \n", downhill_p.x, downhill_p.y);
			printf("grad_h: %f %f \n", grad_h.x, grad_h.y);
			printf("unit_step_vec: %f %f \n", unit_step_vec.x, unit_step_vec.y);
			printf("h: %f \n", h);
			printf("downhill_h: %f \n", downhill_h);
			printf("tan_slope_angle: %f \n", tan_slope_angle);
		}*/

		//const float max_second_deriv = 0.1f;

		const float max_talus_angle = (process_deposited_sed != 0) ? constants->tan_max_deposited_talus_angle : constants->tan_max_talus_angle;
	
		if(tan_slope_angle > max_talus_angle/* && (d2_h_dx2 < max_second_deriv) && (d2_h_dy2 < max_second_deriv)*/)
		{
			// Move some material downhill
			thermal_vel = unit_step_vec; // In pixel coords

			float thermal_move_h = constants->delta_t * ((process_deposited_sed != 0) ? constants->K_tdep : constants->K_t);// * (tan_slope_angle - constants->tan_max_talus_angle);//min(1.0f, tan_slope_angle - constants->tan_max_talus_angle);// * sqrt(square(normal.x) + square(normal.y));

			if(process_deposited_sed != 0)
				thermal_move_h = min(thermal_move_h, state_middle->deposited_sed_h); // Make sure we don't move out more than present in this cell

			thermal_move_vol = thermal_move_h * square(constants->cell_w);

			// Note: (tan_slope_angle - constants->tan_max_talus_angle) factor introduces ridgeline artifacts on x and y axes.
		}

		/*if(x == W/2 + 10 && y == W/2 + 10)
		{
			printf("----------------------\n");
			printf("thermal_vel: %f %f \n", thermal_vel.x, thermal_vel.y);
		}*/
	}

	state_middle->thermal_vel = thermal_vel;
	state_middle->thermal_move_vol = thermal_move_vol;

#else
	__global const TerrainState* const state_0      = &terrain_state[x_minus_1 + y_plus_1  * W];
	__global const TerrainState* const state_1      = &terrain_state[x         + y_plus_1  * W];
	__global const TerrainState* const state_2      = &terrain_state[x_plus_1  + y_plus_1  * W];
	__global const TerrainState* const state_3      = &terrain_state[x_minus_1 + y         * W];
	__global const TerrainState* const state_middle = &terrain_state[x         + y         * W];
	__global const TerrainState* const state_4      = &terrain_state[x_plus_1  + y         * W];
	__global const TerrainState* const state_5      = &terrain_state[x_minus_1 + y_minus_1 * W];
	__global const TerrainState* const state_6      = &terrain_state[x         + y_minus_1 * W];
	__global const TerrainState* const state_7      = &terrain_state[x_plus_1  + y_minus_1 * W];

	__global       ThermalErosionState* const thermal_erosion_state_middle = &thermal_erosion_state[x         + y          *W];

	const float middle_h = state_middle->height;// + state_middle->deposited_sed;
	float h_0 = middle_h - state_0->height/* + state_0->deposited_sed*/; // height diff between adjacent cell and middle cell
	float h_1 = middle_h - state_1->height/* + state_1->deposited_sed*/;
	float h_2 = middle_h - state_2->height/* + state_2->deposited_sed*/;
	float h_3 = middle_h - state_3->height/* + state_3->deposited_sed*/;
	float h_4 = middle_h - state_4->height/* + state_4->deposited_sed*/;
	float h_5 = middle_h - state_5->height/* + state_5->deposited_sed*/;
	float h_6 = middle_h - state_6->height/* + state_6->deposited_sed*/;
	float h_7 = middle_h - state_7->height/* + state_7->deposited_sed*/;

	const float max_height_diff = // H
		max(
			max(
				max(h_0, h_1),
				max(h_2, h_3)
			),
			max(
				max(h_4, h_5),
				max(h_6, h_7)
			)
		);

	// tan(theta) = h / cell_w        [for immediately adjacent cells]
	const float tan_angle_0 = h_0 * constants->recip_cell_w * (1 / sqrt(2.f));
	const float tan_angle_1 = h_1 * constants->recip_cell_w;
	const float tan_angle_2 = h_2 * constants->recip_cell_w * (1 / sqrt(2.f));
	const float tan_angle_3 = h_3 * constants->recip_cell_w;
	const float tan_angle_4 = h_4 * constants->recip_cell_w;
	const float tan_angle_5 = h_5 * constants->recip_cell_w * (1 / sqrt(2.f));
	const float tan_angle_6 = h_6 * constants->recip_cell_w;
	const float tan_angle_7 = h_7 * constants->recip_cell_w * (1 / sqrt(2.f));

	const float tan_max_talus_angle = constants->tan_max_talus_angle;

	if(tan_angle_0 < tan_max_talus_angle) h_0 = 0;
	if(tan_angle_1 < tan_max_talus_angle) h_1 = 0;
	if(tan_angle_2 < tan_max_talus_angle) h_2 = 0;
	if(tan_angle_3 < tan_max_talus_angle) h_3 = 0;
	if(tan_angle_4 < tan_max_talus_angle) h_4 = 0;
	if(tan_angle_5 < tan_max_talus_angle) h_5 = 0;
	if(tan_angle_6 < tan_max_talus_angle) h_6 = 0;
	if(tan_angle_7 < tan_max_talus_angle) h_7 = 0;

	// Total height difference, for cells for which the height difference exceeds the max talus angle
	/*const float total_height_diff = 
		((tan_angle_0 > tan_max_talus_angle) ? h_0 : 0.0) + 
		((tan_angle_1 > tan_max_talus_angle) ? h_1 : 0.0) + 
		((tan_angle_2 > tan_max_talus_angle) ? h_2 : 0.0) + 
		((tan_angle_3 > tan_max_talus_angle) ? h_3 : 0.0) + 
		((tan_angle_4 > tan_max_talus_angle) ? h_4 : 0.0) + 
		((tan_angle_5 > tan_max_talus_angle) ? h_5 : 0.0) + 
		((tan_angle_6 > tan_max_talus_angle) ? h_6 : 0.0) + 
		((tan_angle_7 > tan_max_talus_angle) ? h_7 : 0.0);*/
	const float total_height_diff = 
		h_0 + 
		h_1 + 
		h_2 + 
		h_3 + 
		h_4 + 
		h_5 + 
		h_6 + 
		h_7;

	

	const float a = square(constants->cell_w); // cell area
	const float R = 1.0f; // hardness
	float common_factors;
	if(max_height_diff > 0 && total_height_diff > 0)
	{
		const float norm_factor = 1.f / total_height_diff;
		common_factors = norm_factor * a * constants->delta_t * constants->K_t * R * max_height_diff * 0.5f;
	}
	else
		common_factors = 0;

	thermal_erosion_state_middle->flux[0] = h_0 * common_factors;
	thermal_erosion_state_middle->flux[1] = h_1 * common_factors;
	thermal_erosion_state_middle->flux[2] = h_2 * common_factors;
	thermal_erosion_state_middle->flux[3] = h_3 * common_factors;
	thermal_erosion_state_middle->flux[4] = h_4 * common_factors;
	thermal_erosion_state_middle->flux[5] = h_5 * common_factors;
	thermal_erosion_state_middle->flux[6] = h_6 * common_factors;
	thermal_erosion_state_middle->flux[7] = h_7 * common_factors;
#endif
}


// Sets flux in thermal_erosion_state
__kernel void thermalErosionDepositedFluxKernel(
	__global TerrainState* restrict const terrain_state, 
	__global ThermalErosionState* restrict const thermal_erosion_state, 
	__constant Constants* restrict const constants
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int x_minus_1 = max(x-1, 0);
	const int x_plus_1  = min(x+1, constants->W-1);
	const int y_minus_1 = max(y-1, 0);
	const int y_plus_1  = min(y+1, constants->H-1);

#if 1
	__global const TerrainState* const state_left     = &terrain_state[x_minus_1 + y         * constants->W];
	__global const TerrainState* const state_right    = &terrain_state[x_plus_1  + y         * constants->W];
	__global const TerrainState* const state_top      = &terrain_state[x         + y_plus_1  * constants->W];
	__global const TerrainState* const state_bot      = &terrain_state[x         + y_minus_1 * constants->W];
	__global       TerrainState* const state_middle   = &terrain_state[x         + y         * constants->W];

	const float L_h = state_left  ->height + state_left  ->deposited_sed_h; // state_left ->sediment[0] + state_left ->sediment[1] + state_left ->sediment[2];// + state_left ->water;
	const float R_h = state_right ->height + state_right ->deposited_sed_h; // state_right->sediment[0] + state_right->sediment[1] + state_right->sediment[2];// + state_right->water;
	const float B_h = state_bot   ->height + state_bot   ->deposited_sed_h; // state_bot  ->sediment[0] + state_bot  ->sediment[1] + state_bot  ->sediment[2];// + state_bot  ->water;
	const float T_h = state_top   ->height + state_top   ->deposited_sed_h; // state_top  ->sediment[0] + state_top  ->sediment[1] + state_top  ->sediment[2];// + state_top  ->water;
	const float   h = state_middle->height + state_middle->deposited_sed_h; // state_top  ->sediment[0] + state_top  ->sediment[1] + state_top  ->sediment[2];// + state_top  ->water;

	// Compute first derivs of height
	const float dh_dx = (R_h - L_h) * (0.5f * constants->recip_cell_w); // dh/dx = (R_h - L_h) / (2*cell_w) = (R_h - L_h) * 0.5 * (1/cell_w)
	const float dh_dy = (T_h - B_h) * (0.5f * constants->recip_cell_w);

	// Compute curvature (second deriv)
	const float d2_h_dx2 = (R_h - 2*h + L_h) / square(constants->cell_w); // (d^2u/dx^2, d^2v/dx^2)
	const float d2_h_dy2 = (T_h - 2*h + B_h) / square(constants->cell_w); // (d^2u/dy^2, d^2v/dy^2)

	state_middle->height_laplacian = d2_h_dx2 + d2_h_dy2;

	float2 thermal_vel = (float2)(0.0, 0.0);
	float thermal_move_vol = 0.0;

	float2 grad_h = (float2)(dh_dx, dh_dy);
	float grad_h_len = length(grad_h);
	if(grad_h_len > 1.0e-4f)
	{
		float2 unit_step_vec = -grad_h / grad_h_len;
		

		//float2 step_dir = -grad_h / grad_h_len; // normalised step vector

		// let step_dir = -grad / ||grad||
		// step_h = dot(step_dir, grad) = dot(-grad / ||grad||, grad) = -||grad||^2 / ||grad = -||grad||
		// with second derivs:
		// step_h = dot(step_dir, (d^2h/dx^2, d^2h/dy^2)) = (-dh/dx d^2h/dx^2 + -dh/dy d^2h/dy^2) / ||grad|| = 
		//float step_h = dot(step_dir, grad) + dot(step_dir, (float2)(d2_h_dx2, d2_h_dy2));
		//float step_delta_h = dot(step_dir, grad_h + (float2)(d2_h_dx2, d2_h_dy2));

		//float step_delta_h = dot(step_vec, grad_h) + (d2_h_dx2 * square(step_vec.x)
		float2 dv = unit_step_vec;// * constants->cell_w;
		float2 downhill_p = (float2)((float)x, (float)y) + dv;
	//	float step_delta_h = dv.x * dh_dx + dv.y * dh_dy + 0.5f * (square(dv.x) * d2_h_dx2 + square(dv.y) * d2_h_dy2);

		// Read sedimentation value at (old_x, old_y)
		const int old_xi = clamp((int)downhill_p.x, 0, constants->W-1);
		const int old_yi = clamp((int)downhill_p.y, 0, constants->H-1);
		const int old_xi1 = clamp((int)downhill_p.x + 1, 0, constants->W-1);
		const int old_yi1 = clamp((int)downhill_p.y + 1, 0, constants->H-1);

		const float t_x = downhill_p.x - (int)downhill_p.x;//(float)(int)floor_old_x;
		const float t_y = downhill_p.y - (int)downhill_p.y;//(float)(int)floor_old_y;

		float downhill_h = biLerp(
			terrain_state[old_xi  + old_yi  * constants->W].height,
			terrain_state[old_xi1 + old_yi  * constants->W].height,
			terrain_state[old_xi  + old_yi1 * constants->W].height,
			terrain_state[old_xi1 + old_yi1 * constants->W].height,
			t_x, t_y);

		//const float3 normal = normalize((float3)(-dh_dx, -dh_dy, 1));

		//const float tan_slope_angle = sqrt(square(normal.x) + square(normal.y)) / normal.z;
		//const float tan_slope_angle = -step_delta_h;
		 const float tan_slope_angle = (h - downhill_h) / constants->cell_w;

		/*if(x == constants->W/2 + 10 && y == constants->W/2 + 10)
		{
			printf("----------------------\n");
			printf("(x, y): %f %f \n", (float)x, (float)y);
			printf("downhill_p: %f %f \n", downhill_p.x, downhill_p.y);
			printf("grad_h: %f %f \n", grad_h.x, grad_h.y);
			printf("unit_step_vec: %f %f \n", unit_step_vec.x, unit_step_vec.y);
			printf("h: %f \n", h);
			printf("downhill_h: %f \n", downhill_h);
			printf("tan_slope_angle: %f \n", tan_slope_angle);
		}*/

		//const float max_second_deriv = 0.1f;
	
		if(tan_slope_angle > constants->tan_max_deposited_talus_angle/* && (d2_h_dx2 < max_second_deriv) && (d2_h_dy2 < max_second_deriv)*/)
		{
			// Move some material downhill
			thermal_vel = unit_step_vec;//(float2)(normal.x, normal.y) * 1.f; // TEMP HACK
			thermal_move_vol = square(constants->cell_w) * constants->delta_t * constants->K_t;// * (tan_slope_angle - constants->tan_max_talus_angle);//min(1.0f, tan_slope_angle - constants->tan_max_talus_angle);// * sqrt(square(normal.x) + square(normal.y));

			// Note: (tan_slope_angle - constants->tan_max_talus_angle) factor introduces ridgeline artifacts on x and y axes.
		}

	/*	if(x == constants->W/2 + 10 && y == constants->W/2 + 10)
		{
			printf("----------------------\n");
			printf("thermal_vel: %f %f \n", thermal_vel.x, thermal_vel.y);
		}*/
	}

	state_middle->thermal_vel = thermal_vel;
	state_middle->thermal_move_vol = thermal_move_vol;





	//const float3 normal = normalize((float3)(-dh_dx, -dh_dy, 1));

	//const float tan_slope_angle = sqrt(square(normal.x) + square(normal.y)) / normal.z;

	//const float max_second_deriv = 0.1f;
	//float2 thermal_vel;
	//float thermal_move_vol;
	//if(0) // TEMP HACK tan_slope_angle > constants->tan_max_talus_angle/* && (d2_h_dx2 < max_second_deriv) && (d2_h_dy2 < max_second_deriv)*/)
	//{
	//	// Move some material downhill
	//	thermal_vel = (float2)(normal.x, normal.y) * 1.f; // TEMP HACK
	//	thermal_move_vol = min(
	//		state_middle->deposited_sed, // Make sure we don't move out more than present in this cell
	//		1000.0f * square(constants->cell_w) * constants->delta_t * constants->K_tdep * min(1.0f, tan_slope_angle - constants->tan_max_deposited_talus_angle)// * sqrt(square(normal.x) + square(normal.y));
	//	);
	//}
	//else
	//{
	//	thermal_vel = (float2)(0.0, 0.0);
	//	thermal_move_vol = 0.0; // constants->delta_t * constants->K_t * sqrt(square(normal.x) + square(normal.y));
	//}

	//state_middle->thermal_vel = thermal_vel;
	//state_middle->thermal_move_vol = thermal_move_vol;

#else

	__global const TerrainState* const state_0      = &terrain_state[x_minus_1 + y_plus_1  * W];
	__global const TerrainState* const state_1      = &terrain_state[x         + y_plus_1  * W];
	__global const TerrainState* const state_2      = &terrain_state[x_plus_1  + y_plus_1  * W];
	__global const TerrainState* const state_3      = &terrain_state[x_minus_1 + y         * W];
	__global const TerrainState* const state_middle = &terrain_state[x         + y         * W];
	__global const TerrainState* const state_4      = &terrain_state[x_plus_1  + y         * W];
	__global const TerrainState* const state_5      = &terrain_state[x_minus_1 + y_minus_1 * W];
	__global const TerrainState* const state_6      = &terrain_state[x         + y_minus_1 * W];
	__global const TerrainState* const state_7      = &terrain_state[x_plus_1  + y_minus_1 * W];

	__global       ThermalErosionState* const thermal_erosion_state_middle = &thermal_erosion_state[x         + y          *W];


	const float middle_h = state_middle->height + state_middle->deposited_sed; // state_middle->sediment[0] + state_middle->sediment[1] + state_middle->sediment[2];
	float h_0 = middle_h - (state_0->height + state_0->deposited_sed);// state_0->sediment[0] + state_0->sediment[1] + state_0->sediment[2]); // height diff between adjacent cell and middle cell
	float h_1 = middle_h - (state_1->height + state_1->deposited_sed);// state_1->sediment[0] + state_1->sediment[1] + state_1->sediment[2]);
	float h_2 = middle_h - (state_2->height + state_2->deposited_sed);// state_2->sediment[0] + state_2->sediment[1] + state_2->sediment[2]);
	float h_3 = middle_h - (state_3->height + state_3->deposited_sed);// state_3->sediment[0] + state_3->sediment[1] + state_3->sediment[2]);
	float h_4 = middle_h - (state_4->height + state_4->deposited_sed);// state_4->sediment[0] + state_4->sediment[1] + state_4->sediment[2]);
	float h_5 = middle_h - (state_5->height + state_5->deposited_sed);// state_5->sediment[0] + state_5->sediment[1] + state_5->sediment[2]);
	float h_6 = middle_h - (state_6->height + state_6->deposited_sed);// state_6->sediment[0] + state_6->sediment[1] + state_6->sediment[2]);
	float h_7 = middle_h - (state_7->height + state_7->deposited_sed);// state_7->sediment[0] + state_7->sediment[1] + state_7->sediment[2]);

	const float max_height_diff = // H
		max(
			max(
				max(h_0, h_1),
				max(h_2, h_3)
			),
			max(
				max(h_4, h_5),
				max(h_6, h_7)
			)
		);

	const float tan_angle_0 = h_0 * constants->recip_cell_w * (1 / sqrt(2.f));
	const float tan_angle_1 = h_1 * constants->recip_cell_w;
	const float tan_angle_2 = h_2 * constants->recip_cell_w * (1 / sqrt(2.f));
	const float tan_angle_3 = h_3 * constants->recip_cell_w;
	const float tan_angle_4 = h_4 * constants->recip_cell_w;
	const float tan_angle_5 = h_5 * constants->recip_cell_w * (1 / sqrt(2.f));
	const float tan_angle_6 = h_6 * constants->recip_cell_w;
	const float tan_angle_7 = h_7 * constants->recip_cell_w * (1 / sqrt(2.f));

	const float tan_max_talus_angle = constants->tan_max_deposited_talus_angle;

	if(tan_angle_0 < tan_max_talus_angle) h_0 = 0;
	if(tan_angle_1 < tan_max_talus_angle) h_1 = 0;
	if(tan_angle_2 < tan_max_talus_angle) h_2 = 0;
	if(tan_angle_3 < tan_max_talus_angle) h_3 = 0;
	if(tan_angle_4 < tan_max_talus_angle) h_4 = 0;
	if(tan_angle_5 < tan_max_talus_angle) h_5 = 0;
	if(tan_angle_6 < tan_max_talus_angle) h_6 = 0;
	if(tan_angle_7 < tan_max_talus_angle) h_7 = 0;

	// Total height difference, for cells for which the height difference exceeds the max talus angle
	/*const float total_height_diff = 
		((tan_angle_0 > tan_max_talus_angle) ? h_0 : 0.0) + 
		((tan_angle_1 > tan_max_talus_angle) ? h_1 : 0.0) + 
		((tan_angle_2 > tan_max_talus_angle) ? h_2 : 0.0) + 
		((tan_angle_3 > tan_max_talus_angle) ? h_3 : 0.0) + 
		((tan_angle_4 > tan_max_talus_angle) ? h_4 : 0.0) + 
		((tan_angle_5 > tan_max_talus_angle) ? h_5 : 0.0) + 
		((tan_angle_6 > tan_max_talus_angle) ? h_6 : 0.0) + 
		((tan_angle_7 > tan_max_talus_angle) ? h_7 : 0.0);*/
	const float total_height_diff = 
		h_0 + 
		h_1 + 
		h_2 + 
		h_3 + 
		h_4 + 
		h_5 + 
		h_6 + 
		h_7;

	const float a = square(constants->cell_w); // cell area
	const float R = 1.f; // hardness TEMP
	float common_factors;
	if(max_height_diff > 0 && total_height_diff > 0)
	{
		const float norm_factor = 1.f / total_height_diff;
		common_factors = norm_factor * a * constants->delta_t * constants->K_tdep * R * max_height_diff * 0.5f;
	}
	else
		common_factors = 0;

	/*float sum_flux = 
		h_0 * common_factors +
		h_1 * common_factors +
		h_2 * common_factors +
		h_3 * common_factors +
		h_4 * common_factors +
		h_5 * common_factors +
		h_6 * common_factors +
		h_7 * common_factors;*/
	float sum_flux = total_height_diff * common_factors;

	const float cur_deposited_sed_h = state_middle->deposited_sed;

	float K = 1;
	if(cur_deposited_sed_h > 0)
	{
		if(sum_flux > cur_deposited_sed_h)
		{
			K = cur_deposited_sed_h / sum_flux;
		}
	}
	else
		K = 0;

	thermal_erosion_state_middle->flux[0] = h_0 * common_factors * K;
	thermal_erosion_state_middle->flux[1] = h_1 * common_factors * K;
	thermal_erosion_state_middle->flux[2] = h_2 * common_factors * K;
	thermal_erosion_state_middle->flux[3] = h_3 * common_factors * K;
	thermal_erosion_state_middle->flux[4] = h_4 * common_factors * K;
	thermal_erosion_state_middle->flux[5] = h_5 * common_factors * K;
	thermal_erosion_state_middle->flux[6] = h_6 * common_factors * K;
	thermal_erosion_state_middle->flux[7] = h_7 * common_factors * K;
#endif
}


/*

0      1      2


3      x      4


5      6      7

*/
__kernel void thermalErosionMovementKernel(
	__global const ThermalErosionState* restrict const thermal_erosion_state, 
	__global       TerrainState* restrict const terrain_state, 
	__constant Constants* restrict const constants,
	int process_deposited_sed
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int x_minus_1 = max(x-1, 0);
	const int x_plus_1  = min(x+1, constants->W-1);
	const int y_minus_1 = max(y-1, 0);
	const int y_plus_1  = min(y+1, constants->H-1);

#if 1
	__global       TerrainState* const state_middle   = &terrain_state[x         + y          *constants->W];

	// Loop over neighbouring cells
	float2 water_pos = (float2)(0.f, 0.f);
	float2 water_vel = (float2)(0.f, 0.f);
	float in_thermal_move_vol = 0.f; // Total volume of solid moved into this cell in this timestep
	//float sediment_vol = 0.f;
	for(int ny = y-1; ny <= y+1; ny++)
	for(int nx = x-1; nx <= x+1; nx++)
	{
		if(nx >= 0 && nx < constants->W && ny >= 0 && ny < constants->H)
		{
			__global const TerrainState* const n_state = &terrain_state[nx + ny * constants->W];
			float2 cell_n_thermal_pos = (float2)((float)nx, (float)ny); // source position for cell n
			float2 new_pos = cell_n_thermal_pos + n_state->thermal_vel * constants->delta_t;

			/*if(x == W/2-20 && y == W/2-20)
			{
				printf("nx, ny: %f, %f \n", (float)nx, (float)ny);
				printf("n_state->thermal_vel: %f, %f \n", n_state->thermal_vel.x, n_state->thermal_vel.y);
				printf("n_state->thermal_move_vol: %f \n", n_state->thermal_move_vol);
			}*/
			
			float3 ovrlp = overlap(new_pos, (float2)((float)x, (float)y), /*diffusion_radius=*/0.5f); // compute overlap with this cell
			float overlapRelativeArea = ovrlp.z;
			float2 overlapCenterOfMass = ovrlp.xy;
			float overlapVol = overlapRelativeArea * n_state->thermal_move_vol; // compute vol of mass moved from cell n into this cell

			in_thermal_move_vol    += overlapVol;
			//water_pos    += overlapCenterOfMass    * overlapMass; //add the overlap center weighted by mass
			//water_vel    += n_state->water_vel     * overlapMass; //add the particle velocity weighted by overlap mass(momentum)
			//sediment_vol += n_state->suspended_vol * overlapMass; // Add the particle sediment_vol weighted by overlap mass
		}
	}

	//normalize
	//if(water_mass > 0.0) //if not vacuum
	//{
	//  water_pos /= water_mass; //center of mass
	//  water_vel /= water_mass; //average velocity
	//  sediment_vol /= water_mass;
	//}

	// Write to state array
	//state_middle->water_position = water_pos;
	//state_middle->water_vel = water_vel;
	//state_middle->water_mass = water_mass;
	//state_middle->suspended_vol = sediment_vol;
	const float delta_vol = in_thermal_move_vol - state_middle->thermal_move_vol;

	if(process_deposited_sed != 0)
		state_middle->deposited_sed_h += delta_vol / square(constants->cell_w);
	else
		state_middle->height          += delta_vol / square(constants->cell_w);


	// Apply height_laplacian smoothing
	if(process_deposited_sed == 0)
	{
		if(fabs(state_middle->height_laplacian) > constants->laplacian_threshold)
			state_middle->height += state_middle->height_laplacian * constants->K_smooth * constants->delta_t;
	}

#else
	__global const ThermalErosionState* const state_0      = &thermal_erosion_state[x_minus_1 + y_plus_1  * W];
	__global const ThermalErosionState* const state_1      = &thermal_erosion_state[x         + y_plus_1  * W];
	__global const ThermalErosionState* const state_2      = &thermal_erosion_state[x_plus_1  + y_plus_1  * W];
	__global const ThermalErosionState* const state_3      = &thermal_erosion_state[x_minus_1 + y         * W];
	__global const ThermalErosionState* const state_middle = &thermal_erosion_state[x         + y         * W];
	__global const ThermalErosionState* const state_4      = &thermal_erosion_state[x_plus_1  + y         * W];
	__global const ThermalErosionState* const state_5      = &thermal_erosion_state[x_minus_1 + y_minus_1 * W];
	__global const ThermalErosionState* const state_6      = &thermal_erosion_state[x         + y_minus_1 * W];
	__global const ThermalErosionState* const state_7      = &thermal_erosion_state[x_plus_1  + y_minus_1 * W];

	__global       TerrainState* const middle_terrain_state = &terrain_state[x         + y          *W];

	float flux_0 = state_0->flux[7]; // Flux_0 = flux from cell located up and to the left of this one, in the down and right direction.
	float flux_1 = state_1->flux[6];
	float flux_2 = state_2->flux[5];
	float flux_3 = state_3->flux[4];
	float flux_4 = state_4->flux[3];
	float flux_5 = state_5->flux[2];
	float flux_6 = state_6->flux[1];
	float flux_7 = state_7->flux[0];

	if(x == 0) // If this cell is on the left edge:
	{
		flux_0 = flux_3 = flux_5 = 0; // Zero flux coming from cells located to the left.
	}
	else if(x == W - 1)
	{
		flux_2 = flux_4 = flux_7 = 0;
	}

	if(y == 0) // If this cell is on the bottom edge:
	{
		flux_5 = flux_6 = flux_7 = 0; // Zero flux coming from cells located to the bottom.
	}
	else if(y == W - 1) // If this cell is on the top edge:
	{
		flux_0 = flux_1 = flux_2 = 0;
	}

	const float sum_material_in = 
		flux_0 +
		flux_1 +
		flux_2 +
		flux_3 +
		flux_4 +
		flux_5 +
		flux_6 +
		flux_7;

	const float sum_material_out = 
		state_middle->flux[0] + 
		state_middle->flux[1] + 
		state_middle->flux[2] + 
		state_middle->flux[3] + 
		state_middle->flux[4] + 
		state_middle->flux[5] + 
		state_middle->flux[6] + 
		state_middle->flux[7];

	const float net_material_change = sum_material_in - sum_material_out;

	middle_terrain_state->height += net_material_change;
#endif
}


__kernel void thermalErosionDepositedMovementKernel(
	__global const ThermalErosionState* restrict const thermal_erosion_state, 
	__global       TerrainState* restrict const terrain_state, 
	__constant Constants* restrict const constants
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int x_minus_1 = max(x-1, 0);
	const int x_plus_1  = min(x+1, constants->W-1);
	const int y_minus_1 = max(y-1, 0);
	const int y_plus_1  = min(y+1, constants->H-1);

#if 1

	__global       TerrainState* const state_middle   = &terrain_state[x         + y          *constants->W];

	// Loop over neighbouring cells
	float2 water_pos = (float2)(0.f, 0.f);
	float2 water_vel = (float2)(0.f, 0.f);
	float in_thermal_move_deposited_vol = 0.f; // Total volume of solid moved into this cell in this timestep
	for(int ny = y-1; ny <= y+1; ny++)
	for(int nx = x-1; nx <= x+1; nx++)
	{
		if(nx >= 0 && nx < constants->W && ny >= 0 && ny < constants->H)
		{
			__global const TerrainState* const n_state = &terrain_state[nx + ny * constants->W];
			float2 cell_n_thermal_pos = (float2)((float)nx, (float)ny); // source position for cell n
			float2 new_pos = cell_n_thermal_pos + n_state->thermal_vel * constants->delta_t;

			/*if(x == W/2-20 && y == W/2-20)
			{
				printf("nx, ny: %f, %f \n", (float)nx, (float)ny);
				printf("n_state->thermal_vel: %f, %f \n", n_state->thermal_vel.x, n_state->thermal_vel.y);
				printf("n_state->thermal_move_vol: %f \n", n_state->thermal_move_vol);
			}*/
			
			float3 ovrlp = overlap(new_pos, (float2)((float)x, (float)y), /*diffusion_radius=*/0.5f); // compute overlap with this cell
			float overlapRelativeArea = ovrlp.z;
			float2 overlapCenterOfMass = ovrlp.xy;
			float overlapVol = overlapRelativeArea * n_state->thermal_move_vol; // compute vol of deposited sediment mass moved from cell n into this cell

			in_thermal_move_deposited_vol    += overlapVol;
			//water_pos    += overlapCenterOfMass    * overlapMass; //add the overlap center weighted by mass
			//water_vel    += n_state->water_vel     * overlapMass; //add the particle velocity weighted by overlap mass(momentum)
			//sediment_vol += n_state->suspended_vol * overlapMass; // Add the particle sediment_vol weighted by overlap mass
		}
	}

	const float delta_vol = in_thermal_move_deposited_vol - state_middle->thermal_move_vol;
	state_middle->deposited_sed_h += delta_vol / square(constants->cell_w);


	// Apply height_laplacian smoothing
//TEMP	state_middle->deposited_sed = max(0.f, state_middle->deposited_sed + state_middle->height_laplacian * constants->K_smooth * constants->delta_t);

#else

	__global const ThermalErosionState* const state_0      = &thermal_erosion_state[x_minus_1 + y_plus_1  * W];
	__global const ThermalErosionState* const state_1      = &thermal_erosion_state[x         + y_plus_1  * W];
	__global const ThermalErosionState* const state_2      = &thermal_erosion_state[x_plus_1  + y_plus_1  * W];
	__global const ThermalErosionState* const state_3      = &thermal_erosion_state[x_minus_1 + y         * W];
	__global const ThermalErosionState* const state_middle = &thermal_erosion_state[x         + y         * W];
	__global const ThermalErosionState* const state_4      = &thermal_erosion_state[x_plus_1  + y         * W];
	__global const ThermalErosionState* const state_5      = &thermal_erosion_state[x_minus_1 + y_minus_1 * W];
	__global const ThermalErosionState* const state_6      = &thermal_erosion_state[x         + y_minus_1 * W];
	__global const ThermalErosionState* const state_7      = &thermal_erosion_state[x_plus_1  + y_minus_1 * W];

	__global       TerrainState* const middle_terrain_state = &terrain_state[x         + y          *W];

	float flux_0 = state_0->flux[7]; // Flux_0 = flux from cell located up and to the left of this one, in the down and right direction.
	float flux_1 = state_1->flux[6];
	float flux_2 = state_2->flux[5];
	float flux_3 = state_3->flux[4];
	float flux_4 = state_4->flux[3];
	float flux_5 = state_5->flux[2];
	float flux_6 = state_6->flux[1];
	float flux_7 = state_7->flux[0];

	if(x == 0) // If this cell is on the left edge:
	{
		flux_0 = flux_3 = flux_5 = 0; // Zero flux coming from cells located to the left.
	}
	else if(x == W - 1)
	{
		flux_2 = flux_4 = flux_7 = 0;
	}

	if(y == 0) // If this cell is on the bottom edge:
	{
		flux_5 = flux_6 = flux_7 = 0; // Zero flux coming from cells located to the bottom.
	}
	else if(y == W - 1) // If this cell is on the top edge:
	{
		flux_0 = flux_1 = flux_2 = 0;
	}

	const float sum_material_in = 
		flux_0 +
		flux_1 +
		flux_2 +
		flux_3 +
		flux_4 +
		flux_5 +
		flux_6 +
		flux_7;

	const float sum_material_out = 
		state_middle->flux[0] + 
		state_middle->flux[1] + 
		state_middle->flux[2] + 
		state_middle->flux[3] + 
		state_middle->flux[4] + 
		state_middle->flux[5] + 
		state_middle->flux[6] + 
		state_middle->flux[7];

	const float net_material_change = sum_material_in - sum_material_out;

	middle_terrain_state->deposited_sed += net_material_change;
#endif
}



// evaporation kernel.  Updates 'water' in terrain_state
__kernel void evaporationKernel(
	__global       TerrainState* restrict const terrain_state, 
	__constant Constants* restrict const constants
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	__global       TerrainState* const state_middle   = &terrain_state[x         + y          *constants->W];

	

	const float old_water_depth = waterHeightForMass(state_middle->water_mass, constants);

	const float d_new = old_water_depth * (1 - constants->K_e * constants->delta_t);// evaporation rate depends on water depth.  Makes no physical sense but useful.
	//const float d_new = max(0.f, old_water_depth - constants->K_e * constants->delta_t); // Make evaporation rate not depend on water depth

	const float new_water_mass = waterMassForHeight(d_new, constants);

	state_middle->water_mass = new_water_mass;
}


typedef struct
{
	float pos[3];
	float normal[3];
	float uv[2];
} Vertex;

inline float totalTerrainHeight(__global TerrainState* restrict const state, bool include_water, __constant Constants* restrict const constants)
{
	return state->height + state->deposited_sed_h + (include_water ? waterHeightForMass(state->water_mass, constants) : 0.f);
}

__kernel void setHeightFieldMeshKernel(
	__global       TerrainState* restrict const terrain_state, 
	__constant Constants* restrict const constants,
	__global Vertex* restrict const vertex_buffer,
	unsigned int vertex_buffer_offset_B,
	write_only image2d_t terrain_texture,
	__global Vertex* restrict const water_vertex_buffer,
	unsigned int water_vertex_buffer_offset_B
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	__global Vertex*       mesh_vert_0 = &      vertex_buffer[      vertex_buffer_offset_B / sizeof(Vertex)];
	__global Vertex* water_mesh_vert_0 = &water_vertex_buffer[water_vertex_buffer_offset_B / sizeof(Vertex)];

	int vert_xres = max(2, constants->W);
	int vert_yres = max(2, constants->H);
	int quad_xres = vert_xres - 1; // Number of quads in x and y directions
	int quad_yres = vert_yres - 1; // Number of quads in x and y directions

	__global Vertex*       mesh_vert = &      mesh_vert_0[x + y * vert_xres];
	__global Vertex* water_mesh_vert = &water_mesh_vert_0[x + y * vert_xres];

	float quad_w_x = constants->cell_w; // Width in metres of each quad
	float quad_w_y = quad_w_x;
	if(constants->H <= 4)
	{
		//quad_w_y *= 20.f; // For height = 1 (1-d debugging case), display strip a bit wider
	}

	const int max_src_x = constants->W - 1;
	const int max_src_y = constants->H - 1; // Store these so we can handle width 1 sims

	const float p_x = x * quad_w_x;
	const float p_y = y * quad_w_y;
	const float dx = quad_w_x;
	const float dy = quad_w_y;

	const int src_x = min(x, max_src_x);
	const int src_y = min(y, max_src_y);
	const int src_x_1 = min(x + 1, max_src_x);
	const int src_y_1 = min(y + 1, max_src_y);
	
	{
		const float z    = totalTerrainHeight(&terrain_state[src_x   + src_y  *constants->W], constants->include_water_height, constants);
		const float z_dx = totalTerrainHeight(&terrain_state[src_x_1 + src_y  *constants->W], constants->include_water_height, constants);
		const float z_dy = totalTerrainHeight(&terrain_state[src_x   + src_y_1*constants->W], constants->include_water_height, constants);

		const float3 p_dx_minus_p = (float3)(dx, 0, z_dx - z); // p(p_x + dx, dy) - p(p_x, p_y) = (p_x + dx, d_y, z_dx) - (p_x, p_y, z) = (d_x, 0, z_dx - z)
		const float3 p_dy_minus_p = (float3)(0, dy, z_dy - z);

		const float3 normal = normalize(cross(p_dx_minus_p, p_dy_minus_p));

		mesh_vert->pos[0] = p_x;
		mesh_vert->pos[1] = p_y;
		mesh_vert->pos[2] = z;

		mesh_vert->normal[0] = normal.x;
		mesh_vert->normal[1] = normal.y;
		mesh_vert->normal[2] = normal.z;
	}


	// Set water mesh
	{
		const float z    = totalTerrainHeight(&terrain_state[src_x   + src_y  *constants->W], /*include_water=*/true, constants); // constants->include_water_height);
		const float z_dx = totalTerrainHeight(&terrain_state[src_x_1 + src_y  *constants->W], /*include_water=*/true, constants); // constants->include_water_height);
		const float z_dy = totalTerrainHeight(&terrain_state[src_x   + src_y_1*constants->W], /*include_water=*/true, constants); // constants->include_water_height);

		const float3 p_dx_minus_p = (float3)(dx, 0, z_dx - z); // p(p_x + dx, dy) - p(p_x, p_y) = (p_x + dx, d_y, z_dx) - (p_x, p_y, z) = (d_x, 0, z_dx - z)
		const float3 p_dy_minus_p = (float3)(0, dy, z_dy - z);

		const float3 normal = normalize(cross(p_dx_minus_p, p_dy_minus_p));

		water_mesh_vert->pos[0] = p_x;
		water_mesh_vert->pos[1] = p_y;
		water_mesh_vert->pos[2] = z + constants->water_z_bias;

		water_mesh_vert->normal[0] = normal.x;
		water_mesh_vert->normal[1] = normal.y;
		water_mesh_vert->normal[2] = normal.z;
	}

	// Write to terrain texture
	//const float3 rock_col = pow((float3)(64.0 / 255.0, 60.0 / 255.0, 45 / 255.0), 2.2); // brown
	//const float3 deposited_col = pow((float3)(103 / 255.0, 91 / 255.0, 67 / 255.0), 2.2); // lighter orange brown
	//const float3 deposited_col = pow((float3)(103 / 255.0, 121 / 255.0, 67 / 255.0), 2.2); // lighter orange brown
	const float3 rock_col       = (float3)(constants->rock_col[0],       constants->rock_col[1],       constants->rock_col[2]);
	const float3 deposited_col  = (float3)(constants->sediment_col[0],   constants->sediment_col[1],   constants->sediment_col[2]);
	const float3 vegetation_col = (float3)(constants->vegetation_col[0], constants->vegetation_col[1], constants->vegetation_col[2]);

	const float3 snow_col = (float3)(0.95f);
	const float3 water_col = (float3)(0,0,1.f);


	const float water_h = waterHeightForMass(terrain_state[src_x   + src_y  *constants->W].water_mass, constants);

//	if(x == 100 && y == 100)
//		printf("!!!!!!!!!!!!!! water_h: %f  \n", water_h);

	float3 extinction = (float3)(1.0, 0.10, 0.1) * 12;
	float3 exp_optical_depth = constants->draw_water ? exp(extinction * -water_h) : 1.f;
	

	const float water_frac = constants->draw_water ? (1.f - exp(-2.f * water_h)) : 0.f;

	const float3 rock_sed_col = mix(rock_col, deposited_col, smoothstep(0.f, 0.3f, terrain_state[src_x   + src_y  *constants->W].deposited_sed_h));

	//const float vegetation_frac = smoothstep(0.4f, 0.8f, normal.z) * (1.f - smoothstep(0.5f, 0.7f, water_h));

	//const float3 ground_col = mix(rock_sed_col, vegetation_col, vegetation_frac);
	const float3 ground_col = rock_sed_col;
	
//	float3 attentuated_ground_col = ground_col * exp_optical_depth;

//	float3 inscatter_radiance_sigma_s_over_sigma_t = (float3)(1000000.0, 10000000.0, 30000000.0) * 0.00000003f;
//	float3 inscattering = inscatter_radiance_sigma_s_over_sigma_t * ((float3)(1.0) - exp_optical_depth);

	float3 final_col = ground_col; // TEMP attentuated_ground_col + inscattering;
	//const float3 final_col = mix(
	//	,
	//	water_col,
	//	water_frac
	//);

	
	if(constants->debug_draw_channel == TextureShow_WaterSpeed)
	{
		const float speed = length(terrain_state[src_x + src_y * constants->W].water_vel);
		final_col = (float3)(speed / constants->debug_display_max_val);
	}
	else if(constants->debug_draw_channel == TextureShow_WaterDepth)
	{
		const float depth = waterHeightForMass(terrain_state[src_x + src_y * constants->W].water_mass, constants);
		final_col = (float3)(depth / constants->debug_display_max_val);
	}
	else if(constants->debug_draw_channel == TextureShow_SuspendedSedimentVol)
	{
		const float suspended_vol = terrain_state[src_x + src_y * constants->W].suspended_vol;
		final_col = (float3)(suspended_vol / constants->debug_display_max_val);
	}
	else if(constants->debug_draw_channel == TextureShow_DepositedSedimentH)
	{
		const float h = terrain_state[src_x + src_y * constants->W].deposited_sed_h;
		final_col = (float3)(h / constants->debug_display_max_val);
	}

	write_imagef(terrain_texture, (int2)(x, y), (float4)(final_col, 1.f));
}
