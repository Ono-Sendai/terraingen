/*=====================================================================
erosion_kernel.cl
-----------------
Copyright Nicholas Chapman 2023 -
=====================================================================*/

// See "Fast Hydraulic Erosion Simulation and Visualization on GPU"
// Also
// "Fast Hydraulic and Thermal Erosion on the GPU"
// http://www.cescg.org/CESCG-2011/papers/TUBudapest-Jako-Balazs.pdf


#define DO_SEMILAGRANGIAN_ADVECTION 0

inline float square(float x)
{
	return x*x;
}


typedef struct
{
	float height; // terrain height ('b')
	float water; // water height ('d')
	float suspended; // Amount of suspended sediment. ('s')
	float deposited_sed; // deposited sediment

	float u, v; // currently storing water flux (m^3/s) in x and y directions. OLD: velocity

} TerrainState;


typedef struct
{
	float f_L, f_R, f_T, f_B; // outflow flux.  (m^3 s^-1)
	float sed_f_L, sed_f_R, sed_f_T, sed_f_B; // outflow sediment flux.  (m^3 s^-1)

} FlowState;

typedef struct
{
	float flux[8];

} ThermalErosionState;


//typedef struct
//{
//	float u, v; // velocity
//
//} WaterVelState;


typedef struct 
{
	float delta_t; // time step
	float r; // rainfall rate
	float A; // cross-sectional 'pipe' area
	float g; // gravity accel magnitude. positive.
	float l; // virtual pipe length
	float f; // fricton constant
	float l_x; // width between grid points in x direction
	float l_y;

	float K_c;// = 0.01; // 1; // sediment capacity constant
	float K_s;// = 0.01; // 0.5; // dissolving constant.
	float K_d;// = 0.01; // 1; // deposition constant
	float K_dmax;// = 0.1f; // Maximum erosion depth: water depth at which erosion stops.
	float K_e; // Evaporation constant

	float K_t; // thermal erosion constant
	float K_tdep; // thermal erosion constant for deposited sediment
	float max_talus_angle;
	float tan_max_talus_angle;
	float max_deposited_talus_angle;
	float tan_max_deposited_talus_angle;
} Constants;


// Sets f_L, f_T, f_R, f_B in new_flow_state
__kernel void flowSimulationKernel(
	__global const TerrainState* restrict const terrain_state, 
	__global const FlowState* restrict const flow_state, 
	__global       FlowState* restrict const new_flow_state, 
	__global Constants* restrict const constants
)
{
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

	// Compute intermediate water height (eqn. 1)
	const float d_1 = state_middle->water + constants->delta_t * constants->r;

	// Step 2: Flow simulation

	// Eqn. 3: Compute total height difference between this cell and adjacent cells 
	// NOTE: since rainfall is constant for all cells, it cancels out, so ignore when computing height differences.
	const float middle_total_h = state_middle->height + state_middle->deposited_sed + state_middle->water;
	const float delta_h_L = middle_total_h - (state_left ->height + state_left ->deposited_sed + state_left ->water);
	const float delta_h_T = middle_total_h - (state_top  ->height + state_top  ->deposited_sed + state_top  ->water);
	const float delta_h_R = middle_total_h - (state_right->height + state_right->deposited_sed + state_right->water);
	const float delta_h_B = middle_total_h - (state_bot  ->height + state_bot  ->deposited_sed + state_bot  ->water);


	const float friction_factor = (1.0f - constants->f * constants->delta_t);

	// Eqn. 2: Compute outflow flux to adjacent cells
	const float h_p = state_middle->water;
	const float w = 1.0f; // pipe width
	// Assume w = l, so w/l factor in flux deriv = 1.
	const float flux_factor = constants->delta_t * h_p * constants->g;
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

	

	// d_1 * l_x * l_y = current water volume in cell
	// (f_L_next + f_T_next + f_R_next + f_B_next) * delta_t = volume of water to be removed next timestep.  (m^3 s^-1  .  s = m^3)
	// If the volume of water to be removed is > current volume, we scale down the volume of water to be removed.
	const float cur_vol = d_1 * constants->l_x * constants->l_y;
	float K = min(1.f, cur_vol / ((f_L_next + f_T_next + f_R_next + f_B_next) * constants->delta_t)); // Eqn. 4

	f_L_next *= K;
	f_T_next *= K;
	f_R_next *= K;
	f_B_next *= K;

	new_flow_state_middle->f_L = f_L_next;
	new_flow_state_middle->f_R = f_R_next;
	new_flow_state_middle->f_T = f_T_next;
	new_flow_state_middle->f_B = f_B_next;


	//NEW: set out sediment flux
	const float out_L_frac = f_L_next / cur_vol;
	const float out_R_frac = f_R_next / cur_vol;
	const float out_B_frac = f_B_next / cur_vol;
	const float out_T_frac = f_T_next / cur_vol;

	const float cur_suspended = state_middle->suspended;
	new_flow_state_middle->sed_f_L = cur_suspended * out_L_frac;
	new_flow_state_middle->sed_f_R = cur_suspended * out_R_frac;
	new_flow_state_middle->sed_f_T = cur_suspended * out_T_frac;
	new_flow_state_middle->sed_f_B = cur_suspended * out_B_frac;
}


// Sets flux in thermal_erosion_state
__kernel void thermalErosionFluxKernel(
	__global const TerrainState* restrict const terrain_state, 
	__global       ThermalErosionState* restrict const thermal_erosion_state, 
	__global Constants* restrict const constants
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int x_minus_1 = max(x-1, 0);
	const int x_plus_1  = min(x+1, W-1);
	const int y_minus_1 = max(y-1, 0);
	const int y_plus_1  = min(y+1, H-1);

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
	const float h_0 = middle_h - state_0->height/* + state_0->deposited_sed*/; // height diff between adjacent cell and middle cell
	const float h_1 = middle_h - state_1->height/* + state_1->deposited_sed*/;
	const float h_2 = middle_h - state_2->height/* + state_2->deposited_sed*/;
	const float h_3 = middle_h - state_3->height/* + state_3->deposited_sed*/;
	const float h_4 = middle_h - state_4->height/* + state_4->deposited_sed*/;
	const float h_5 = middle_h - state_5->height/* + state_5->deposited_sed*/;
	const float h_6 = middle_h - state_6->height/* + state_6->deposited_sed*/;
	const float h_7 = middle_h - state_7->height/* + state_7->deposited_sed*/;

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

	const float tan_angle_0 = h_0 * (1 / sqrt(2.f));
	const float tan_angle_1 = h_1;
	const float tan_angle_2 = h_2 * (1 / sqrt(2.f));
	const float tan_angle_3 = h_3;
	const float tan_angle_4 = h_4;
	const float tan_angle_5 = h_5 * (1 / sqrt(2.f));
	const float tan_angle_6 = h_6;
	const float tan_angle_7 = h_7 * (1 / sqrt(2.f));

	const float tan_max_talus_angle = constants->tan_max_talus_angle;

	// Total height difference, for cells for which the height difference exceeds the max talus angle
	const float total_height_diff = 
		((tan_angle_0 > tan_max_talus_angle) ? h_0 : 0.0) + 
		((tan_angle_1 > tan_max_talus_angle) ? h_1 : 0.0) + 
		((tan_angle_2 > tan_max_talus_angle) ? h_2 : 0.0) + 
		((tan_angle_3 > tan_max_talus_angle) ? h_3 : 0.0) + 
		((tan_angle_4 > tan_max_talus_angle) ? h_4 : 0.0) + 
		((tan_angle_5 > tan_max_talus_angle) ? h_5 : 0.0) + 
		((tan_angle_6 > tan_max_talus_angle) ? h_6 : 0.0) + 
		((tan_angle_7 > tan_max_talus_angle) ? h_7 : 0.0);

	const float norm_factor = 1.f / total_height_diff;

	const float a = 1.0f; // cell area
	const float R = 1.0f; // hardness
	float common_factors;
	if(max_height_diff > 0 && total_height_diff > 0)
		common_factors = norm_factor * a * constants->delta_t * constants->K_t * R * max_height_diff * 0.5f;
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
}



// Sets flux in thermal_erosion_state
__kernel void thermalErosionDepositedFluxKernel(
	__global const TerrainState* restrict const terrain_state, 
	__global       ThermalErosionState* restrict const thermal_erosion_state, 
	__global Constants* restrict const constants
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int x_minus_1 = max(x-1, 0);
	const int x_plus_1  = min(x+1, W-1);
	const int y_minus_1 = max(y-1, 0);
	const int y_plus_1  = min(y+1, H-1);

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
	const float h_0 = middle_h - (state_0->height + state_0->deposited_sed);// state_0->sediment[0] + state_0->sediment[1] + state_0->sediment[2]); // height diff between adjacent cell and middle cell
	const float h_1 = middle_h - (state_1->height + state_1->deposited_sed);// state_1->sediment[0] + state_1->sediment[1] + state_1->sediment[2]);
	const float h_2 = middle_h - (state_2->height + state_2->deposited_sed);// state_2->sediment[0] + state_2->sediment[1] + state_2->sediment[2]);
	const float h_3 = middle_h - (state_3->height + state_3->deposited_sed);// state_3->sediment[0] + state_3->sediment[1] + state_3->sediment[2]);
	const float h_4 = middle_h - (state_4->height + state_4->deposited_sed);// state_4->sediment[0] + state_4->sediment[1] + state_4->sediment[2]);
	const float h_5 = middle_h - (state_5->height + state_5->deposited_sed);// state_5->sediment[0] + state_5->sediment[1] + state_5->sediment[2]);
	const float h_6 = middle_h - (state_6->height + state_6->deposited_sed);// state_6->sediment[0] + state_6->sediment[1] + state_6->sediment[2]);
	const float h_7 = middle_h - (state_7->height + state_7->deposited_sed);// state_7->sediment[0] + state_7->sediment[1] + state_7->sediment[2]);

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

	const float tan_angle_0 = h_0 * (1 / sqrt(2.f));
	const float tan_angle_1 = h_1;
	const float tan_angle_2 = h_2 * (1 / sqrt(2.f));
	const float tan_angle_3 = h_3;
	const float tan_angle_4 = h_4;
	const float tan_angle_5 = h_5 * (1 / sqrt(2.f));
	const float tan_angle_6 = h_6;
	const float tan_angle_7 = h_7 * (1 / sqrt(2.f));

	const float tan_max_talus_angle = constants->tan_max_deposited_talus_angle;

	// Total height difference, for cells for which the height difference exceeds the max talus angle
	const float total_height_diff = 
		((tan_angle_0 > tan_max_talus_angle) ? h_0 : 0.0) + 
		((tan_angle_1 > tan_max_talus_angle) ? h_1 : 0.0) + 
		((tan_angle_2 > tan_max_talus_angle) ? h_2 : 0.0) + 
		((tan_angle_3 > tan_max_talus_angle) ? h_3 : 0.0) + 
		((tan_angle_4 > tan_max_talus_angle) ? h_4 : 0.0) + 
		((tan_angle_5 > tan_max_talus_angle) ? h_5 : 0.0) + 
		((tan_angle_6 > tan_max_talus_angle) ? h_6 : 0.0) + 
		((tan_angle_7 > tan_max_talus_angle) ? h_7 : 0.0);

	const float norm_factor = 1.f / total_height_diff;

	const float cur_deposited_sed = state_middle->deposited_sed;

	const float a = 1.0f; // cell area
	const float R = 1.f; // hardness TEMP
	float common_factors;
	if(max_height_diff > 0 && total_height_diff > 0)
		common_factors = norm_factor * a * constants->delta_t * constants->K_tdep * R * max_height_diff * 0.5f;
	else
		common_factors = 0;

	float sum_flux = 
		h_0 * common_factors +
		h_1 * common_factors +
		h_2 * common_factors +
		h_3 * common_factors +
		h_4 * common_factors +
		h_5 * common_factors +
		h_6 * common_factors +
		h_7 * common_factors;

	float K = 1;
	if(cur_deposited_sed > 0)
	{
		if(sum_flux > cur_deposited_sed)
		{
			K = cur_deposited_sed / sum_flux;
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
}


// Updates water, u, v in terrain_state
__kernel void waterAndVelFieldUpdateKernel(
	__global const FlowState* restrict const flow_state, 
	__global       TerrainState* restrict const terrain_state, 
	__global Constants* restrict const constants
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int x_minus_1 = max(x-1, 0);
	const int x_plus_1  = min(x+1, W-1);
	const int y_minus_1 = max(y-1, 0);
	const int y_plus_1  = min(y+1, H-1);


	__global const FlowState* const state_left     = &flow_state[x_minus_1 + y         * W];
	__global const FlowState* const state_right    = &flow_state[x_plus_1  + y         * W];
	__global const FlowState* const state_top      = &flow_state[x         + y_plus_1  * W];
	__global const FlowState* const state_bot      = &flow_state[x         + y_minus_1 * W];
	__global const FlowState* const state_middle   = &flow_state[x         + y          *W];

	__global TerrainState* const terrain_state_middle   = &terrain_state[x         + y          *W];


	// Step 3: Water surface and velocity field update

	//const float d_1 = state_middle->water; // Current water height of middle cell
	// Compute intermediate water height (eqn. 1)
	const float d_1 = terrain_state_middle->water + constants->delta_t * constants->r;

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
	float d_2 = max(0.f, d_1 + delta_V / (constants->l_x * constants->l_y)); // Eqn. 7: new water height for middle cell: change in height = change in volume / cell area.  
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


	const float delta_sed_V = constants->delta_t *
		((in_sed_left_R + in_sed_right_L + in_sed_top_B + in_sed_bot_T) - // inwards flow
		 (state_middle->sed_f_L + state_middle->sed_f_R + state_middle->sed_f_T + state_middle->sed_f_B)); // outwards sediment flow
	// m^3 = s * (m^3 s^-1)

	// Compute new amount of sediment
	const float new_suspended = max(0.f, terrain_state_middle->suspended + delta_sed_V);

	//TEMP: store unit discharge in u, v
	float u = delta_W_x;
	float v = delta_W_y;

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
	 
	//float u = delta_W_x / (d_bar * constants->l_x); // u_{t+delta_t}
	//float v = delta_W_y / (d_bar * constants->l_y); // v_{t+delta_t}

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

	terrain_state_middle->water = d_2;
#if !DO_SEMILAGRANGIAN_ADVECTION
	terrain_state_middle->suspended = new_suspended;
#endif
	terrain_state_middle->u = u;
	terrain_state_middle->v = v;
}


// Updates 'height', 'suspended', 'sediment' in terrain_state
__kernel void erosionAndDepositionKernel(
	__global       TerrainState* restrict const terrain_state, 
	__global Constants* restrict const constants
	)
{
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
	__global       TerrainState* const state_middle   = &terrain_state[x         + y          *W];

	
	const float L_h = state_left ->height + state_left ->deposited_sed; // state_left ->sediment[0] + state_left ->sediment[1] + state_left ->sediment[2];// + state_left ->water;
	const float R_h = state_right->height + state_right->deposited_sed; // state_right->sediment[0] + state_right->sediment[1] + state_right->sediment[2];// + state_right->water;
	const float B_h = state_bot  ->height + state_bot  ->deposited_sed; // state_bot  ->sediment[0] + state_bot  ->sediment[1] + state_bot  ->sediment[2];// + state_bot  ->water;
	const float T_h = state_top  ->height + state_top  ->deposited_sed; // state_top  ->sediment[0] + state_top  ->sediment[1] + state_top  ->sediment[2];// + state_top  ->water;

	const float dh_dx = (R_h - L_h) * (1.f / (2*constants->l_x));
	const float dh_dy = (T_h - B_h) * (1.f / (2*constants->l_y));

	const float3 normal = normalize((float3)(-dh_dx, -dh_dy, 1));

	//const float theta = acos(1 / sqrt(dh_dx*dh_dx + dh_dy*dh_dy + 1));
	//const float sin_alpha = sin(theta);
	const float cos_alpha = 1.0f / sqrt(square(dh_dx) + square(dh_dy) + 1); // https://math.stackexchange.com/questions/1044044/local-tilt-angle-based-on-height-field
	const float sin_alpha = sqrt(1 - min(1.0f, cos_alpha*cos_alpha));

	//const float alpha = max(fabs(state_right->height - state_left->height), fabs(state_top->height - state_bot->height));
	///float sin_alpha = alpha;
	//sin_alpha = min(0.5f, sin_alpha);
	const float use_sin_alpha = max(0.3f, sin_alpha); // NOTE: min sin alpha

	float v_len = sqrt(square(state_middle->u) + square(state_middle->v));

	// Compute l_max as a function of water height (d)  (eqn. 10 from 'Fast Hydraulic and Thermal Erosion on the GPU')

	const float3 water_vel = (float3)(state_middle->u, state_middle->v, state_middle->u * dh_dx + state_middle->v * dh_dy);
	const float3 unit_water_vel = normalize(water_vel);

	const float hit_dot = max(0.05f, -dot(unit_water_vel, normal));
			
	const float water_d = state_middle->water;
	/*float l_max;
	if(d <= 0)
		l_max = 0;
	else if(d >= constants->K_dmax)
		l_max = 1;
	else
		l_max = 1 - (constants->K_dmax - d) / constants->K_dmax;*/

	//const float water_factor = min(0.01f, state_middle->water * 10.0f);
	//const float water_factor = min(1.0f, state_middle->water * 1.0f);

	const float water_depth_factor = 1.f;//min(water_d, constants->K_dmax);

	// Compute Sediment transport capacity (eq 10)

	//const float C = 0.001f * constants->K_c * v_len;
	//const float q = v_len * max(0.f, min(water_d, 1.0f));
	const float q = min(1.0f, v_len); //fabs(state_middle->u) + fabs(state_middle->v)); // unit water discharge
	const float q_to_gamma = square(q);
	const float S = use_sin_alpha;
	const float S_to_beta = pow(S, 1.5f);
	const float C = constants->K_c * S_to_beta * q_to_gamma;
	//const float C = /*hit_dot **/ /*d * */constants->K_c * v_len * /*use_sin_alpha **/ water_depth_factor;//use_sin_alpha * v_len /** (1.0 - l_max)*/;//f/min(10000.0f, v_len);
			
	float height = state_middle->height;
	float suspended = state_middle->suspended;
	float deposited_sed = state_middle->deposited_sed;
	//const float d_2 = state_middle->water;
	//float b_new/*, d_3*/;
	
	const float suspended_sum = suspended; // suspended[0] + suspended[1] + suspended[2];
	if(C > suspended_sum) // suspended amount is smaller than transport capacity, dissolve soil into water:
	{
		float sed_change = hit_dot * constants->delta_t * constants->K_s * (C - suspended_sum); //delta_t * K_s * (C - s_t);
		float sed_change_rock = sed_change * 0.3f; //delta_t * K_s * (C - s_t);
		float sed_change_dep  = sed_change * 0.7f; //delta_t * K_s * (C - s_t);

		// Dissolve any deposited sediment into the water
		const float deposited_sed_delta = min(sed_change_dep, deposited_sed); // Dissolve <= the amount of deposited sediment here.
		deposited_sed -= deposited_sed_delta;
		suspended += deposited_sed_delta;

		sed_change_dep -= deposited_sed_delta;
		

		if(sed_change > 0) // If we have dissolved all deposited sediment, and there is still dissolving to be done:
		{
			// Dissolve underlying rock

			height -= sed_change_rock;// Reduce terrain height
			suspended += sed_change_rock; // Add to suspended height
		}
	}
	else // else suspended amount exceeds transport capacity, so deposit sediment:
	{
		float sed_change = constants->delta_t * constants->K_d * (suspended_sum - C);

		suspended -= sed_change;
		deposited_sed += sed_change;
	}
		
	//if(x == 200 && y == 256)
	//	printf("s_t:  %1.15f   , C: %1.15f   \n", s_t, C);

	// Write
	state_middle->height = height;
	//state_middle->water = d_2;
	state_middle->suspended = suspended;
	state_middle->deposited_sed = deposited_sed;
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


// sediment transportation kernel.  Updates 'suspended' in terrain_state
__kernel void sedimentTransportationKernel(
	__global       TerrainState* restrict const terrain_state, 
	__global Constants* restrict const constants
	)
{
#if DO_SEMILAGRANGIAN_ADVECTION
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	__global       TerrainState* const state_middle   = &terrain_state[x         + y          *W];

	float u = state_middle->u;
	float v = state_middle->v;

	const float old_x = clamp((float)x - /*state_middle->*/u * constants->delta_t/* / constants->l_x*/, 0.0f, (float)(W-1)); // NOTE: should take into account l_x, l_y here
	const float old_y = clamp((float)y - /*state_middle->*/v * constants->delta_t/* / constants->l_y*/, 0.0f, (float)(H-1));
	//const float old_x = clamp((float)x - /*state_middle->*/1/* / constants->l_x*/, 0.0f, (float)(W-1)); // NOTE: should take into account l_x, l_y here
	//const float old_y = clamp((float)y - /*state_middle->*/0/* / constants->l_y*/, 0.0f, (float)(H-1));

	const float floor_old_x = floor(old_x);
	const float floor_old_y = floor(old_y);
	const float t_x = old_x - (float)(int)floor_old_x;
	const float t_y = old_y - (float)(int)floor_old_y;
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
		terrain_state[old_xi  + old_yi  * W].suspended * one_t_x * one_t_y +
		terrain_state[old_xi1 + old_yi  * W].suspended * t_x     * one_t_y +
		terrain_state[old_xi  + old_yi1 * W].suspended * one_t_x * t_y     +
		terrain_state[old_xi1 + old_yi1 * W].suspended * t_x     * t_y     ;

	state_middle->suspended = old_suspended;
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
	__global Constants* restrict const constants
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int x_minus_1 = max(x-1, 0);
	const int x_plus_1  = min(x+1, W-1);
	const int y_minus_1 = max(y-1, 0);
	const int y_plus_1  = min(y+1, H-1);

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

	middle_terrain_state->height += net_material_change; // NOTE: Take into account area?
}


__kernel void thermalErosionDepositedMovementKernel(
	__global const ThermalErosionState* restrict const thermal_erosion_state, 
	__global       TerrainState* restrict const terrain_state, 
	__global Constants* restrict const constants
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int x_minus_1 = max(x-1, 0);
	const int x_plus_1  = min(x+1, W-1);
	const int y_minus_1 = max(y-1, 0);
	const int y_plus_1  = min(y+1, H-1);

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

	middle_terrain_state->deposited_sed += net_material_change; // NOTE: Take into account area?
}



// evaporation kernel.  Updates 'water' in terrain_state
__kernel void evaporationKernel(
	__global       TerrainState* restrict const terrain_state, 
	__global Constants* restrict const constants
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	__global       TerrainState* const state_middle   = &terrain_state[x         + y          *W];

	const float d_new = state_middle->water * (1 - constants->K_e * constants->delta_t);

	state_middle->water = d_new;
}


typedef struct
{
	float pos[3];
	float normal[3];
	float uv[2];
} Vertex;

inline float totalTerrainHeight(__global TerrainState* restrict const state)
{
	return state->height + state->deposited_sed;
}

__kernel void setHeightFieldMeshKernel(
	__global       TerrainState* restrict const terrain_state, 
	__global Constants* restrict const constants,
	__global Vertex* restrict const vertex_buffer,
	unsigned int vertex_buffer_offset_B,
	write_only image2d_t terrain_texture
)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	__global Vertex* mesh_vert_0 = &vertex_buffer[vertex_buffer_offset_B / sizeof(Vertex)];

	int vert_xres = max(2, W);
	int vert_yres = max(2, H);
	int quad_xres = vert_xres - 1; // Number of quads in x and y directions
	int quad_yres = vert_yres - 1; // Number of quads in x and y directions

	__global Vertex* mesh_vert = &mesh_vert_0[x + y * vert_xres];

	float quad_w_x = 1.f;//(float)W/ quad_xres; // Width in metres of each quad
	float quad_w_y = quad_w_x;
	if(H <= 10)
	{
		quad_w_y *= 20.f; // For height = 1 (1-d debugging case), display strip a bit wider
	}

	const int max_src_x = W - 1;
	const int max_src_y = H - 1; // Store these so we can handle width 1 sims

	const float p_x = x * quad_w_x;
	const float p_y = y * quad_w_y;
	const float dx = 1.f;
	const float dy = 1.f;

	const int src_x = min(x, max_src_x);
	const int src_y = min(y, max_src_y);
	const int src_x_1 = min(x + 1, max_src_x);
	const int src_y_1 = min(y + 1, max_src_y);
	const float z    = totalTerrainHeight(&terrain_state[src_x   + src_y  *W]);// + (cur_heightfield_show == HeightFieldShow::HeightFieldShow_TerrainAndWater ? sim.terrain_state.elem(src_x,   src_y).water : 0);
	const float z_dx = totalTerrainHeight(&terrain_state[src_x_1 + src_y  *W]);// + (cur_heightfield_show == HeightFieldShow::HeightFieldShow_TerrainAndWater ? sim.terrain_state.elem(src_x_1, src_y).water : 0);
	const float z_dy = totalTerrainHeight(&terrain_state[src_x   + src_y_1*W]);// + (cur_heightfield_show == HeightFieldShow::HeightFieldShow_TerrainAndWater ? sim.terrain_state.elem(src_x, src_y_1).water : 0);

	const float3 p_dx_minus_p = (float3)(dx, 0, z_dx - z); // p(p_x + dx, dy) - p(p_x, p_y) = (p_x + dx, d_y, z_dx) - (p_x, p_y, z) = (d_x, 0, z_dx - z)
	const float3 p_dy_minus_p = (float3)(0, dy, z_dy - z);

	const float3 normal = normalize(cross(p_dx_minus_p, p_dy_minus_p));

	const float3 pos = (float3)(p_x, p_y, z);
	mesh_vert->pos[0] = p_x;
	mesh_vert->pos[1] = p_y;
	mesh_vert->pos[2] = z;

	mesh_vert->normal[0] = normal.x;
	mesh_vert->normal[1] = normal.y;
	mesh_vert->normal[2] = normal.z;


	// Write to terrain texture
	const float3 rock_col = pow((float3)(64.0 / 255.0, 60.0 / 255.0, 45 / 255.0), 2.2); // brown
	const float3 deposited_col = pow((float3)(103 / 255.0, 91 / 255.0, 67 / 255.0), 2.2); // lighter orange brown
	const float3 snow_col = (float3)(0.95f);

	//const float val = terrain_state[src_x   + src_y  *W].deposited_sed + terrain_state[src_x   + src_y  *W].water;
	const float3 final_col = mix(rock_col, snow_col, smoothstep(0.f, 0.3f, terrain_state[src_x   + src_y  *W].deposited_sed));

	write_imagef(terrain_texture, (int2)(x, y), (float4)(final_col, 1.f));
}
