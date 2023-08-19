/*=====================================================================
erosion_kernel.cl
-----------------
Copyright Nicholas Chapman 2023 -
=====================================================================*/

// See "Fast Hydraulic Erosion Simulation and Visualization on GPU"
// http://ww.w.roxlu.com/downloads/scholar/004.fluid.fast_hydrolic_erosion_simulation_and_visualisation.pdf
// Also
// "Fast Hydraulic and Thermal Erosion on the GPU"
// http://www.cescg.org/CESCG-2011/papers/TUBudapest-Jako-Balazs.pdf


inline float square(float x)
{
	return x*x;
}


typedef struct
{
	float height; // terrain height (b)
	float water; // water height (d)
	float suspended; // Amount of suspended sediment. (s)

	float u, v; // velocity

} TerrainState;


typedef struct
{
	float f_L, f_R, f_T, f_B; // outflow flux.  (m^3 s^-1)

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
	const float middle_total_h = state_middle->height + state_middle->water;
	const float delta_h_L = middle_total_h - (state_left ->height + state_left ->water);
	const float delta_h_T = middle_total_h - (state_top  ->height + state_top  ->water);
	const float delta_h_R = middle_total_h - (state_right->height + state_right->water);
	const float delta_h_B = middle_total_h - (state_bot  ->height + state_bot  ->water);

	// Eqn. 2: Compute outflow flux to adjacent cells
	const float flux_factor = constants->delta_t * constants->A * constants->g / constants->l;
	float f_L_next = max(0.f, flow_state_middle->f_L + flux_factor * delta_h_L); // If this cell is higher than left cell, delta_h_L is positive
	float f_T_next = max(0.f, flow_state_middle->f_T + flux_factor * delta_h_T);
	float f_R_next = max(0.f, flow_state_middle->f_R + flux_factor * delta_h_R);
	float f_B_next = max(0.f, flow_state_middle->f_B + flux_factor * delta_h_B);

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
	float K = min(1.f, d_1 * constants->l_x * constants->l_y / ((f_L_next + f_T_next + f_R_next + f_B_next) * constants->delta_t)); // Eqn. 4

	f_L_next *= K;
	f_T_next *= K;
	f_R_next *= K;
	f_B_next *= K;

	new_flow_state_middle->f_L = f_L_next;
	new_flow_state_middle->f_R = f_R_next;
	new_flow_state_middle->f_T = f_T_next;
	new_flow_state_middle->f_B = f_B_next;
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


	const float middle_h = state_middle->height;
	const float h_0 = middle_h - state_0->height; // height diff between adjacent cell and middle cell
	const float h_1 = middle_h - state_1->height;
	const float h_2 = middle_h - state_2->height;
	const float h_3 = middle_h - state_3->height;
	const float h_4 = middle_h - state_4->height;
	const float h_5 = middle_h - state_5->height;
	const float h_6 = middle_h - state_6->height;
	const float h_7 = middle_h - state_7->height;

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

	//float max_speed_comp = 1.f;

	float u, v;
	if(d_bar <= 1.0e-4f) // If the water height is ~= 0, then avoid divide by zero below and consider the water velocity to be zero.
	{
		u = v = 0;
	}
	else
	{
		// From eqn. 9:
		//m^s-1 = m^3 s^-1  / (m     * m)
		float new_u = delta_W_x / (d_bar * constants->l_x); // u_{t+delta_t}
		float new_v = delta_W_y / (d_bar * constants->l_y); // v_{t+delta_t}

		const float old_u = terrain_state_middle->u;
		const float old_v = terrain_state_middle->v;

		// TEMP HACK:
		u = old_u * 0.8f + new_u * 0.1f;
		v = old_v * 0.8f + new_v * 0.1f;
	}
	 
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
	terrain_state_middle->u = u;
	terrain_state_middle->v = v;
}


// Updates 'height', 'suspended' in terrain_state
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

	
	const float L_h = state_left ->height + state_left ->water;
	const float R_h = state_right->height + state_right->water;
	const float B_h = state_bot  ->height + state_bot  ->water;
	const float T_h = state_top  ->height + state_top  ->water;

	const float dh_dx = (R_h - L_h) * (1.f / (2*constants->l_x));
	const float dh_dy = (T_h - B_h) * (1.f / (2*constants->l_y));
	//const float theta = acos(1 / sqrt(dh_dx*dh_dx + dh_dy*dh_dy + 1));
	//const float sin_alpha = sin(theta);
	const float cos_alpha = 1.0f / sqrt(square(dh_dx) + square(dh_dy) + 1); // https://math.stackexchange.com/questions/1044044/local-tilt-angle-based-on-height-field
	const float sin_alpha = sqrt(1 - min(1.0f, cos_alpha*cos_alpha));

	//const float alpha = max(fabs(state_right->height - state_left->height), fabs(state_top->height - state_bot->height));
	///float sin_alpha = alpha;
	//sin_alpha = min(0.5f, sin_alpha);
	const float use_sin_alpha = max(0.1f, sin_alpha);

	float v_len = sqrt(square(state_middle->u) + square(state_middle->v));

	// Compute l_max as a function of water height (d)  (eqn. 10 from 'Fast Hydraulic and Thermal Erosion on the GPU')
			
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

	const float water_depth_factor = min(water_d, constants->K_dmax);

	// Compute Sediment transport capacity (eq 10)

	// TEMP NEW d
	const float C = /*d * */constants->K_c * v_len * use_sin_alpha * water_depth_factor;//use_sin_alpha * v_len /** (1.0 - l_max)*/;//f/min(10000.0f, v_len);
			
	const float b_t = state_middle->height;
	const float s_t = state_middle->suspended;
	//const float d_2 = state_middle->water;
	float b_new, s_1/*, d_3*/;
	if(C > s_t) // suspended amount is smaller than transport capacity, dissolve soil into water:
	{
		const float sed_change = constants->delta_t * constants->K_s * (C - s_t); //delta_t * K_s * (C - s_t);
		b_new = b_t - sed_change; // Reduce terrain height
		s_1   = s_t + sed_change; // Add to suspended height
		//d_3   = d_2 + sed_change; // Add to water height (not in paper?!)

		//s_1 = min(s_1, d_3);
	}
	else // else suspended amount exceeds transport capacity:
	{
		const float sed_change = constants->delta_t * constants->K_d * (s_t - C);
		b_new = b_t + sed_change; // Increase terrain height
		s_1   = s_t - sed_change; // Decrease suspended height
		//d_3   = d_2 - sed_change; // Decreased water height
	}
		
	//if(x == 200 && y == 256)
	//	printf("s_t:  %1.15f   , C: %1.15f   \n", s_t, C);

	state_middle->height = b_new;
	//state_middle->water = d_2;
	state_middle->suspended = s_1;
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


// sediment transportation kernel.  Updates 'suspended' in terrain_state
__kernel void sedimentTransportationKernel(
	__global       TerrainState* restrict const terrain_state, 
	__global Constants* restrict const constants
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	__global       TerrainState* const state_middle   = &terrain_state[x         + y          *W];

	const float old_x = clamp((float)x - state_middle->u * constants->delta_t/* / constants->l_x*/, 0.0f, (float)(W-1)); // NOTE: should take into account l_x, l_y here
	const float old_y = clamp((float)y - state_middle->v * constants->delta_t/* / constants->l_y*/, 0.0f, (float)(H-1));

	const float floor_old_x = floor(old_x);
	const float floor_old_y = floor(old_y);
	const float t_x = old_x - floor_old_x;
	const float t_y = old_y - floor_old_y;
	const int old_xi = clamp((int)floor_old_x, 0, W-1);
	const int old_yi = clamp((int)floor_old_y, 0, H-1);
	const int old_xi1 = clamp((int)floor_old_x + 1, 0, W-1);
	const int old_yi1 = clamp((int)floor_old_y + 1, 0, H-1);

	// Read sedimentation value at (old_x, old_y)
	const float old_s = biLerp(
		terrain_state[old_xi  + old_yi  * W].suspended,
		terrain_state[old_xi1 + old_yi  * W].suspended,
		terrain_state[old_xi  + old_yi1 * W].suspended,
		terrain_state[old_xi1 + old_yi1 * W].suspended,
		t_x, t_y);

	state_middle->suspended = old_s;
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
