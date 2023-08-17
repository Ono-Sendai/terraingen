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


#define W_mask (W - 1)
#define H_mask (H - 1)

inline int wrapX(int x)
{
	return x & W_mask;
}

inline int wrapY(int y)
{
	return y & H_mask;
}


inline float biLerp(float a, float b, float c, float d, float t_x, float t_y)
{
	const float one_t_x = 1 - t_x;
	const float one_t_y = 1 - t_y;
	return 
		one_t_x * one_t_y * a + 
		t_x * one_t_y * b + 
		one_t_x * t_y * c + 
		t_x * t_y * d;
}

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
	float f_L, f_R, f_T, f_B; // outflow flux

} FlowState;


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

	const int x_minus_1 = wrapX(x - 1);
	const int x_plus_1  = wrapX(x + 1);
	const int y_minus_1 = wrapY(y - 1);
	const int y_plus_1  = wrapY(y + 1);

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

	// Compute intermediate water height (d_1) for adjacent cells
	//const float d_1_L = state_left->water  + delta_t * r; // d_1(x-1, y)
	//const float d_1_T = state_top->water   + delta_t * r; // d_1(x, y+1)
	//const float d_1_R = state_right->water + delta_t * r; // d_1(x+1, y)
	//const float d_1_B = state_bot->water   + delta_t * r; // d_1(x, y-1)

	//// Eqn. 3: Compute total height difference between this cell and adjacent cells 
	//const float delta_h_L = (state_middle->height + d_1) - (state_left->height  + d_1_L);
	//const float delta_h_T = (state_middle->height + d_1) - (state_top->height   + d_1_T);
	//const float delta_h_R = (state_middle->height + d_1) - (state_right->height + d_1_R);
	//const float delta_h_B = (state_middle->height + d_1) - (state_bot->height   + d_1_B);

	//const float d_1_L = state_left->water  + delta_t * r; // d_1(x-1, y)
	//const float d_1_T = state_top->water   + delta_t * r; // d_1(x, y+1)
	//const float d_1_R = state_right->water + delta_t * r; // d_1(x+1, y)
	//const float d_1_B = state_bot->water   + delta_t * r; // d_1(x, y-1)

	// Eqn. 3: Compute total height difference between this cell and adjacent cells 
	// NOTE: since rainfall is constant for all cells, it cancels out.
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

	if(x == 0)
		f_L_next = 0;
	if(x == W-1)
		f_R_next = 0;
	if(y == 0)
		f_B_next = 0;
	if(y == H-1)
		f_T_next = 0;

	const float K = min(1.f, d_1 * constants->l_x * constants->l_y / ((f_L_next + f_T_next + f_R_next + f_B_next) * constants->delta_t)); // Eqn. 4

	f_L_next *= K;
	f_T_next *= K;
	f_R_next *= K;
	f_B_next *= K;

	new_flow_state_middle->f_L = f_L_next;
	new_flow_state_middle->f_R = f_R_next;
	new_flow_state_middle->f_T = f_T_next;
	new_flow_state_middle->f_B = f_B_next;
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

	const int x_minus_1 = wrapX(x - 1);
	const int x_plus_1  = wrapX(x + 1);
	const int y_minus_1 = wrapY(y - 1);
	const int y_plus_1  = wrapY(y + 1);

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

	// Compute net volume change for the water (eqn 6):
	const float delta_V = constants->delta_t *
		((state_left  ->f_R + state_right ->f_L + state_top   ->f_B + state_bot   ->f_T) - // inwards flow
		 (state_middle->f_L + state_middle->f_R + state_middle->f_T + state_middle->f_B)); // outwards flow

	float d_2 = d_1 + delta_V / (constants->l_x * constants->l_y); // Eqn. 7: new water height for middle cell

	// Eqn 8.  Compute average amount of water passing through cell (x, y) in the x direction:
	const float delta_W_x = (state_left->f_R - state_middle->f_L + state_middle->f_R - state_right->f_L) * 0.5f;
	//const float delta_W_y = (state_bot->f_T  - state_middle->f_B + state_middle->f_T - state_top->f_B  ) * 0.5f; // +y direction is down

	// Compute average amount of water passing through cell (x, y) in the y direction:
	//const float delta_W_y = (state_top->f_B  - state_middle->f_T + state_middle->f_B - state_bot->f_T  ) * 0.5f;
	const float delta_W_y = (state_bot->f_T  - state_middle->f_B + state_middle->f_T - state_top->f_B  ) * 0.5f;

	const float d_bar = (d_1 + d_2) * 0.5f; // Average water height

	//float max_speed_comp = 1.f;

	// From eqn. 9:
	float u = delta_W_x / (d_bar * constants->l_x); // u_{t+delta_t}
	float v = delta_W_y / (d_bar * constants->l_y); // v_{t+delta_t}

	//if(d_2 < 0.01f) // TEMP: force water depth to 0 if too small
	//{
	//	d_2 = 0;
	//	u = 0;
	//	v = 0;
	//}

	//const float v_len = sqrt(u*u + v*v);

	//if(x == 200 && y == 200)
	//	printf("v_len: %f  \n", v_len);
	/*if(v_len > 4.0f)
	{
	u /= v_len;
	v /= v_len;
	}*/

	terrain_state_middle->water = d_2;
	terrain_state_middle->u = u;
	terrain_state_middle->v = v;
}


// Updates height, water, suspended in terrain_state
__kernel void erosionAndDepositionKernel(
	__global       TerrainState* restrict const terrain_state, 
	__global Constants* restrict const constants
	)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const int x_minus_1 = wrapX(x - 1);
	const int x_plus_1  = wrapX(x + 1);
	const int y_minus_1 = wrapY(y - 1);
	const int y_plus_1  = wrapY(y + 1);

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
	const float use_sin_alpha = sin_alpha;//max(0.1f, sin_alpha);
	const float v_len = sqrt(square(state_middle->u) + square(state_middle->v));

	// Compute l_max as a function of water height (d)  (eqn. 10 from 'Fast Hydraulic and Thermal Erosion on the GPU')
			
	const float d = state_middle->water;
	float l_max;
	if(d <= 0)
		l_max = 0;
	else if(d >= constants->K_dmax)
		l_max = 1;
	else
		l_max = 1 - (constants->K_dmax - d) / constants->K_dmax;

	//const float water_factor = min(0.01f, state_middle->water * 10.0f);
	//const float water_factor = min(1.0f, state_middle->water * 1.0f);

	// Compute Sediment transport capacity (eq 10)

	// TEMP NEW d
	const float C = /*d * */constants->K_c * v_len * use_sin_alpha;//use_sin_alpha * v_len /** (1.0 - l_max)*/;//f/min(10000.0f, v_len);
			
	const float b_t = state_middle->height;
	const float s_t = state_middle->suspended;
	const float d_2 = state_middle->water;
	float b_new, s_1, d_3;
	if(C > s_t) // suspended amount is smaller than transport capacity, dissolve soil into water:
	{
		const float sed_change = constants->delta_t * constants->K_s * (C - s_t); //delta_t * K_s * (C - s_t);
		b_new = b_t - sed_change; // Reduce terrain height
		s_1   = s_t + sed_change; // Add to suspended height
		d_3   = d_2 + sed_change; // Add to water height (not in paper?!)

		//s_1 = min(s_1, d_3);
	}
	else // else suspended amount exceeds transport capacity:
	{
		const float sed_change = constants->delta_t * constants->K_d * (s_t - C);
		b_new = b_t + sed_change; // Increase terrain height
		s_1   = s_t - sed_change; // Decrease suspended height
		d_3   = d_2 - sed_change; // Decreased water height
	}
		
	//if(x == 200 && y == 256)
	//	printf("s_t:  %1.15f   , C: %1.15f   \n", s_t, C);

	state_middle->height = b_new;
	state_middle->water = d_2;
	state_middle->suspended = s_1;
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

	const float old_x = (float)x - state_middle->u * constants->delta_t / constants->l_x; // NOTE: should take into account l_x, l_y here
	const float old_y = (float)y - state_middle->v * constants->delta_t / constants->l_y;

	const float t_x = old_x - floor(old_x);
	const float t_y = old_y - floor(old_y);
	const int old_xi = wrapX((int)floor(old_x));
	const int old_yi = wrapY((int)floor(old_y));
	const int old_xi1 = wrapX(old_xi + 1);
	const int old_yi1 = wrapY(old_yi + 1);

	// Read sedimentation value at (old_x, old_y)
	const float old_s = biLerp(
		terrain_state[old_xi  + old_yi  * W].suspended,
		terrain_state[old_xi1 + old_yi  * W].suspended,
		terrain_state[old_xi  + old_yi1 * W].suspended,
		terrain_state[old_xi1 + old_yi1 * W].suspended,
		t_x, t_y);

	state_middle->suspended = old_s;
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
