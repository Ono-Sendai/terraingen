
For the origin of the pipe stuff, see 'Fast Hydraulic Erosion Simulation and Visualization on GPU'.

See also [Reading / Literature](reading.md)

## Acceleration from water depth differences

Pressure in cell at depth $h$ is 

$$
p(h) = h \rho g + p_0
$$ 

Where $p_0$ is atmospheric pressure, $\rho$ is density, and $g$ is magnitude of gravitational accell. (See https://pressbooks.online.ucf.edu/osuniversityphysics/chapter/14-1-fluids-density-and-pressure/)

Consider a 'pipe' of length $l$, connecting cells 1 and 2.
Suppose pipe has width $w$ and height $h_p$.

The pressure on the water in the pipe at cell 1 at height z above ground level is

$$
p_1(z) = p(h_1 - z) = (h_1 - z) \rho g + p_0
$$

The pressure on the water in the pipe at cell 2 at height z above ground level is


$$
p_2(z) = p(h_2 - z) = (h_2 - z) \rho g + p_0
$$

Assuming pipe connects from the bottom of cell 1 to bottom of cell 2, the pressure difference at height z is

$$
p_2(z) - p_1(z) = ((h_2 - z) \rho g + p_0) - ((h_1 - z) \rho g + p_0)
$$

$$
p_2(z) - p_1(z) = (h_2 - h_1) \rho g
$$

Since force = pressure * area,
The net force on the fluid in the pipe in the positive x direction is 

$$
f_1 - f_2 = ((h_1 - h_2) \rho g) A
$$

Where the pipe cross-sectional area is $A = w \times h_p$

The mass of the fluid in the pipe is 
$$m = V \rho =  A  l \rho$$

The acceleration on the fluid is (from $f = ma$):

$$a = (f_1 - f_2)/m = {(h_1 - h_2) \rho g A \over  A l \rho }
= (h_1 - h_2) g / l
$$

This is independent of the height of the pipe ($h_p$), and the fluid density ($\rho$).

The volume flux (volume flowing through the pipe per unit time), is:

$$F_{1,2}$$      

(units of $m^3 s^{-1}$)

The flow velocity is

$$v = F_{1,2} / A$$

e.g.

$$F_{1,2} = v A$$

the rate of change of flux is

$${dF_{1,2} \over dt} = {dvA \over dt}
$$

Assuming A is constant (not correct but will do for now)
gives

$${dF_{1,2} \over dt} = A {dv \over dt} = A (h_1 - h_2) g / l = 
w h_p (h_1 - h_2) g / l
$$

Note that this depends on h_p, the height of the tube.  
If we assume that h_p = current fluid height in the cell, then the rate of flux change depends on the fluid height.

If we assume the width of the pipe ($w$) = pipe length ($l$) we get

$${dF_{1,2} \over dt} = (h_1 - h_2) h_p g
$$

For a time step of $\Delta_t$, we have

$$ dF_{1,2}^{t+\Delta_t} = dF_{1,2}^t + {dF_{1,2} \over dt} \Delta_t = $$

$$ dF_{1,2}^{t+\Delta_t} = dF_{1,2}^t + (h_1 - h_2) h_p g \Delta_t
$$

## Acceleration from ground height differences

Assume the ground heightfield is h(x, y)

The non-unit length normal is 

$$n = (-\partial h(x, y)/\partial x, -\partial h(x, y)/\partial y, 1)
$$

with 

$$||n|| = \sqrt{(\partial h/\partial x)^2 + (\partial h/\partial y)^2 + 1}
$$


The x component of the gravity accel vector projected onto the ground plane is

$$ a_x = g (-\partial h/\partial x/||n||)
$$

Suppose we have ground heights $g_1$ and $g_2$ under cells 1 and 2, and they are spaced $l$ apart.
Then 

$$\partial h/\partial x = (g_2 - g_1) / l
$$

And so 

$$a_x = g (-(g_2 - g_1) / l) / ||n||
$$

For small ground slope angles $||n|| \simeq 1$, so

$$
a_x = g (g_1 - g_2) / l
$$

Which is basically the same result as for water depth differences.

## Friction force

Assuming friction force for water in a pipe is proportional to the top-down surface area of the cell ($w \times w$), as well as the average water velocity, the friction force is

$$
f_f = -w^2 k_f v
$$

where $K_f$ is a friction coefficient, and $v$ is the (average) fluid velocity.

This results in an acceleration of

$$
a = f_f/m = {-w^2 k_f v \over w^2 h_p \rho } = {-k_f v \over h_p \rho }
$$

Folding the denisty $\rho$ into the friction constant:
$$
a = {-k_f' v \over h_p \rho }
$$

