
# Reading

### Fast Hydraulic Erosion Simulation and Visualization on GPU
by Xing Mei, Philippe Decaudin, Bao-Gang Hu
2011
https://inria.hal.science/inria-00402079/document


Introduces 'virtual pipe' method.

Does sediment transport with a semi-Lagrangian advection method.
A fundamental problem with the semi-Lagrangian advection method is that it does not conserve mass for fields with divergence.   
How this manifests is loss of water and soil/sediment over the course of the simulation.


### Fast Hydraulic and Thermal Erosion on the GPU
by Balázs Jákó
2011
http://www.cescg.org/CESCG-2011/papers/TUBudapest-Jako-Balazs.pdf

Similar to 'Fast Hydraulic Erosion Simulation and Visualization on GPU' but introduces max talus angle and 'thermal erosion': that is, soil is transported downhill if the hill angle is too large.
This is essential for realistic terrain features.


### Interactive Terrain Modeling Using Hydraulic Erosion
2008
Ondrej Štava, Bedrich Beneš, Matthew Brisbin, Jaroslav Krivánek

Introduces a technique for multiple layers of soil 

### Wikipedia

https://en.wikipedia.org/wiki/Sediment_transport


## Other terrain generation / erosion simulation implementations


### SoilMachine

https://github.com/weigert/SoilMachine

Advanced technique with multiple layers of differing soils, but runs on CPU, so may be relatively slow.

### Three Ways of Generating Terrain with Erosion Features

https://github.com/dandrino/terrain-erosion-3-ways

General overview of practical techniques including machine learning.

### Terrain erosion sandbox in WebGl

https://github.com/LanLou123/Webgl-Erosion/tree/master

Nice looking and interactive WebGL code.


### https://reef3d.wordpress.com/sflow/

Looks like actually rigorous and accurate academic/industrial code.
