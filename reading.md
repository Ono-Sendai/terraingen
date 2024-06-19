
# Reading

### Fast Hydraulic Erosion Simulation and Visualization on GPU
by Xing Mei, Philippe Decaudin, Bao-Gang Hu: 
https://inria.hal.science/inria-00402079/document

Introduces 'virtual pipe' method.

Does sediment transport with a semi-Lagrangian advection method.
A fundamental problem with the semi-Lagrangian advection method is that it does not conserve mass for fields with divergence.   
How this manifests is loss of water and soil/sediment over the course of the simulation.


### Fast Hydraulic and Thermal Erosion on the GPU
by Balázs Jákó
http://www.cescg.org/CESCG-2011/papers/TUBudapest-Jako-Balazs.pdf

Similar to 'Fast Hydraulic Erosion Simulation and Visualization on GPU' but introduces max talus angle and 'thermal erosion': that is, soil is transported downhill if the hill angle is too large.
This is essential for realistic terrain features.