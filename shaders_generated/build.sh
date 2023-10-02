#!/bin/sh
glslangValidator -V NaivePathTraceMega.comp -o NaivePathTraceMega.comp.spv -DGLSL -I.. -I/home/frol/PROG/HydraRepos/HydraCore3/external/LiteScene -I/home/frol/PROG/kernel_slicer/apps/LiteMathAux -I/home/frol/PROG/kernel_slicer/apps/LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
glslangValidator -V PathTraceMega.comp -o PathTraceMega.comp.spv -DGLSL -I.. -I/home/frol/PROG/HydraRepos/HydraCore3/external/LiteScene -I/home/frol/PROG/kernel_slicer/apps/LiteMathAux -I/home/frol/PROG/kernel_slicer/apps/LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
glslangValidator -V PackXYMega.comp -o PackXYMega.comp.spv -DGLSL -I.. -I/home/frol/PROG/HydraRepos/HydraCore3/external/LiteScene -I/home/frol/PROG/kernel_slicer/apps/LiteMathAux -I/home/frol/PROG/kernel_slicer/apps/LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
glslangValidator -V CastSingleRayMega.comp -o CastSingleRayMega.comp.spv -DGLSL -I.. -I/home/frol/PROG/HydraRepos/HydraCore3/external/LiteScene -I/home/frol/PROG/kernel_slicer/apps/LiteMathAux -I/home/frol/PROG/kernel_slicer/apps/LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
glslangValidator -V RayTraceMega.comp -o RayTraceMega.comp.spv -DGLSL -I.. -I/home/frol/PROG/HydraRepos/HydraCore3/external/LiteScene -I/home/frol/PROG/kernel_slicer/apps/LiteMathAux -I/home/frol/PROG/kernel_slicer/apps/LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 