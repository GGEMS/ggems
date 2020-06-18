/*!
  \file TrackThroughVoxelizedSolid.cl

  \brief OpenCL kernel tracking particles within voxelized solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday June 16, 2020
*/

#include "GGEMS/physics/GGEMSPrimaryParticlesStack.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolidStack.hh"
#include "GGEMS/materials/GGEMSMaterialsStack.hh"
#include "GGEMS/physics/GGEMSParticleCrossSectionsStack.hh"
#include "GGEMS/geometries/GGEMSRayTracing.hh"

/*!
  \fn __kernel void track_through_voxelized_solid(__global GGEMSPrimaryParticles* primary_particle, __global GGEMSVoxelizedSolidData* voxelized_solid_data, __global GGEMSParticleCrossSections* particle_cross_sections, __global GGEMSMaterialTables* materials)
  \param primary_particle - pointer to primary particles on OpenCL memory
  \param voxelized_solid_data - pointer to voxelized solid data
  \param particle_cross_sections - pointer to cross sections activated in navigator
  \param materials - pointer on material in navigator
  \brief OpenCL kernel tracking particles within voxelized solid
  \return no returned value
*/
__kernel void track_through_voxelized_solid(
  __global GGEMSPrimaryParticles* primary_particle,
  __global GGEMSVoxelizedSolidData* voxelized_solid_data,
  __global GGEMSParticleCrossSections* particle_cross_sections,
  __global GGEMSMaterialTables* materials)
{
  // Getting index of thread
  GGint const kGlobalIndex = get_global_id(0);
}
