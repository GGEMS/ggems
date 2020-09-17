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
#include "GGEMS/randoms/GGEMSRandomStack.hh"
#include "GGEMS/maths/GGEMSMatrixOperations.hh"
#include "GGEMS/navigators/GGEMSPhotonNavigator.hh"

/*!
  \fn __kernel void track_through_voxelized_solid(__global GGEMSPrimaryParticles* primary_particle, __global GGEMSRandom* random, __global GGEMSVoxelizedSolidData* voxelized_solid_data, __global GGEMSParticleCrossSections* particle_cross_sections, __global GGEMSMaterialTables* materials)
  \param primary_particle - pointer to primary particles on OpenCL memory
  \param random - pointer on random numbers
  \param voxelized_solid_data - pointer to voxelized solid data
  \param particle_cross_sections - pointer to cross sections activated in navigator
  \param materials - pointer on material in navigator
  \brief OpenCL kernel tracking particles within voxelized solid
  \return no returned value
*/
__kernel void track_through_voxelized_solid(
  __global GGEMSPrimaryParticles* primary_particle,
  __global GGEMSRandom* random,
  __global GGEMSVoxelizedSolidData const* voxelized_solid_data,
  __global GGuchar const* label_data,
  __global GGEMSParticleCrossSections const* particle_cross_sections,
  __global GGEMSMaterialTables const* materials)
{
  // Getting index of thread
  GGint const kParticleID = get_global_id(0);

  // Checking if the current navigator is the selected navigator
  if (primary_particle->navigator_id_[kParticleID] != voxelized_solid_data->navigator_id_) return;

  // Checking status of particle
  if (primary_particle->status_[kParticleID] == DEAD) {
    #ifdef GGEMS_TRACKING
    if (kParticleID == primary_particle->particle_tracking_id) {
      printf("[GGEMS Kernel track_through_voxelized_solid] The particle id %d is dead before track to out step!!!\n", kParticleID);
    }
    #endif
    return;
  }

  // Get tolerance of navigator
  GGfloat const kTolerance = voxelized_solid_data->tolerance_;

  // Track particle until out of navigator
  while (primary_particle->status_[kParticleID] == ALIVE) {
    // Position of particle
    GGfloat3 position = {
      primary_particle->px_[kParticleID],
      primary_particle->py_[kParticleID],
      primary_particle->pz_[kParticleID]
    };

    // Direction of particle
    GGfloat3 direction = {
      primary_particle->dx_[kParticleID],
      primary_particle->dy_[kParticleID],
      primary_particle->dz_[kParticleID]
    };

    // Get index of voxelized phantom, x, y, z and w (global index)
    GGint4 kIndexVoxel = {0, 0, 0, 0};
    kIndexVoxel.x = (GGint)((position.x + voxelized_solid_data->position_xyz_.x) / voxelized_solid_data->voxel_sizes_xyz_.x);
    kIndexVoxel.y = (GGint)((position.y + voxelized_solid_data->position_xyz_.y) / voxelized_solid_data->voxel_sizes_xyz_.y);
    kIndexVoxel.z = (GGint)((position.z + voxelized_solid_data->position_xyz_.z) / voxelized_solid_data->voxel_sizes_xyz_.z);
    kIndexVoxel.w = kIndexVoxel.x
      + kIndexVoxel.y * voxelized_solid_data->number_of_voxels_xyz_.x
      + kIndexVoxel.z * voxelized_solid_data->number_of_voxels_xyz_.x * voxelized_solid_data->number_of_voxels_xyz_.y;

    // Get the material that compose this volume
    GGuchar const kMaterialID = label_data[kIndexVoxel.w];

    // Find next discrete photon interaction
    GetPhotonNextInteraction(primary_particle, random, particle_cross_sections, kMaterialID, kParticleID);
    GGfloat next_interaction_distance = primary_particle->next_interaction_distance_[kParticleID];
    GGuchar next_discrete_process = primary_particle->next_discrete_process_[kParticleID];

    // Get the borders of the current voxel
    GGfloat const kXMinVoxel = kIndexVoxel.x*voxelized_solid_data->voxel_sizes_xyz_.x - voxelized_solid_data->position_xyz_.x;
    GGfloat const kYMinVoxel = kIndexVoxel.y*voxelized_solid_data->voxel_sizes_xyz_.y - voxelized_solid_data->position_xyz_.y;
    GGfloat const kZMinVoxel = kIndexVoxel.z*voxelized_solid_data->voxel_sizes_xyz_.z - voxelized_solid_data->position_xyz_.z;
    GGfloat const kXMaxVoxel = kXMinVoxel + voxelized_solid_data->voxel_sizes_xyz_.x;
    GGfloat const kYMaxVoxel = kYMinVoxel + voxelized_solid_data->voxel_sizes_xyz_.y;
    GGfloat const kZMaxVoxel = kZMinVoxel + voxelized_solid_data->voxel_sizes_xyz_.z;

    // Get safety position of particle to be sure particle is inside voxel
    TransportGetSafetyInsideAABB(&position, kXMinVoxel, kXMaxVoxel, kYMinVoxel, kYMaxVoxel, kZMinVoxel, kZMaxVoxel, kTolerance);

    // Get the distance to next boundary
    GGfloat const distance_to_next_boundary = ComputeDistanceToAABB(&position, &direction, kXMinVoxel, kXMaxVoxel, kYMinVoxel, kYMaxVoxel, kZMinVoxel, kZMaxVoxel, kTolerance);

    // If distance to next boundary is inferior to distance to next interaction
    // we move particle to boundary
    if (distance_to_next_boundary <= next_interaction_distance) {
      next_interaction_distance = distance_to_next_boundary + kTolerance;
      next_discrete_process = TRANSPORTATION;
    }

    #ifdef GGEMS_TRACKING
    if (kParticleID == primary_particle->particle_tracking_id) {
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] ################################################################################\n");
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Particle id: %d\n", kParticleID);
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Particle type: ");
      if (primary_particle->pname_[kParticleID] == PHOTON) printf("gamma\n");
      else if (primary_particle->pname_[kParticleID] == ELECTRON) printf("e-\n");
      else if (primary_particle->pname_[kParticleID] == POSITRON) printf("e+\n");
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Position (x, y, z): %e %e %e mm\n", position.x/mm, position.y/mm, position.z/mm);
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Direction (x, y, z): %e %e %e\n", direction.x, direction.y, direction.z);
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Energy: %e keV\n", primary_particle->E_[kParticleID]/keV);
      printf("\n");
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Navigator id: %u\n", voxelized_solid_data->navigator_id_);
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Nb voxels: %u %u %u\n", voxelized_solid_data->number_of_voxels_xyz_.x, voxelized_solid_data->number_of_voxels_xyz_.y, voxelized_solid_data->number_of_voxels_xyz_.z);
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Voxel size: %e %e %e mm\n", voxelized_solid_data->voxel_sizes_xyz_.x/mm, voxelized_solid_data->voxel_sizes_xyz_.y/mm, voxelized_solid_data->voxel_sizes_xyz_.z/mm);
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Navigator X Borders: %e %e mm\n", voxelized_solid_data->border_min_xyz_.x/mm, voxelized_solid_data->border_max_xyz_.x/mm);
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Navigator Y Borders: %e %e mm\n", voxelized_solid_data->border_min_xyz_.y/mm, voxelized_solid_data->border_max_xyz_.y/mm);
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Navigator Z Borders: %e %e mm\n", voxelized_solid_data->border_min_xyz_.z/mm, voxelized_solid_data->border_max_xyz_.z/mm);
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Voxel X Borders: %e %e mm\n", kXMinVoxel/mm, kXMaxVoxel/mm);
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Voxel Y Borders: %e %e mm\n", kYMinVoxel/mm, kYMaxVoxel/mm);
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Voxel Z Borders: %e %e mm\n", kZMinVoxel/mm, kZMaxVoxel/mm);
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Index of current voxel (x, y, z): %d %d %d\n", kIndexVoxel.x, kIndexVoxel.y, kIndexVoxel.z);
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Global Index of current voxel: %d\n", kIndexVoxel.w);
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Material in voxel: %s\n", particle_cross_sections->material_names_[kMaterialID]);
      printf("\n");
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Next process: ");
      if (next_discrete_process == COMPTON_SCATTERING) printf("COMPTON_SCATTERING\n");
      if (next_discrete_process == PHOTOELECTRIC_EFFECT) printf("PHOTOELECTRIC_EFFECT\n");
      if (next_discrete_process == RAYLEIGH_SCATTERING) printf("RAYLEIGH_SCATTERING\n");
      if (next_discrete_process == TRANSPORTATION) printf("TRANSPORTATION\n");
      printf("[GGEMS OpenCL kernel track_through_voxelized_solid] Next interaction distance: %e mm\n", next_interaction_distance/mm);
    }
    #endif

    // Moving particle to next postion
    position = GGfloat3Add(position, GGfloat3Scale(direction, next_interaction_distance));

    // Get safety position of particle to be sure particle is outside voxel
    TransportGetSafetyOutsideAABB(&position, kXMinVoxel, kXMaxVoxel, kYMinVoxel, kYMaxVoxel, kZMinVoxel, kZMaxVoxel, kTolerance);

    // Update TOF, true for photon only
    primary_particle->tof_[kParticleID] += next_interaction_distance * C_LIGHT;

    // Storing new position
    primary_particle->px_[kParticleID] = position.x;
    primary_particle->py_[kParticleID] = position.y;
    primary_particle->pz_[kParticleID] = position.z;

    // Checking if particle outside voxelized solid navigator
    if (!IsParticleInVoxelizedNavigator(&position, voxelized_solid_data)) {
      primary_particle->particle_navigator_distance_[kParticleID] = OUT_OF_WORLD; // Reset to initiale value
      primary_particle->navigator_id_[kParticleID] = 255; // Out of world navigator
      break;
    }

    // Resolve process if different of TRANSPORTATION
    if (next_discrete_process != TRANSPORTATION) {
      PhotonDiscreteProcess(primary_particle, random, materials, particle_cross_sections, kMaterialID, kParticleID);
    }
  }
}
