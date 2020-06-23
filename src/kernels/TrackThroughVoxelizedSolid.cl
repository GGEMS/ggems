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
  __global GGEMSVoxelizedSolidData const* voxelized_solid_data,
  __global GGuchar const* label_data,
  __global GGEMSParticleCrossSections const* particle_cross_sections,
  __global GGEMSMaterialTables const* materials)
{
  // Getting index of thread
  GGint const kGlobalIndex = get_global_id(0);

  // Checking status of particle
  if (primary_particle->status_[kGlobalIndex] == DEAD) return;

  // Checking if the current navigator is the selected navigator
  if (primary_particle->navigator_id_[kGlobalIndex] != voxelized_solid_data->navigator_id_) return;

  // Position of particle
  GGfloat3 position;
  position.x = primary_particle->px_[kGlobalIndex];
  position.y = primary_particle->py_[kGlobalIndex];
  position.z = primary_particle->pz_[kGlobalIndex];

  // Direction of particle
  GGfloat3 direction;
  direction.x = primary_particle->dx_[kGlobalIndex];
  direction.y = primary_particle->dy_[kGlobalIndex];
  direction.z = primary_particle->dz_[kGlobalIndex];

  // Get index of voxelized phantom, x, y, z and w (global index)
  GGint4 index_voxel;
  index_voxel.x = (GGint)((position.x + voxelized_solid_data->position_xyz_.x) / voxelized_solid_data->voxel_sizes_xyz_.x);
  index_voxel.y = (GGint)((position.y + voxelized_solid_data->position_xyz_.y) / voxelized_solid_data->voxel_sizes_xyz_.y);
  index_voxel.z = (GGint)((position.z + voxelized_solid_data->position_xyz_.z) / voxelized_solid_data->voxel_sizes_xyz_.z);
  index_voxel.w = index_voxel.x
    + index_voxel.y * voxelized_solid_data->number_of_voxels_xyz_.x
    + index_voxel.z * voxelized_solid_data->number_of_voxels_xyz_.x * voxelized_solid_data->number_of_voxels_xyz_.y;

  // Get the material that compose this volume
  GGuchar index_label = label_data[index_voxel.w];

  printf("******\n");
  printf("TRACK THROUGH\n");
  printf("Navigator: %u\n", voxelized_solid_data->navigator_id_);
  printf("Nb voxels: %u %u %u\n", voxelized_solid_data->number_of_voxels_xyz_.x, voxelized_solid_data->number_of_voxels_xyz_.y, voxelized_solid_data->number_of_voxels_xyz_.z);
  printf("Voxel size: %4.7f %4.7f %4.7f mm\n", voxelized_solid_data->voxel_sizes_xyz_.x, voxelized_solid_data->voxel_sizes_xyz_.y, voxelized_solid_data->voxel_sizes_xyz_.z);
  printf("Border X: %4.7f %4.7f mm\n", voxelized_solid_data->border_min_xyz_.x, voxelized_solid_data->border_max_xyz_.x);
  printf("Border Y: %4.7f %4.7f mm\n", voxelized_solid_data->border_min_xyz_.y, voxelized_solid_data->border_max_xyz_.y);
  printf("Border Z: %4.7f %4.7f mm\n", voxelized_solid_data->border_min_xyz_.z, voxelized_solid_data->border_max_xyz_.z);
  printf("Position: %4.7f %4.7f %4.7f mm\n", position.x, position.y, position.z);
  printf("Direction: %4.7f %4.7f %4.7f mm\n", direction.x, direction.y, direction.z);
  printf("Index voxel: %d %d %d %d\n", index_voxel.x, index_voxel.y, index_voxel.z, index_voxel.w);
  printf("Label voxel: %d %u\n", index_label, materials->total_number_of_chemical_elements_);
  printf("Material name voxel: %s\n", particle_cross_sections->material_names_[index_label]);

/*
    // Read position
    f32xyz pos;
    pos.x = particles->px[part_id];
    pos.y = particles->py[part_id];
    pos.z = particles->pz[part_id];

    // Read direction
    f32xyz dir;
    dir.x = particles->dx[part_id];
    dir.y = particles->dy[part_id];
    dir.z = particles->dz[part_id];

    // Defined index phantom
    f32xyz ivoxsize;
    ivoxsize.x = 1.0 / vol->spacing_x;
    ivoxsize.y = 1.0 / vol->spacing_y;
    ivoxsize.z = 1.0 / vol->spacing_z;
    ui32xyzw index_phantom;
    index_phantom.x = ui32 ( ( pos.x+vol->off_x ) * ivoxsize.x );
    index_phantom.y = ui32 ( ( pos.y+vol->off_y ) * ivoxsize.y );
    index_phantom.z = ui32 ( ( pos.z+vol->off_z ) * ivoxsize.z );

    index_phantom.w = index_phantom.z*vol->nb_vox_x*vol->nb_vox_y
                      + index_phantom.y*vol->nb_vox_x
                      + index_phantom.x; // linear index

    // Get the material that compose this volume
    ui16 mat_id = vol->values[ index_phantom.w ];

    //// Find next discrete interaction ///////////////////////////////////////

    photon_get_next_interaction ( particles, parameters, photon_CS_table, mat_id, part_id );
    f32 next_interaction_distance = particles->next_interaction_distance[part_id];
    ui8 next_discrete_process = particles->next_discrete_process[part_id];

    //// Get the next distance boundary volume /////////////////////////////////

    // get voxel params
    f32 vox_xmin = index_phantom.x*vol->spacing_x - vol->off_x;
    f32 vox_ymin = index_phantom.y*vol->spacing_y - vol->off_y;
    f32 vox_zmin = index_phantom.z*vol->spacing_z - vol->off_z;
    f32 vox_xmax = vox_xmin + vol->spacing_x;
    f32 vox_ymax = vox_ymin + vol->spacing_y;
    f32 vox_zmax = vox_zmin + vol->spacing_z;

    // get a safety position for the particle within this voxel (sometime a particle can be right between two voxels)
    // TODO: In theory this have to be applied just at the entry of the particle within the volume
    //       in order to avoid particle entry between voxels. Then, computing improvement can be made
    //       by calling this function only once, just for the particle step=0.    - JB
    pos = transport_get_safety_inside_AABB( pos, vox_xmin, vox_xmax,
                                            vox_ymin, vox_ymax, vox_zmin, vox_zmax, parameters->geom_tolerance );

    // compute the next distance boundary
    f32 boundary_distance = hit_ray_AABB( pos, dir, vox_xmin, vox_xmax,
                                          vox_ymin, vox_ymax, vox_zmin, vox_zmax );

    if ( boundary_distance <= next_interaction_distance )
    {
        next_interaction_distance = boundary_distance + parameters->geom_tolerance; // Overshoot
        next_discrete_process = GEOMETRY_BOUNDARY;
    }

    //// Move particle //////////////////////////////////////////////////////

    // get the new position
    pos = fxyz_add ( pos, fxyz_scale ( dir, next_interaction_distance ) );

    // get safety position (outside the current voxel)
    pos = transport_get_safety_outside_AABB( pos, vox_xmin, vox_xmax,
                                             vox_ymin, vox_ymax, vox_zmin, vox_zmax, parameters->geom_tolerance );

    // update tof
    particles->tof[part_id] += c_light * next_interaction_distance;

    // store new position
    particles->px[part_id] = pos.x;
    particles->py[part_id] = pos.y;
    particles->pz[part_id] = pos.z;

    // Stop simulation if out of the phantom
    if ( !test_point_AABB_with_tolerance( pos, vol->xmin, vol->xmax, vol->ymin, vol->ymax,
                                          vol->zmin, vol->zmax, parameters->geom_tolerance ) )
    {
        particles->status[part_id] = PARTICLE_FREEZE;
        return;
    }

    //// Apply discrete process //////////////////////////////////////////////////

    if ( next_discrete_process != GEOMETRY_BOUNDARY )
    {        

        // Resolve discrete process
        SecParticle electron = photon_resolve_discrete_process ( particles, parameters, photon_CS_table,
                                                                 materials, mat_id, part_id );

        // If the process is PHOTON_COMPTON or PHOTON_RAYLEIGH the scatter
        // order is incremented
        if( next_discrete_process == PHOTON_COMPTON
                || next_discrete_process == PHOTON_RAYLEIGH )
        {
            particles->scatter_order[ part_id ] += 1;
        }

        //// Here e- are not tracked, and lost energy not drop
        //// Energy cut
        if ( particles->E[ part_id ] <= materials->photon_energy_cut[ mat_id ])
        {
            // kill without mercy (energy not drop)
            particles->status[part_id] = PARTICLE_DEAD;
            return;
        }
    }    
*/
}
