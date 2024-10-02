// ************************************************************************
// * This file is part of GGEMS.                                          *
// *                                                                      *
// * GGEMS is free software: you can redistribute it and/or modify        *
// * it under the terms of the GNU General Public License as published by *
// * the Free Software Foundation, either version 3 of the License, or    *
// * (at your option) any later version.                                  *
// *                                                                      *
// * GGEMS is distributed in the hope that it will be useful,             *
// * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
// * GNU General Public License for more details.                         *
// *                                                                      *
// * You should have received a copy of the GNU General Public License    *
// * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
// *                                                                      *
// ************************************************************************

/*!
  \file TrackThroughGGEMSMeshedSolid.cl

  \brief OpenCL kernel tracking particles within voxelized solid

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday June 16, 2020
*/

#include "GGEMS/physics/GGEMSPrimaryParticles.hh"
#include "GGEMS/geometries/GGEMSMeshedSolidData.hh"
#include "GGEMS/geometries/GGEMSRayTracing.hh"
#include "GGEMS/materials/GGEMSMaterialTables.hh"
#include "GGEMS/physics/GGEMSParticleCrossSections.hh"
#include "GGEMS/randoms/GGEMSRandom.hh"
#include "GGEMS/maths/GGEMSMatrixOperations.hh"
#include "GGEMS/navigators/GGEMSPhotonNavigator.hh"
#include "GGEMS/physics/GGEMSMuData.hh"

#if defined(DOSIMETRY)
#include "GGEMS/navigators/GGEMSDoseRecording.hh"
#endif

#define MAX_TRIANGLE_INTERACTION 16

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
inline float DotMesh(GGfloat3 v0, GGfloat3 v1)
{
  float dot_product = 0.0f;
  dot_product = v0.x * v1.x;
  dot_product += v0.y * v1.y;
  dot_product += v0.z * v1.z;
  return dot_product;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

inline GGfloat3 CrossMesh(GGfloat3 v0, GGfloat3 v1)
{
  GGfloat3 v = {
    (v0.y * v1.z) - (v0.z * v1.y),
    (v0.z * v1.x) - (v0.x * v1.z),
    (v0.x * v1.y) - (v0.y * v1.x)
  };
  return v;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

inline GGfloat DistanceSquare(GGfloat3 p0, GGfloat3 p1)
{
  return (p1.x - p0.x)*(p1.x - p0.x) + (p1.y - p0.y)*(p1.y - p0.y) + (p1.z - p0.z)*(p1.z - p0.z);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

inline int MollerTrumboreRayTriangle(GGfloat3 pi, GGfloat3 d, GGEMSTriangle3* triangle, GGfloat3* po)
{
  // E1 & E2
  GGfloat3 edge1 = {
    triangle->pts_[1].x_ - triangle->pts_[0].x_,
    triangle->pts_[1].y_ - triangle->pts_[0].y_,
    triangle->pts_[1].z_ - triangle->pts_[0].z_
  };

  GGfloat3 edge2 = {
    triangle->pts_[2].x_ - triangle->pts_[0].x_,
    triangle->pts_[2].y_ - triangle->pts_[0].y_,
    triangle->pts_[2].z_ - triangle->pts_[0].z_
  };

  // h = D (ray direction) X E2
  GGfloat3 h = CrossMesh(d, edge2);

  // a = (D X E2).E1
  GGfloat a = DotMesh(edge1, h);

  // If a == 0
  if (a > -EPSILON6 && a < EPSILON6) return 0;

  // f = 1 / (D X E2).E1
  GGfloat f = 1.0f/a;
  // s = T = O (ray_origin) - V0
  GGfloat3 s = {
    pi.x - triangle->pts_[0].x_,
    pi.y - triangle->pts_[0].y_,
    pi.z - triangle->pts_[0].z_
  };

  // u = f * (D X E2).T
  GGfloat u = f * DotMesh(s, h);

  if (u < 0.0f || u > 1.0f) return 0;

  // q = T X E1
  GGfloat3 q = CrossMesh(s, edge1);
  // v = f * (T X E1).D
  GGfloat v = f * DotMesh(q, d);

  if (v < 0.0f || u + v > 1.0f) return 0;

  GGfloat t = f * DotMesh(q, edge2);

  if (t > EPSILON6) {
    po->x = pi.x + d.x * t;
    po->y = pi.y + d.y * t;
    po->z = pi.z + d.z * t;
    return 1;
  }
  else {
    return 0;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

inline GGint TestLineNode(GGfloat3 p, GGfloat3 d, GGEMSNode node)
{
  GGfloat min[3] = {
    node.center_.x_ - node.half_width_[0],
    node.center_.y_ - node.half_width_[1],
    node.center_.z_ - node.half_width_[2]
  };

  GGfloat max[3] = {
    node.center_.x_ + node.half_width_[0],
    node.center_.y_ + node.half_width_[1],
    node.center_.z_ + node.half_width_[2]
  };

  GGfloat tmin = 0.0f; // set to 0 to get first hit on line
  GGfloat tmax = FLT_MAX; // set to max distance ray can travel (for segment)

  GGfloat direction[3] = {d.x, d.y, d.z};
  GGfloat position[3] = {p.x, p.y, p.z};

  // For all three slabs
  for (int i = 0; i < 3; ++i) {
    if (fabs(direction[i]) < EPSILON6) {
      // Ray is parallel to slab. No hit if origin not within slab
      if (position[i] < min[i] || position[i] > max[i]) return 0;
    } else {
      // Compute intersection t value of ray with near and far plane of slab
      GGfloat ood = 1.0f / direction[i];
      GGfloat t1 = (min[i] - position[i]) * ood;
      GGfloat t2 = (max[i] - position[i]) * ood;

      // Make t1 be intersection with near plane, t2 with far plane
      if (t1 > t2) {
        GGfloat tmp = t2;
        t2 = t1;
        t1 = tmp;
      }

      // Compute the intersection of slab intersection intervals
      if (t1 > tmin) tmin = t1;
      if (t2 < tmax) tmax = t2;

      // Exit with no collision as soon as slab intersection becomes empty
      if (tmin > tmax) return 0;
    }
  }

  // Ray intersects all 3 slabs
  return 1;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

inline GGint GetMeshIOPositions(GGfloat3 const p, GGfloat3 const d, GGEMSMeshedSolidData const* mesh_data, GGfloat3 mesh_pos[MAX_TRIANGLE_INTERACTION])
{
  // Get infos about nodes
  GGEMSNode* nodes = mesh_data->nodes_;
  GGint number_of_nodes = mesh_data->total_nodes_;

  GGint index_mesh_pos = 0;

  GGfloat3 point_on_mesh_surface;
  GGfloat nearest_point[MAX_TRIANGLE_INTERACTION];

  // Loop over node
  for (GGint n = 0; n < number_of_nodes; ++n) {
    if (!nodes[n].triangle_list_) continue; // No triangle in a node

    // Checking intersection line and node
    if (!TestLineNode(p, d, nodes[n])) continue;

    // Loop over triangle in node
    for (GGEMSTriangle3* t = nodes[n].triangle_list_; t; t = t->next_triangle_) {
      if (MollerTrumboreRayTriangle(p, d, t, &point_on_mesh_surface))
      {
        // Compute distance squared between 'p' and 'point_on_mesh_surface'
        GGfloat d2 = DistanceSquare(p, point_on_mesh_surface);
        GGint shift = 0;
        if (index_mesh_pos == 0) {
          mesh_pos[0].x = point_on_mesh_surface.x;
          mesh_pos[0].y = point_on_mesh_surface.y;
          mesh_pos[0].z = point_on_mesh_surface.z;
          nearest_point[0] = d2;
        } else {
          for (GGint i = index_mesh_pos - 1; i >= 0; --i) {
            // Compare distance
            if (d2 < nearest_point[i]) shift++;
            else i = 0;
          }

          if (shift == 0) {
            mesh_pos[index_mesh_pos].x = point_on_mesh_surface.x;
            mesh_pos[index_mesh_pos].y = point_on_mesh_surface.y;
            mesh_pos[index_mesh_pos].z = point_on_mesh_surface.z;
            nearest_point[index_mesh_pos] = d2;
          } else {
            for (GGint j = 0; j < shift; ++j) {
              mesh_pos[index_mesh_pos - j].x = mesh_pos[index_mesh_pos - j - 1].x;
              mesh_pos[index_mesh_pos - j].y = mesh_pos[index_mesh_pos - j - 1].y;
              mesh_pos[index_mesh_pos - j].z = mesh_pos[index_mesh_pos - j - 1].z;
              nearest_point[index_mesh_pos - j] = nearest_point[index_mesh_pos - j - 1];
            }
            mesh_pos[index_mesh_pos - shift].x = point_on_mesh_surface.x;
            mesh_pos[index_mesh_pos - shift].y = point_on_mesh_surface.y;
            mesh_pos[index_mesh_pos - shift].z = point_on_mesh_surface.z;
            nearest_point[index_mesh_pos - shift] = d2;
          }
        }
        index_mesh_pos++;
        if (index_mesh_pos == MAX_TRIANGLE_INTERACTION) break;
      }
    }
  }

  return index_mesh_pos; // Number of triangles interacting with line
}

/*!
  \fn kernel void track_through_ggems_meshed_solid(GGsize const particle_id_limit, global GGEMSPrimaryParticles* primary_particle, global GGEMSRandom* random, global GGEMSMeshedSolidData const* meshed_solid_data, global GGuchar const* label_data, global GGEMSParticleCrossSections const* particle_cross_sections, global GGEMSMaterialTables const* materials, global GGEMSMuMuEnData const* attenuations, GGfloat const threshold)
  \param particle_id_limit - particle id limit
  \param primary_particle - pointer to primary particles on OpenCL memory
  \param random - pointer on random numbers
  \param meshed_solid_data - pointer to meshed solid data
  \param label_data - pointer storing label of material
  \param particle_cross_sections - pointer to cross sections activated in navigator
  \param materials - pointer on material in navigator
  \param attenuations - pointer on attenuation values
  \param threshold - energy threshold
  \brief OpenCL kernel tracking particles within meshed solid
*/
kernel void track_through_ggems_meshed_solid(
  GGsize const particle_id_limit,
  global GGEMSPrimaryParticles* primary_particle,
  global GGEMSRandom* random,
  global GGEMSMeshedSolidData const* meshed_solid_data,
  global GGuchar const* label_data,
  global GGEMSParticleCrossSections const* particle_cross_sections,
  global GGEMSMaterialTables const* materials,
  global GGEMSMuMuEnData const* attenuations,
  GGfloat const threshold
  #ifdef DOSIMETRY
  ,global GGEMSDoseParams* dose_params,
  global GGDosiType* edep_tracking,
  global GGDosiType* edep_squared_tracking,
  global GGint* hit_tracking,
  global GGint* photon_tracking
  #endif
)
{
  // Getting index of thread
  GGsize global_id = get_global_id(0);

  // Return if index > to particle limit
  if (global_id >= particle_id_limit) return;

  // Checking if the current navigator is the selected navigator
  if (primary_particle->solid_id_[global_id] != meshed_solid_data->solid_id_) return;

  // Checking status of particle
  if (primary_particle->status_[global_id] == DEAD) {
    #if defined(GGEMS_TRACKING)
    if (global_id == primary_particle->particle_tracking_id) {
      printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] ################################################################################\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] The particle id %d is dead!!!\n", global_id);
    }
    #endif
    return;
  }

  // Get the position and direction in local OBB coordinate
  // In mesh navigation global == local
  GGfloat3 local_position = {
    primary_particle->px_[global_id],
    primary_particle->py_[global_id],
    primary_particle->pz_[global_id]
  };

  GGfloat3 local_direction = {
    primary_particle->dx_[global_id],
    primary_particle->dy_[global_id],
    primary_particle->dz_[global_id]
  };

  GGfloat3 mesh_surface_pos[MAX_TRIANGLE_INTERACTION];
  GGfloat3 next_triangle_pos;

  // Loop for navigation of particles to meshed volume
  do {
    // Get position of interaction with triangles
    GGint n_inter_triangles = GetMeshIOPositions(local_position, local_direction, meshed_solid_data, mesh_surface_pos);

    if (n_inter_triangles == 0) { // No iteraction particle/meshed volume
      primary_particle->particle_solid_distance_[global_id] = OUT_OF_WORLD; // Reset to initiale value
      primary_particle->solid_id_[global_id] = -1; // Out of world
      break;
    }

    // n_inter_triangles even : particle outside meshed volume
    // n_inter_triangles odd : particle inside meshed volume
    if (n_inter_triangles%2 == 0) { // Move particles to first triangle
      local_position.x = mesh_surface_pos[0].x;
      local_position.y = mesh_surface_pos[0].y;
      local_position.z = mesh_surface_pos[0].z;

      next_triangle_pos.x = mesh_surface_pos[1].x;
      next_triangle_pos.y = mesh_surface_pos[1].y;
      next_triangle_pos.z = mesh_surface_pos[1].z;
    } else {
      next_triangle_pos.x = mesh_surface_pos[0].x;
      next_triangle_pos.y = mesh_surface_pos[0].y;
      next_triangle_pos.z = mesh_surface_pos[0].z;
    }

    // Find next discrete photon interaction
    GetPhotonNextInteraction(primary_particle, random, particle_cross_sections, 0, global_id);
    GGfloat next_interaction_distance = primary_particle->next_interaction_distance_[global_id];
    GGchar next_discrete_process = primary_particle->next_discrete_process_[global_id];

    // Computing distance to next triangle
    GGfloat dX = (next_triangle_pos.x - local_position.x);
    GGfloat dY = (next_triangle_pos.y - local_position.y);
    GGfloat dZ = (next_triangle_pos.z - local_position.z);
    GGfloat distance_to_next_triangle = sqrt(dX*dX + dY*dY + dZ*dZ);

    // If distance to next triangle is inferior to distance to next interaction we move particle to next triangle
    if (distance_to_next_triangle <= next_interaction_distance) {
      next_interaction_distance = distance_to_next_triangle + GEOMETRY_TOLERANCE;
      next_discrete_process = TRANSPORTATION;
    // #if defined(DOSIMETRY)
    //  if (photon_tracking) dose_photon_tracking(dose_params, photon_tracking, &local_position);
    //  #endif
    }

    #if defined(GGEMS_TRACKING)
    if (global_id == primary_particle->particle_tracking_id) {
      printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] ################################################################################\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Particle id: %d\n", global_id);
      printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Particle type: ");
      if (primary_particle->pname_[global_id] == PHOTON) printf("gamma\n");
      else if (primary_particle->pname_[global_id] == ELECTRON) printf("e-\n");
      else if (primary_particle->pname_[global_id] == POSITRON) printf("e+\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Local position (x, y, z): %e %e %e mm\n", local_position.x/mm, local_position.y/mm, local_position.z/mm);
      printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Local direction (x, y, z): %e %e %e\n", local_direction.x, local_direction.y, local_direction.z);
      printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Energy: %e keV\n", primary_particle->E_[global_id]/keV);
      printf("\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Solid id: %u\n", meshed_solid_data->solid_id_);
      //printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Nb voxels: %u %u %u\n", number_of_voxels.x, number_of_voxels.y, number_of_voxels.z);
      //printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Voxel size: %e %e %e mm\n", voxel_size.x/mm, voxel_size.y/mm, voxel_size.z/mm);
      //printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Solid X Borders: %e %e mm\n", border_min.x/mm, border_max.x/mm);
      //printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Solid Y Borders: %e %e mm\n", border_min.y/mm, border_max.y/mm);
      //printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Solid Z Borders: %e %e mm\n", border_min.z/mm, border_max.z/mm);
      //printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Voxel X Borders: %e %e mm\n", voxel_border_min.x/mm, voxel_border_max.x/mm);
      //printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Voxel Y Borders: %e %e mm\n", voxel_border_min.y/mm, voxel_border_max.y/mm);
      //printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Voxel Z Borders: %e %e mm\n", voxel_border_min.z/mm, voxel_border_max.z/mm);
      //printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Index of current voxel (x, y, z): %d %d %d\n", voxel_id.x, voxel_id.y, voxel_id.z);
      printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Material: %s\n", particle_cross_sections->material_names_[0]);
      printf("\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Next process: ");
      if (next_discrete_process == COMPTON_SCATTERING) printf("COMPTON_SCATTERING\n");
      if (next_discrete_process == PHOTOELECTRIC_EFFECT) printf("PHOTOELECTRIC_EFFECT\n");
      if (next_discrete_process == RAYLEIGH_SCATTERING) printf("RAYLEIGH_SCATTERING\n");
      if (next_discrete_process == TRANSPORTATION) printf("TRANSPORTATION\n");
      printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Next interaction distance: %e mm\n", next_interaction_distance/mm);
      printf("[GGEMS OpenCL kernel track_through_ggems_meshed_solid] Distance to next triangle: %e mm\n", distance_to_next_triangle/mm);
    }
    #endif

    // Moving particle to next position
    local_position = local_position + local_direction*next_interaction_distance;

    // Storing new position in local
    primary_particle->px_[global_id] = local_position.x + GEOMETRY_TOLERANCE;
    primary_particle->py_[global_id] = local_position.y + GEOMETRY_TOLERANCE;
    primary_particle->pz_[global_id] = local_position.z + GEOMETRY_TOLERANCE;

    //#if defined(DOSIMETRY)
    //GGfloat initial_energy = primary_particle->E_[global_id];
    //#endif

    // Resolve process if different of TRANSPORTATION
    if (next_discrete_process != TRANSPORTATION) {

      PhotonDiscreteProcess(primary_particle, random, materials, particle_cross_sections, 0, global_id);

      // If process is COMPTON_SCATTERING or RAYLEIGH_SCATTERING scatter order is incremented
      if (next_discrete_process == COMPTON_SCATTERING || next_discrete_process == RAYLEIGH_SCATTERING)
      {
        primary_particle->scatter_[global_id] = TRUE;
      }

      //#if defined(DOSIMETRY)
      //GGfloat edep = initial_energy - primary_particle->E_[global_id];
      //dose_record_standard(dose_params, edep_tracking, edep_squared_tracking, hit_tracking, edep, &local_position);
      //#endif

      local_direction.x = primary_particle->dx_[global_id];
      local_direction.y = primary_particle->dy_[global_id];
      local_direction.z = primary_particle->dz_[global_id];

      #if defined(OPENGL)
      if (global_id < MAXIMUM_DISPLAYED_PARTICLES) {
        // Storing OpenGL index on OpenCL private memory
        GGint stored_particles_gl = primary_particle->stored_particles_gl_[global_id];

        // Checking if buffer is full
        if (stored_particles_gl != MAXIMUM_INTERACTIONS) {
          primary_particle->px_gl_[global_id*MAXIMUM_INTERACTIONS+stored_particles_gl] = local_position.x;
          primary_particle->py_gl_[global_id*MAXIMUM_INTERACTIONS+stored_particles_gl] = local_position.y;
          primary_particle->pz_gl_[global_id*MAXIMUM_INTERACTIONS+stored_particles_gl] = local_position.z;

          // Storing final index
          primary_particle->stored_particles_gl_[global_id] += 1;
        }
      }
      #endif
    }

    //#if defined(DOSIMETRY) 
    //GGint E_index = BinarySearchLeft(initial_energy, attenuations->energy_bins_, attenuations->number_of_bins_, 0, 0);
    //GGfloat mu_en = 0.0f;
    //if (E_index == 0) {
    //  mu_en = attenuations->mu_en_[material_id*attenuations->number_of_bins_];
    //}
    //else {
    //  mu_en = LinearInterpolation(
    //    attenuations->energy_bins_[E_index-1], attenuations->mu_en_[material_id*attenuations->number_of_bins_ + E_index-1],
    //    attenuations->energy_bins_[E_index], attenuations->mu_en_[material_id*attenuations->number_of_bins_ + E_index],
    //    initial_energy
    //  );
    //}
    //GGfloat edep = initial_energy * mu_en * next_interaction_distance * 0.1f;
    //dose_record_standard(dose_params, edep_tracking, edep_squared_tracking, hit_tracking, edep, &local_position);
    //#endif

    // Apply threshold
    if (primary_particle->E_[global_id] <= materials->photon_energy_cut_[0]) {
    // #if defined(DOSIMETRY)
    // dose_record_standard(dose_params, edep_tracking, edep_squared_tracking, hit_tracking, primary_particle->E_[global_id], &local_position);
    // #endif
      primary_particle->status_[global_id] = DEAD;
    }
  } while (primary_particle->status_[global_id] == ALIVE);

  // Compute distance between particles and voxelized navigator
  GGfloat distance = ComputeDistanceToOBB(&local_position, &local_direction, &meshed_solid_data->obb_geometry_);

  // Project particles to limit of volume
  primary_particle->px_[global_id] = local_position.x + local_direction.x * (distance + GEOMETRY_TOLERANCE);
  primary_particle->py_[global_id] = local_position.y + local_direction.y * (distance + GEOMETRY_TOLERANCE);
  primary_particle->pz_[global_id] = local_position.z + local_direction.z * (distance + GEOMETRY_TOLERANCE);

  primary_particle->dx_[global_id] = local_direction.x;
  primary_particle->dy_[global_id] = local_direction.y;
  primary_particle->dz_[global_id] = local_direction.z;
}
