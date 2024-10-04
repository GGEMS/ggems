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
  \file GetPrimariesGGEMSXRaySource.cl

  \brief OpenCL kernel generating primaries for X-Ray source

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 22, 2019
*/

#include "GGEMS/physics/GGEMSPrimaryParticles.hh"
#include "GGEMS/randoms/GGEMSKissEngine.hh"
#include "GGEMS/maths/GGEMSReferentialTransformation.hh"
#include "GGEMS/maths/GGEMSMathAlgorithms.hh"
#include "GGEMS/physics/GGEMSParticleConstants.hh"
#include "GGEMS/physics/GGEMSDirectionConstants.hh"
#include "GGEMS/physics/GGEMSProcessConstants.hh"

/*!
  \fn kernel void get_primaries_ggems_xray_source(GGsize const particle_id_limit, global GGEMSPrimaryParticles* primary_particle,
  global GGEMSRandom* random, GGchar const particle_name, global GGfloat const* energy_spectrum, global GGfloat const* cdf,
  GGint const number_of_energy_bins, GGfloat const aperture, GGfloat3 const focal_spot_size, global GGfloat44 const* matrix_transformation,
  GGchar const direction_type, global GGfloat const* theta_cdf, global GGfloat const* theta_angles, global GGfloat const* phi_cdf,
  global GGfloat const* phi_angles, GGint const theta_angles_size, GGint const phi_angles_size)
  \param particle_id_limit - particle id limit
  \param primary_particle - buffer of primary particles
  \param random - buffer for random number
  \param particle_name - name of particle
  \param energy_spectrum - energy spectrum
  \param cdf - cumulative derivative function
  \param number_of_energy_bins - number of energy bins
  \param aperture - source aperture
  \param focal_spot_size - focal spot size of xray-source
  \param matrix_transformation - matrix storing information about axis
  \param direction_type - direction type for the source
  \param theta_cdf - cumulative derivative function for theta weights
  \param theta_angles - source theta angles
  \param phi_cdf - cumulative derivative function for phi weights
  \param phi_angles - source phi angles
  \param theta_angles_size - size of the theta angles array
  \param phi_angles_size - size of the phi angles array
  \brief Generate primaries for xray source
*/
kernel void get_primaries_ggems_xray_source(
  GGsize const particle_id_limit,
  global GGEMSPrimaryParticles* primary_particle,
  global GGEMSRandom* random,
  GGchar const particle_name,
  global GGfloat const* energy_spectrum,
  global GGfloat const* cdf,
  GGint const number_of_energy_bins,
  GGfloat const aperture,
  GGfloat3 const focal_spot_size,
  global GGfloat44 const* matrix_transformation,
  GGchar const direction_type,
  global GGfloat const* theta_cdf,
  global GGfloat const* theta_angles,
  global GGfloat const* phi_cdf,
  global GGfloat const* phi_angles,
  GGint const theta_angles_size,
  GGint const phi_angles_size
)
{
  // Get the index of thread
  GGsize global_id = get_global_id(0);

  // Return if index > to particle limit
  if (global_id >= particle_id_limit) return;

  // Get direction of the cone beam. The beam is targeted to the isocenter,
  // so the direction is directly related to the position of the source.
  // Local position of X-ray source is 0 0 0
  GGfloat3 global_position = {0.0f, 0.0f, 0.0f};
  global_position = LocalToGlobalPosition(matrix_transformation, &global_position);
  GGfloat3 direction = normalize((GGfloat3)(0.0f, 0.0f, 0.0f) - global_position);

  if (direction_type == 0) { // Isotropic
    // Get random angles
    GGdouble phi = KissUniform(random, global_id);
    GGdouble theta = KissUniform(random, global_id);

    phi *= (GGdouble)TWO_PI;
    GGdouble new_aperture = 1.0 - cos((GGdouble)aperture);
    theta = acos(1.0 - new_aperture*theta);

    // Compute rotation
    GGfloat3 rotation = {
        cos(phi) * sin(theta),
        sin(phi) * sin(theta),
        cos(theta)
    };

    // Apply deflection (global coordinate)
    direction = RotateUnitZ(&rotation, &direction);
    direction = normalize(direction);
  } else if (direction_type == 1) { //Histogram
    GGint not_accepted_events = 0;
    GGint max_not_accepted_events = 10000;
    bool accept_angle = false;
    while (!accept_angle) {
      if(not_accepted_events > max_not_accepted_events) {
        // "Too many events not accepted"
        return;
      }
      not_accepted_events++;
      // Getting a random for theta and phi angle
      GGfloat rndm_for_theta = KissUniform(random, global_id);
      GGfloat rndm_for_phi = KissUniform(random, global_id);

      // Get index in cdf
      GGint index_for_theta = BinarySearchLeft(rndm_for_theta, theta_cdf, theta_angles_size, 0, 0);
      GGint index_for_phi = BinarySearchLeft(rndm_for_phi, phi_cdf, phi_angles_size, 0, 0);

      // Setting theta and phi
      GGdouble theta = (index_for_theta == theta_angles_size - 1) ?
        theta_angles[index_for_theta] :
        LinearInterpolation(theta_cdf[index_for_theta], theta_angles[index_for_theta],
                            theta_cdf[index_for_theta + 1], theta_angles[index_for_theta + 1],
                            rndm_for_theta);

      GGdouble phi = (index_for_phi == phi_angles_size - 1) ?
        phi_angles[index_for_phi] :
        LinearInterpolation(phi_cdf[index_for_phi], phi_angles[index_for_phi],
                            phi_cdf[index_for_phi + 1], phi_angles[index_for_phi + 1],
                            rndm_for_phi);

      // Compute rotation
      GGfloat3 rotation = (GGfloat3)(
          cos(phi) * sin(theta),
          cos(theta),
          sin(phi) * sin(theta)
      );

      // Apply deflection (global coordinate)
      direction = normalize((GGfloat3)(0.0f, 0.0f, 0.0f) - global_position);
      direction = RotateUnitZ(&rotation, &direction);
      direction = normalize(direction);

      GGfloat3 normal_vector = normalize((GGfloat3)(1.0f, 0.0f, 0.0f) - global_position);
      GGfloat acceptance_angle = acos(dot(direction, normal_vector));

      GGfloat epsilon = 1e-6f;
      // Check if the angle is accepted or aperture is too small
      if (acceptance_angle <= aperture || fabs(aperture) < epsilon) {
          accept_angle = true;
          not_accepted_events = 0;
          primary_particle->dx_[global_id] = direction.x;
          primary_particle->dy_[global_id] = direction.y;
          primary_particle->dz_[global_id] = direction.z;
      }
    }
  }
  

  // Position with focal (local)
  global_position.x = focal_spot_size.x * (KissUniform(random, global_id) - 0.5f);
  global_position.y = focal_spot_size.y * (KissUniform(random, global_id) - 0.5f);
  global_position.z = focal_spot_size.z * (KissUniform(random, global_id) - 0.5f);

  // Apply transformation (local to global frame)
  global_position = LocalToGlobalPosition(matrix_transformation, &global_position);

  // Getting a random energy
  GGfloat rndm_for_energy = KissUniform(random, global_id);

  // Get index in cdf
  GGint index_for_energy = BinarySearchLeft(rndm_for_energy, cdf, number_of_energy_bins, 0, 0);

  // Setting the energy for particles
  primary_particle->E_[global_id] = (index_for_energy == number_of_energy_bins - 1) ?
    energy_spectrum[index_for_energy] :
    LinearInterpolation(cdf[index_for_energy], energy_spectrum[index_for_energy], cdf[index_for_energy + 1], energy_spectrum[index_for_energy + 1], rndm_for_energy);

  // Then set the mandatory field to create a new particle
  primary_particle->px_[global_id] = global_position.x;
  primary_particle->py_[global_id] = global_position.y;
  primary_particle->pz_[global_id] = global_position.z;

  primary_particle->dx_[global_id] = direction.x;
  primary_particle->dy_[global_id] = direction.y;
  primary_particle->dz_[global_id] = direction.z;

  primary_particle->scatter_[global_id] = FALSE;

  primary_particle->status_[global_id] = ALIVE;

  primary_particle->level_[global_id] = PRIMARY;
  primary_particle->pname_[global_id] = particle_name;

  primary_particle->particle_solid_distance_[global_id] = OUT_OF_WORLD;
  primary_particle->next_discrete_process_[global_id] = NO_PROCESS;
  primary_particle->next_interaction_distance_[global_id] = 0.0f;

  #ifdef OPENGL
  // Storing vertex position for OpenGL
  if (global_id < MAXIMUM_DISPLAYED_PARTICLES) {
    primary_particle->stored_particles_gl_[global_id] = 0;

    for (GGint i = 0; i < MAXIMUM_INTERACTIONS; ++i) {
      primary_particle->px_gl_[global_id*MAXIMUM_INTERACTIONS+i] = 0.0f;
      primary_particle->py_gl_[global_id*MAXIMUM_INTERACTIONS+i] = 0.0f;
      primary_particle->pz_gl_[global_id*MAXIMUM_INTERACTIONS+i] = 0.0f;
    }

    // Storing OpenGL index on OpenCL private memory
    //GGint stored_particles_gl = primary_particle->stored_particles_gl_[global_id];

    // Checking if buffer is full
   // if (stored_particles_gl != MAXIMUM_INTERACTIONS) {

      primary_particle->px_gl_[global_id*MAXIMUM_INTERACTIONS] = primary_particle->px_[global_id];
      primary_particle->py_gl_[global_id*MAXIMUM_INTERACTIONS] = primary_particle->py_[global_id];
      primary_particle->pz_gl_[global_id*MAXIMUM_INTERACTIONS] = primary_particle->pz_[global_id];
      //stored_particles_gl += 1;

      // primary_particle->px_gl_[global_id*MAXIMUM_INTERACTIONS+stored_particles_gl] += primary_particle->dx_[global_id]*2.0f*m;
      // primary_particle->py_gl_[global_id*MAXIMUM_INTERACTIONS+stored_particles_gl] += primary_particle->dy_[global_id]*2.0f*m;
      // primary_particle->pz_gl_[global_id*MAXIMUM_INTERACTIONS+stored_particles_gl] += primary_particle->dz_[global_id]*2.0f*m;
      // stored_particles_gl += 1;

      // Storing final index
      primary_particle->stored_particles_gl_[global_id] = 1;
    //}
  }
  #endif

  #ifdef GGEMS_TRACKING
  if (global_id == primary_particle->particle_tracking_id) {
    printf("[GGEMS OpenCL kernel get_primaries_ggems_xray_source] ################################################################################\n");
    printf("[GGEMS OpenCL kernel get_primaries_ggems_xray_source] Particle id: %d\n", global_id);
    printf("[GGEMS OpenCL kernel get_primaries_ggems_xray_source] Particle type: ");
    if (primary_particle->pname_[global_id] == PHOTON) printf("gamma\n");
    else if (primary_particle->pname_[global_id] == ELECTRON) printf("e-\n");
    else if (primary_particle->pname_[global_id] == POSITRON) printf("e+\n");
    printf("[GGEMS OpenCL kernel get_primaries_ggems_xray_source] Position (x, y, z): %e %e %e mm\n", global_position.x/mm, global_position.y/mm, global_position.z/mm);
    printf("[GGEMS OpenCL kernel get_primaries_ggems_xray_source] Direction (x, y, z): %e %e %e\n", direction.x, direction.y, direction.z);
    printf("[GGEMS OpenCL kernel get_primaries_ggems_xray_source] Energy: %e keV\n", primary_particle->E_[global_id]/keV);
  }
  #endif
}
