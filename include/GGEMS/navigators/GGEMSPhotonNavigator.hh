#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSPHOTONNAVIGATOR_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSPHOTONNAVIGATOR_HH

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
  \file GGEMSPhotonNavigator.hh

  \brief Functions for photon navigation, only for OpenCL kernel usage

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday June 24, 2020
*/

#ifdef __OPENCL_C_VERSION__

#include "GGEMS/maths/GGEMSMathAlgorithms.hh"

#include "GGEMS/randoms/GGEMSKissEngine.hh"

#include "GGEMS/physics/GGEMSComptonScatteringModels.hh"
#include "GGEMS/physics/GGEMSRayleighScatteringModels.hh"
#include "GGEMS/physics/GGEMSPhotoElectricEffectModels.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline void GetPhotonNextInteraction(global GGEMSPrimaryParticles* primary_particle, global GGEMSRandom* random, global GGEMSParticleCrossSections const* particle_cross_sections, GGshort const index_material, GGint const index_particle)
  \param primary_particle - buffer of particles
  \param random - pointer on random numbers
  \param particle_cross_sections - buffer of cross sections
  \param index_material - index of the material
  \param index_particle - index of the particle
  \brief Determine the next photon interaction
*/
inline void GetPhotonNextInteraction(
  global GGEMSPrimaryParticles* primary_particle,
  global GGEMSRandom* random,
  global GGEMSParticleCrossSections const* particle_cross_sections,
  GGshort const index_material,
  GGint const particle_id)
{
  // Getting energy of the particle and the index of energy in cross section table
  GGint energy_id = BinarySearchLeft(primary_particle->E_[particle_id], particle_cross_sections->energy_bins_, particle_cross_sections->number_of_bins_, 0, 0);

  // Initialization of next interaction distance
  GGfloat next_interaction_distance = OUT_OF_WORLD;
  GGchar next_discrete_process = NO_PROCESS;
  GGchar photon_process_id = 0;
  GGfloat interaction_distance = 0.0f;

  // Loop over activated processes
  for (GGchar i = 0; i < particle_cross_sections->number_of_activated_photon_processes_; ++i) {
    // Getting index of process
    photon_process_id = particle_cross_sections->photon_cs_id_[i];

    // Getting the interaction distance
    interaction_distance =
      -log(KissUniform(random, particle_id))/
      particle_cross_sections->photon_cross_sections_[photon_process_id][energy_id + particle_cross_sections->number_of_bins_*index_material];

    if (interaction_distance < next_interaction_distance) {
      next_interaction_distance = interaction_distance;
      next_discrete_process = photon_process_id;
    }
  }

  // Storing results in particle buffer
  primary_particle->E_index_[particle_id] = energy_id;
  primary_particle->next_interaction_distance_[particle_id] = next_interaction_distance;
  primary_particle->next_discrete_process_[particle_id] = next_discrete_process;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline void PhotonDiscreteProcess(global GGEMSPrimaryParticles* primary_particle, global GGEMSRandom* random, global GGEMSMaterialTables const* materials, global GGEMSParticleCrossSections const* particle_cross_sections, GGshort const material_id, GGint const particle_id)
  \param primary_particle - buffer of particles
  \param random - pointer on random numbers
  \param materials - buffer of materials
  \param particle_cross_sections - pointer to cross sections activated in navigator
  \param material_id - index of the material
  \param index_particle - index of the particle
  \brief Launch sampling depending on photon process
*/
inline void PhotonDiscreteProcess(
  global GGEMSPrimaryParticles* primary_particle,
  global GGEMSRandom* random,
  global GGEMSMaterialTables const* materials,
  global GGEMSParticleCrossSections const* particle_cross_sections,
  GGshort const material_id,
  GGint const particle_id
)
{
  // Get photon process
  GGchar next_iteraction_process = primary_particle->next_discrete_process_[particle_id];

  // Select process
  if (next_iteraction_process == COMPTON_SCATTERING) {
    KleinNishinaComptonSampleSecondaries(primary_particle, random, particle_id);
  }
  else if (next_iteraction_process == PHOTOELECTRIC_EFFECT) {
    StandardPhotoElectricSampleSecondaries(primary_particle, particle_id);
  }
  else if (next_iteraction_process == RAYLEIGH_SCATTERING) {
    LivermoreRayleighSampleSecondaries(primary_particle, random, materials, particle_cross_sections, material_id, particle_id);
  }
}

#endif

#endif // GUARD_GGEMS_NAVIGATORS_GGEMSPHOTONNAVIGATOR_HH
