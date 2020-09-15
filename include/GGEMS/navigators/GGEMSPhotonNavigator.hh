#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSPHOTONNAVIGATOR_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSPHOTONNAVIGATOR_HH

/*!
  \file GGEMSPhotonNavigator.hh

  \brief Functions for photon navigation, only for OpenCL kernel usage

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday June 24, 2020
*/

#ifdef OPENCL_COMPILER

#include "GGEMS/maths/GGEMSMathAlgorithms.hh"
#include "GGEMS/randoms/GGEMSKissEngine.hh"
#include "GGEMS/physics/GGEMSComptonScatteringModels.hh"
#include "GGEMS/physics/GGEMSRayleighScatteringModels.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline void GetPhotonNextInteraction(__global GGEMSPrimaryParticles* primary_particle, __global GGEMSRandom* random, __global GGEMSParticleCrossSections const* particle_cross_sections, GGuchar const index_material, GGint const index_particle)
  \param primary_particle - buffer of particles
  \param random - pointer on random numbers
  \param particle_cross_sections - buffer of cross sections
  \param index_material - index of the material
  \param index_particle - index of the particle
  \brief Determine the next photon interaction
*/
inline void GetPhotonNextInteraction(
  __global GGEMSPrimaryParticles* primary_particle,
  __global GGEMSRandom* random,
  __global GGEMSParticleCrossSections const* particle_cross_sections,
  GGuchar const index_material,
  GGint const index_particle)
{
  // Getting energy of the particle and the index of energy in cross section table
  GGfloat const kEnergy = primary_particle->E_[index_particle];
  GGuint const kIndexEnergy = BinarySearchLeft(
    kEnergy,
    particle_cross_sections->energy_bins_,  // energy values in cross section table
    particle_cross_sections->number_of_bins_, // Number of bins in cross section table
    0,
    0
  );

  // Initialization of next interaction distance
  GGfloat next_interaction_distance = OUT_OF_WORLD;
  GGuchar next_discrete_process = NO_PROCESS;
  GGuchar index_photon_process = 0;
  GGfloat cross_section = 0.0f;
  GGfloat interaction_distance = 0.0f;

  // Loop over activated processes
  for (GGuchar i = 0; i < particle_cross_sections->number_of_activated_photon_processes_; ++i) {
    // Getting index of process
    index_photon_process = particle_cross_sections->index_photon_cs_[i];

    // Getting cross section
    cross_section = particle_cross_sections->photon_cross_sections_[index_photon_process][kIndexEnergy + particle_cross_sections->number_of_bins_*index_material];

    // Getting the interaction distance
    interaction_distance = -log(KissUniform(random, index_particle))/cross_section;

    if (interaction_distance < next_interaction_distance) {
      next_interaction_distance = interaction_distance;
      next_discrete_process = index_photon_process;
    }
  }

  // Storing results in particle buffer
  primary_particle->E_index_[index_particle] = kIndexEnergy;
  primary_particle->next_interaction_distance_[index_particle] = next_interaction_distance;
  primary_particle->next_discrete_process_[index_particle] = next_discrete_process;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline void PhotonDiscreteProcess(__global GGEMSPrimaryParticles* primary_particle, __global GGEMSRandom* random, __global GGEMSMaterialTables const* materials, __global GGEMSParticleCrossSections const* particle_cross_sections, GGuchar const index_material, GGint const index_particle)
  \param primary_particle - buffer of particles
  \param random - pointer on random numbers
  \param materials - buffer of materials
  \param particle_cross_sections - pointer to cross sections activated in navigator
  \param index_material - index of the material
  \param index_particle - index of the particle
  \brief Launch sampling depending on photon process
*/
inline void PhotonDiscreteProcess(
  __global GGEMSPrimaryParticles* primary_particle,
  __global GGEMSRandom* random,
  __global GGEMSMaterialTables const* materials,
  __global GGEMSParticleCrossSections const* particle_cross_sections,
  GGuchar const index_material,
  GGint const index_particle
)
{
  // Get photon process
  GGuchar const kNextInteractionProcess = primary_particle->next_discrete_process_[index_particle];

  // Select process
  if (kNextInteractionProcess == COMPTON_SCATTERING) {
    KleinNishinaComptonSampleSecondaries(primary_particle, random, index_particle);
  }
  else if (kNextInteractionProcess == PHOTOELECTRIC_EFFECT) {
    ;
  }
  else if (kNextInteractionProcess == RAYLEIGH_SCATTERING) {
    LivermoreRayleighSampleSecondaries(primary_particle, random, materials, particle_cross_sections, index_material, index_particle);
  }
}

#endif

#endif // GUARD_GGEMS_NAVIGATORS_GGEMSPHOTONNAVIGATOR_HH
