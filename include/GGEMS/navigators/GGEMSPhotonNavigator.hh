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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline void GetPhotonNextInteraction(__global GGEMSPrimaryParticles* primary_particle, __global GGEMSParticleCrossSections const* particle_cross_sections, GGuchar const index_material, GGint const index_particle)
  \param primary_particle - buffer of particles
  \param particle_cross_sections - buffer of cross sections
  \param index_material - index of the material
  \param index_particle - index of the particle
  \brief Determine the next photon interaction
*/
inline void GetPhotonNextInteraction(
  __global GGEMSPrimaryParticles* primary_particle,
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

  printf("**** GetPhotonNextInteraction ****\n");
  printf("Energy: %e\n", kEnergy);
  printf("Energy index: %u\n", kIndexEnergy);
}

#endif

#endif // GUARD_GGEMS_NAVIGATORS_GGEMSPHOTONNAVIGATOR_HH
