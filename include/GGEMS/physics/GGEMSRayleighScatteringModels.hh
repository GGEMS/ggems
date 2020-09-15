#ifndef GUARD_GGEMS_PHYSICS_GGEMSRAYLEIGHSCATTERINGMODELS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSRAYLEIGHSCATTERINGMODELS_HH

/*!
  \file GGEMSRayleighScatteringModels.hh

  \brief Models for Rayleigh scattering, only for OpenCL kernel usage

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday September 14, 2020
*/

#ifdef OPENCL_COMPILER

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline void KleinNishinaComptonSampleSecondaries(__global GGEMSPrimaryParticles* primary_particle, __global GGEMSRandom* random, __global GGEMSMaterialTables const* materials, __global GGEMSParticleCrossSections const* particle_cross_sections, GGuchar const index_material, GGint const index_particle)
  \param primary_particle - buffer of particles
  \param random - pointer on random numbers
  \param materials - buffer of materials
  \param particle_cross_sections - pointer to cross sections activated in navigator
  \param index_material - index of the material
  \param index_particle - index of the particle
  \brief Klein Nishina Compton model, Effects due to binding of atomic electrons are negliged.
*/
inline void LivermoreRayleighSampleSecondaries(
  __global GGEMSPrimaryParticles* primary_particle,
  __global GGEMSRandom* random,
  __global GGEMSMaterialTables const* materials,
  __global GGEMSParticleCrossSections const* particle_cross_sections,
  GGuchar const index_material,
  GGint const index_particle
)
{
  GGfloat const kE0 = primary_particle->E_[index_particle];
  if (kE0 <= 250.0e-6f) { // 250 eV
    primary_particle->status_[index_particle] = DEAD;
    return;
  }

  GGuint const kNumberOfBins = particle_cross_sections->number_of_bins_;
  GGuchar const kNEltsMinusOne = materials->number_of_chemical_elements_[index_material]-1;
  GGushort const kMixtureID = materials->index_of_chemical_elements_[index_material];
  GGuint const kEnergyID = primary_particle->E_index_[index_particle];

  // Get last atom
  GGuchar selected_atomic_number_z = materials->atomic_number_Z_[kMixtureID+kNEltsMinusOne];

  // Select randomly one element that composed the material
  GGuchar i = 0;
  if (kNEltsMinusOne > 0) {
    // Get Cross Section of Livermore Rayleigh
    GGfloat const kCS = LinearInterpolation(
      particle_cross_sections->energy_bins_[kEnergyID],
      particle_cross_sections->photon_cross_sections_[RAYLEIGH_SCATTERING][kEnergyID + kNumberOfBins*index_material],
      particle_cross_sections->energy_bins_[kEnergyID+1],
      particle_cross_sections->photon_cross_sections_[RAYLEIGH_SCATTERING][kEnergyID+1 + kNumberOfBins*index_material],
      kE0
    );

    // Get a random
    GGfloat const x = KissUniform(random, index_particle) * kCS;

    GGfloat cross_section = 0.0f;
    while (i < kNEltsMinusOne) {
      GGuchar atomic_number_z = materials->atomic_number_Z_[kMixtureID+i];
      cross_section += materials->atomic_number_density_[kMixtureID+i] * LinearInterpolation(
        particle_cross_sections->energy_bins_[kEnergyID],
        particle_cross_sections->photon_cross_sections_per_atom_[RAYLEIGH_SCATTERING][kEnergyID + kNumberOfBins*atomic_number_z],
        particle_cross_sections->energy_bins_[kEnergyID+1],
        particle_cross_sections->photon_cross_sections_per_atom_[RAYLEIGH_SCATTERING][kEnergyID+1 + kNumberOfBins*atomic_number_z],
        kE0
      );

      if (x < cross_section) {
        selected_atomic_number_z = atomic_number_z;
        break;
      }
      ++i;
    }
  }

  #ifdef GGEMS_TRACKING
  if (index_particle == primary_particle->particle_tracking_id) {
    printf("\n");
    printf("[GGEMS OpenCL function LivermoreRayleighSampleSecondaries]     Photon energy: %e keV\n", kE0/keV);
    printf("[GGEMS OpenCL function LivermoreRayleighSampleSecondaries]     Direction: %e %e %e\n", primary_particle->dx_[index_particle], primary_particle->dy_[index_particle], primary_particle->dz_[index_particle]);
    printf("[GGEMS OpenCL function LivermoreRayleighSampleSecondaries]     Number of element in material %s: %d\n", particle_cross_sections->material_names_[index_material], materials->number_of_chemical_elements_[index_material]);
    printf("[GGEMS OpenCL function LivermoreRayleighSampleSecondaries]     Selected element: %u\n", selected_atomic_number_z);
  }
  #endif
}

#endif

#endif // GUARD_GGEMS_PHYSICS_GGEMSRAYLEIGHSCATTERINGMODELS_HH
