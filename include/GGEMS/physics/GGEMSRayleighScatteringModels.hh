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
inline void LivermoreRayleighSampleSecondarie(
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

  // Select randomly one element that composed the material
  GGuint const kNumberOfBins = particle_cross_sections->number_of_bins_;
  GGuchar const kNElts = materials->number_of_chemical_elements_[index_material];
  GGushort const kMixtureID = materials->index_of_chemical_elements_[index_material];
  GGuchar z = materials->atomic_number_Z_[kMixtureID];
  GGuint index_energy = primary_particle->E_index_[index_particle];

  GGuchar i = 0;
  if (kNElts > 0) {
    GGfloat x = /*0.3/*KissUniform(random, index_particle)*/ LinearInterpolation(
      particle_cross_sections->energy_bins_[index_energy],
      particle_cross_sections->photon_cross_sections_[2][index_energy + kNumberOfBins*index_material],
      particle_cross_sections->energy_bins_[index_energy+1],
      particle_cross_sections->photon_cross_sections_[2][index_energy+1 + kNumberOfBins*index_material],
      kE0 
    );

    GGfloat xsec = 0.0f;

    while (i < kNElts) {
      z = materials->atomic_number_Z_[kMixtureID+i];
      printf("next z: %u\n", z);
      printf("x: %e\n", x);
      xsec += particle_cross_sections->photon_cross_sections_per_atom_[2][index_energy + kNumberOfBins*z];
      printf("xsec: %e\n", particle_cross_sections->photon_cross_sections_per_atom_[2][index_energy + kNumberOfBins*z]);
      if (x <= xsec) break;
      ++i;
    }
  }

  #ifdef GGEMS_TRACKING
  if (index_particle == primary_particle->particle_tracking_id) {
    printf("\n");
    printf("[GGEMS OpenCL function LivermoreRayleighSampleSecondarie]     Photon energy: %e keV\n", kE0/keV);
    printf("[GGEMS OpenCL function LivermoreRayleighSampleSecondarie]     Direction: %e %e %e\n", primary_particle->dx_[index_particle], primary_particle->dy_[index_particle], primary_particle->dz_[index_particle]);
    printf("[GGEMS OpenCL function LivermoreRayleighSampleSecondarie]     Number of element in material %s: %d\n", particle_cross_sections->material_names_[index_material], materials->number_of_chemical_elements_[index_material]);
    printf("[GGEMS OpenCL function LivermoreRayleighSampleSecondarie]     Selected element: %u\n", z);
  }
  #endif

    /*ui32 n = mat->nb_elements[matindex]-1;    
    ui32 mixture_index = mat->index[matindex];
    ui32 Z = mat->mixture[mixture_index];


    ui32 i = 0;
    if (n > 0) {

        f32 x = prng_uniform( particles, id ) * linear_interpolation(photon_CS_table->E_bins[E_index-1],
                                                                     photon_CS_table->Rayleigh_Lv_CS[E_index-1],
                                                                     photon_CS_table->E_bins[E_index],
                                                                     photon_CS_table->Rayleigh_Lv_CS[E_index],
                                                                     particles->E[id]);
        f32 xsec = 0.0f;
        while (i < n) {
            Z = mat->mixture[mixture_index+i];
            xsec += photon_CS_table->Rayleigh_Lv_xCS[Z*photon_CS_table->nb_bins + E_index];
            if (x <= xsec) break;
            ++i;
        }

    }*/

}

#endif

#endif // GUARD_GGEMS_PHYSICS_GGEMSRAYLEIGHSCATTERINGMODELS_HH
