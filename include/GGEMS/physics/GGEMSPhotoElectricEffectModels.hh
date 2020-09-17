#ifndef GUARD_GGEMS_PHYSICS_GGEMSPHOTOELECTRICEFFECTMODELS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSPHOTOELECTRICEFFECTMODELS_HH

/*!
  \file GGEMSPhotoElectricEffectModels.hh

  \brief Models for PhotoElectric effect, only for OpenCL kernel usage

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday September 17, 2020
*/

#ifdef OPENCL_COMPILER

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline void StandardPhotoElectricSampleSecondaries(__global GGEMSPrimaryParticles* primary_particle, GGint const index_particle)
  \param primary_particle - buffer of particles
  \param index_particle - index of the particle
  \brief Standard Photoelectric model
*/
inline void StandardPhotoElectricSampleSecondaries(
  __global GGEMSPrimaryParticles* primary_particle,
  GGint const index_particle
)
{
  primary_particle->status_[index_particle] = DEAD;
}

#endif

#endif // GUARD_GGEMS_PHYSICS_GGEMSCOMPTONSCATTERINGMODELS_HH
