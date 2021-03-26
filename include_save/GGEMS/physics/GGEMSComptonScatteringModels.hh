#ifndef GUARD_GGEMS_PHYSICS_GGEMSCOMPTONSCATTERINGMODELS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSCOMPTONSCATTERINGMODELS_HH

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
  \file GGEMSComptonScatteringModels.hh

  \brief Models for Compton scattering, only for OpenCL kernel usage

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Friday September 11, 2020
*/

#ifdef __OPENCL_C_VERSION__

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline void KleinNishinaComptonSampleSecondaries(global GGEMSPrimaryParticles* primary_particle, global GGEMSRandom* random, GGint const particle_id)
  \param primary_particle - buffer of particles
  \param random - pointer on random numbers
  \param particle_id - index of the particle
  \brief Klein Nishina Compton model, Effects due to binding of atomic electrons are negliged.
*/
inline void KleinNishinaComptonSampleSecondaries(
  global GGEMSPrimaryParticles* primary_particle,
  global GGEMSRandom* random,
  GGint const particle_id
)
{
  // Energy
  GGfloat kE0 = primary_particle->E_[particle_id];
  GGfloat kE0_MeC2 = kE0 / ELECTRON_MASS_C2;

  // Direction
  GGfloat3 kGammaDirection = {
    primary_particle->dx_[particle_id],
    primary_particle->dy_[particle_id],
    primary_particle->dz_[particle_id]
  };

  // sample the energy rate the scattered gamma
  GGfloat kEps0 = 1.0f / (1.0f + 2.0f*kE0_MeC2);
  GGfloat kEps0Eps0 = kEps0*kEps0;
  GGfloat kAlpha1 = -log(kEps0);
  GGfloat kAlpha2 = kAlpha1 + 0.5f*(1.0f-kEps0Eps0);

  #ifdef GGEMS_TRACKING
  if (particle_id == primary_particle->particle_tracking_id) {
    printf("\n");
    printf("[GGEMS OpenCL function KleinNishinaComptonSampleSecondaries]     Photon energy: %e keV\n", kE0/keV);
    printf("[GGEMS OpenCL function KleinNishinaComptonSampleSecondaries]     Photon direction: %e %e %e\n", kGammaDirection.x, kGammaDirection.y, kGammaDirection.z);
    printf("[GGEMS OpenCL function KleinNishinaComptonSampleSecondaries]     Min. photon energy (back scattering): %e keV\n", kE0*kEps0/keV);
  }
  #endif

  // sample the energy rate of the scattered gamma

  GGfloat3 rndm;
  GGfloat epsilon, epsilonsq, onecost, sint2, greject, costheta, sintheta, phi;
  GGint nloop = 0;
  do {
    ++nloop;
    // false interaction if too many iterations
    if (nloop > 1000) return;

    // Get 3 random numbers
    rndm.x = KissUniform(random, particle_id);
    rndm.y = KissUniform(random, particle_id);
    rndm.z = KissUniform(random, particle_id);

    if (kAlpha1 > kAlpha2*rndm.x) {
      epsilon = exp(-kAlpha1*rndm.y);
      epsilonsq = epsilon*epsilon; 
    }
    else {
      epsilonsq = kEps0Eps0 + (1.0f - kEps0Eps0)*rndm.y;
      epsilon = sqrt(epsilonsq);
    }

    onecost = (1.0f - epsilon)/(epsilon*kE0_MeC2);
    sint2 = onecost*(2.0f-onecost);
    greject = 1.0f - epsilon*sint2/(1.0f+ epsilonsq);
  } while (greject < rndm.z);

  // Scattered gamma angles
  if (sint2 < 0.0f) sint2 = 0.0f;
  costheta = 1.0f - onecost;
  sintheta = sqrt(sint2);
  phi = KissUniform(random, particle_id) * TWO_PI;

  // Update scattered gamma
  GGfloat3 gamma_direction = {sintheta*cos(phi), sintheta*sin(phi), costheta};
  gamma_direction = RotateUnitZ(&gamma_direction, &kGammaDirection);
  gamma_direction = normalize(gamma_direction);
  GGfloat kE1 = kE0*epsilon;

  #ifdef GGEMS_TRACKING
  if (particle_id == primary_particle->particle_tracking_id) {
    printf("[GGEMS OpenCL function KleinNishinaComptonSampleSecondaries]     Scattered photon energy: %e keV\n", kE1/keV);
    printf("[GGEMS OpenCL function KleinNishinaComptonSampleSecondaries]     Scattered photon direction: %e %e %e\n", gamma_direction.x, gamma_direction.y, gamma_direction.z);
  }
  #endif

  primary_particle->E_[particle_id] = kE1;

  primary_particle->dx_[particle_id] = gamma_direction.x;
  primary_particle->dy_[particle_id] = gamma_direction.y;
  primary_particle->dz_[particle_id] = gamma_direction.z;
}

#endif

#endif // GUARD_GGEMS_PHYSICS_GGEMSCOMPTONSCATTERINGMODELS_HH
