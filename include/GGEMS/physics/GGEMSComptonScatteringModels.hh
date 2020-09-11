#ifndef GUARD_GGEMS_PHYSICS_GGEMSCOMPTONSCATTERINGMODELS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSCOMPTONSCATTERINGMODELS_HH

/*!
  \file GGEMSComptonScatteringModels.hh

  \brief Models for Compton scattering, only for OpenCL kernel usage

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Friday September 11, 2020
*/

#ifdef OPENCL_COMPILER

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
  \fn inline void KleinNishinaComptonSampleSecondaries(__global GGEMSPrimaryParticles* primary_particle, __global GGEMSRandom* random, __global GGEMSMaterialTables const* materials, GGint const index_particle)
  \param primary_particle - buffer of particles
  \param random - pointer on random numbers
  \param materials - buffer of materials
  \param index_particle - index of the particle
  \brief Klein Nishina Compton model, Effects due to binding of atomic electrons are negliged.
*/
inline void KleinNishinaComptonSampleSecondaries(
  __global GGEMSPrimaryParticles* primary_particle,
  __global GGEMSRandom* random,
  __global GGEMSMaterialTables const* materials,
  GGint const index_particle
)
{
  // Energy
  GGfloat const kE0 = primary_particle->E_[index_particle];
  GGfloat const kE0_MeC2 = kE0 / ELECTRON_MASS_C2;

  // Direction
  GGfloat3 const kGammaDirection = {
    primary_particle->dx_[index_particle],
    primary_particle->dy_[index_particle],
    primary_particle->dz_[index_particle]
  };

  // sample the energy rate the scattered gamma
  GGfloat const kEps0 = 1.0f / (1.0f + 2.0f*kE0_MeC2);
  GGfloat const kEps0Eps0 = kEps0*kEps0;
  GGfloat const kAlpha1 = -log(kEps0);
  GGfloat const kAlpha2 = kAlpha1 / (kAlpha1 + 0.5f*(1.0f-kEps0Eps0));

  #ifdef GGEMS_TRACKING
  if (index_particle == primary_particle->particle_tracking_id) {
    printf("\n");
    printf("[GGEMS OpenCL function KleinNishinaComptonSampleSecondaries] Photon energy: %e keV\n", kE0/keV);
    printf("[GGEMS OpenCL function KleinNishinaComptonSampleSecondaries] Min. photon energy (back scattering): %e keV\n", kE0*kEps0/keV);
  }
  #endif

/*    f32 gamE0 = particles->E[id];
    f32 E0 = gamE0 / 0.510998910f;
    f32xyz gamDir0 = make_f32xyz(particles->dx[id], particles->dy[id], particles->dz[id]);

    // sample the energy rate pf the scattered gamma

    f32 epszero = 1.0f / (1.0f + 2.0f * E0);
    f32 eps02 = epszero*epszero;
    f32 a1 = -logf(epszero);
    f32 a2 = a1 / (a1 + 0.5f*(1.0f-eps02));

    f32 greject, onecost, eps, eps2, sint2, cosTheta, sinTheta, phi;
    do {
        if (a2 > prng_uniform( particles, id )) {
            eps = expf(-a1 * prng_uniform( particles, id ));
            eps2 = eps*eps;
        } else {
            eps2 = eps02 + (1.0f - eps02) * prng_uniform( particles, id );
            eps = sqrt(eps2);
        }
        onecost = (1.0f - eps) / (eps * E0);
        sint2 = onecost * (2.0f - onecost);
        greject = 1.0f - eps * sint2 / (1.0f + eps2);
    } while (greject < prng_uniform( particles, id ));

    // scattered gamma angles

    if (sint2 < 0.0f) {sint2 = 0.0f;}
    cosTheta = 1.0f - onecost;
    sinTheta = sqrt(sint2);
    phi = prng_uniform( particles, id ) * gpu_twopi;

    // update the scattered gamma

    f32xyz gamDir1 = make_f32xyz(sinTheta*cosf(phi), sinTheta*sinf(phi), cosTheta);
    gamDir1 = rotateUz(gamDir1, gamDir0);
    gamDir1 = fxyz_unit( gamDir1 );

    particles->dx[id] = gamDir1.x;
    particles->dy[id] = gamDir1.y;
    particles->dz[id] = gamDir1.z;

    f32 gamE1  = gamE0 * eps;
    if (gamE1 > 1.0e-06f) {particles->E[id] = gamE1;}
    else {
        particles->status[id] = PARTICLE_DEAD;  // absorbed this particle
        particles->E[id] = gamE1;                // Local energy deposit
    }

    // kinematic of the scattered electron

    SecParticle electron;
    electron.pname = ELECTRON;
    electron.E = gamE0 - gamE1; // eKinE
    electron.dir = make_f32xyz(0.0, 0.0, 0.0);
    electron.endsimu = PARTICLE_DEAD;


    //               DBL_MIN                  cut production
    if (electron.E > 1.0e-38f && electron.E > cutE && flag_electron) {
        electron.dir = fxyz_sub(fxyz_scale(gamDir0, gamE0), fxyz_scale(gamDir1, gamE1));
        electron.dir = fxyz_unit(electron.dir);
        electron.endsimu = PARTICLE_ALIVE;
        //printf("Compton: new e-\n");
    }

    //return e-
    return electron;*/
}

#endif

#endif // GUARD_GGEMS_PHYSICS_GGEMSCOMPTONSCATTERINGMODELS_HH
