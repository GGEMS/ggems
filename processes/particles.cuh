// GGEMS Copyright (C) 2015

/*!
 * \file particles.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */

#ifndef PARTICLES_CUH
#define PARTICLES_CUH

#include "global.cuh"

// Stack of particles, format data is defined as SoA
struct ParticlesData
{
    // property
    f32* E;
    f32* dx;
    f32* dy;
    f32* dz;
    f32* px;
    f32* py;
    f32* pz;
    f32* tof;

    // PRNG
    ui32* prng_state_1;
    ui32* prng_state_2;
    ui32* prng_state_3;
    ui32* prng_state_4;
    ui32* prng_state_5;

    // Scatter for imaging
    ui32* scatter_order;

    // Navigation
    ui32* geometry_id; // current geometry crossed by the particle
    ui32* E_index;     // Energy index within CS and Mat tables
    // Interactions
    f32* next_interaction_distance;
    ui8* next_discrete_process;
    // simulation
    ui8* endsimu;
    ui8* level;
    ui8* pname; // particle name (photon, electron, etc)
    // size
    ui32 size;

    // Secondaries stack
    // Acces to level : Part_ID * size + hierarchy level
    f32* sec_E; // size * hierarchy level
    f32* sec_dx;
    f32* sec_dy;
    f32* sec_dz;
    f32* sec_px;
    f32* sec_py;
    f32* sec_pz;
    f32* sec_tof;
    ui8* sec_pname; // particle name (photon, electron, etc)

}; //

// Struct that handles particles
struct Particles
{
    ParticlesData data_h;
    ParticlesData data_d;

    ui32 size;
};

// Helper to handle secondaries particles
struct SecParticle
{
    f32xyz dir;
    f32 E;
    ui8 pname;
    ui8 endsimu;
};

// Particles class
class ParticleManager
{
public:
    ParticleManager();

    void initialize(GlobalSimulationParametersData *h_params );

    Particles particles; // CPU and GPU stack

    void copy_gpu2cpu(Particles &part);
    void print_stack(Particles part, ui32 n);


private:
    bool m_check_mandatory();

    void m_cpu_malloc_stack();
    void m_gpu_malloc_stack();
    void m_cpu_init_stack_seed( ui32 seed );
    void m_copy_seed_cpu2gpu();

    GlobalSimulationParametersData *mh_params;


};

#endif
