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
struct ParticlesData {
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

}; //

// Struct that handles particles
struct Particles {
    ParticlesData data_h;
    ParticlesData data_d;

    ui32 size;
};

// Helper to handle secondaries particles
struct SecParticle {
    f32xyz dir;
    f32 E;
    ui8 pname;
    ui8 endsimu;
};

/*
// Helper to handle history of particles
struct OneParticleStep {
    f32xyz pos;
    f32xyz dir;
    f32 E;
};
*/

/*
// History class
class HistoryBuilder {
    public:
        HistoryBuilder();

        void cpu_new_particle_track(ui32 a_pname);
        void cpu_record_a_step(ParticleStack particles, ui32 id_part);

        std::vector<ui8> pname;
        std::vector<ui32> nb_steps;
        std::vector< std::vector<OneParticleStep> > history_data;

        ui8 record_flag;      // Record or not
        ui32 max_nb_particles;  // Max nb of particles keep in the history
        ui32 cur_iter;          // Current number of iterations
        ui32 stack_size;        // Size fo the particle stack

    private:
        ui32 current_particle_id;
        ui8 type_of_particles;
};
*/


// Particles class
class ParticleManager {
    public:
        ParticleManager();

        void initialize(GlobalSimulationParameters params);
        //void cpu_print_stack(ui32 nlim);

        Particles particles; // CPU and GPU stack

    private:
        bool m_check_mandatory();

        void m_cpu_malloc_stack();
        //void m_cpu_free_stack();
        void m_gpu_malloc_stack();
        void m_cpu_init_stack_seed(ui32 seed);
        void m_copy_seed_cpu2gpu();

};

#endif
