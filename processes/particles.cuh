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
#include <iomanip>

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

    ui8 nb_of_secondaries;
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

    friend std::ostream& operator<< ( std::ostream& os, const ParticlesData v )
    {
        os  << std::fixed << std::setprecision ( 2 );
        os  << "Particle state : " << std::endl;

        os  << "\t"  <<  "+"  << std::setfill ( '-' ) << std::setw ( 30 ) << "+" << std::endl;
        os  << std::setfill ( ' ' );

        os  << "\t"   << "|"
            << std::left  << std::setw ( 9 ) << ""
            << std::right << std::setw ( 5 ) << "X"
            << std::right << std::setw ( 7 ) << "Y"
            << std::right << std::setw ( 7 ) << "Z"
            << std::setw ( 2 ) << "|" << std::endl;

        os  << "\t"   << "|"
            << std::left  << std::setw ( 9 ) << "Position"
            << std::right << std::setw ( 5 ) << *v.px
            << std::right << std::setw ( 7 ) << *v.py
            << std::right << std::setw ( 7 ) << *v.pz
            << std::setw ( 2 ) << "|" << std::endl;

        os  << "\t"   << "|"
            << std::left  << std::setw ( 9 ) << "Direction"
            << std::right << std::setw ( 5 ) << v.dx
            << std::right << std::setw ( 7 ) << v.dy
            << std::right << std::setw ( 7 ) << v.dz
            << std::setw ( 2 ) << "|" << std::endl;

        os << "\t"   << "|"
           << std::left  << std::setw ( 9 ) << "Energy"
           << std::right << std::setw ( 5 ) << *v.E
           << std::right << std::setw ( 7 ) << *v.E
           << std::right << std::setw ( 7 ) << *v.E
           << std::setw ( 2 ) << "|" << std::endl;

        os << "\t"   <<  "+"  << std::setfill ( '-' ) << std::setw ( 30 ) << "+" << std::endl;
        return os;

    }
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
class ParticleManager
{
public:
    ParticleManager();

    void initialize ( GlobalSimulationParameters params );    

    Particles particles; // CPU and GPU stack

    void copy_gpu2cpu(Particles part);
    void print_stack(Particles part);


private:
    bool m_check_mandatory();

    void m_cpu_malloc_stack();
    //void m_cpu_free_stack();
    void m_gpu_malloc_stack();
    void m_cpu_init_stack_seed ( ui32 seed );
    void m_copy_seed_cpu2gpu();


};

#endif
