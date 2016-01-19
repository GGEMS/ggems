// GGEMS Copyright (C) 2015

/*!
 * \file photon_navigator.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 20 novembre 2015
 *
 *
 *
 */

#ifndef PHOTON_NAVIGATOR_CU
#define PHOTON_NAVIGATOR_CU

#include "photon_navigator.cuh"

__host__ __device__ void photon_get_next_interaction ( ParticlesData &particles,
        GlobalSimulationParametersData parameters,
        PhotonCrossSectionTable photon_CS_table,
        ui16 mat_id, ui32 part_id )
{
//     printf("    ---> Distances : ");
    f32 next_interaction_distance = F32_MAX;
    ui8 next_discrete_process = 0;
    f32 interaction_distance;
    f32 cross_section;

    // Search the energy index to read CS
    f32 energy = particles.E[part_id];
    ui32 E_index = binary_search ( energy, photon_CS_table.E_bins,
                                   photon_CS_table.nb_bins );

    // If photoelectric
    if ( parameters.physics_list[PHOTON_PHOTOELECTRIC] )
    {
        cross_section = get_CS_from_table ( photon_CS_table.E_bins, photon_CS_table.Photoelectric_Std_CS,
                                            energy, E_index, mat_id, photon_CS_table.nb_bins );
        f32 alea = JKISS32 ( particles, part_id );
        interaction_distance = -log ( alea ) / cross_section;
//         printf("Energy : %g, PE %g %g %g ",energy,interaction_distance, alea, cross_section);
        if ( interaction_distance < next_interaction_distance )
        {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_PHOTOELECTRIC;
        }
    }

    // If Compton
    if ( parameters.physics_list[PHOTON_COMPTON] )
    {
        cross_section = get_CS_from_table ( photon_CS_table.E_bins, photon_CS_table.Compton_Std_CS,
                                            energy, E_index, mat_id, photon_CS_table.nb_bins );
        interaction_distance = -log ( JKISS32 ( particles, part_id ) ) / cross_section;
//         printf("CPT %g ",interaction_distance);
        if ( interaction_distance < next_interaction_distance )
        {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_COMPTON;
        }
    }

    // If Rayleigh
    if ( parameters.physics_list[PHOTON_RAYLEIGH] )
    {
        cross_section = get_CS_from_table ( photon_CS_table.E_bins, photon_CS_table.Rayleigh_Lv_CS,
                                            energy, E_index, mat_id, photon_CS_table.nb_bins );
        interaction_distance = -log ( JKISS32 ( particles, part_id ) ) / cross_section;
//         printf("R %g ",interaction_distance);
        if ( interaction_distance < next_interaction_distance )
        {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_RAYLEIGH;
        }
    }
//     printf("\n");
    // Store results
    particles.next_interaction_distance[part_id] = next_interaction_distance;
    particles.next_discrete_process[part_id] = next_discrete_process;
    particles.E_index[part_id] = E_index;

}



__host__ __device__ SecParticle photon_resolve_discrete_process ( ParticlesData &particles,
        GlobalSimulationParametersData parameters,
        PhotonCrossSectionTable photon_CS_table,
        MaterialsTable materials,
        ui16 mat_id, ui32 part_id )
{

    SecParticle electron;
    electron.endsimu = PARTICLE_DEAD;
    electron.dir.x = 0.;
    electron.dir.y = 0.;
    electron.dir.z = 1.;
    electron.E = 0.;
    ui8 next_discrete_process = particles.next_discrete_process[part_id];

    if ( next_discrete_process == PHOTON_COMPTON )
    {
        electron = Compton_SampleSecondaries_standard ( particles, materials.electron_energy_cut[mat_id],
                   part_id, parameters );
    }

    if ( next_discrete_process == PHOTON_PHOTOELECTRIC )
    {
        electron = Photoelec_SampleSecondaries_standard ( particles, materials, photon_CS_table,
                   particles.E_index[part_id], materials.electron_energy_cut[mat_id],
                   mat_id, part_id, parameters );
    }

    if ( next_discrete_process == PHOTON_RAYLEIGH )
    {
        Rayleigh_SampleSecondaries_Livermore ( particles, materials, photon_CS_table, particles.E_index[part_id], mat_id, part_id );

    }

    return electron;

}





















#endif
