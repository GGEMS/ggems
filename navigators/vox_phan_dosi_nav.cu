// GGEMS Copyright (C) 2015

/*!
 * \file vox_phan_img_nav.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */

#ifndef VOX_PHAN_DOSI_NAV_CU
#define VOX_PHAN_DOSI_NAV_CU

#include "vox_phan_dosi_nav.cuh"

////:: GPU Codes

// Move particles to the voxelized volume
__host__ __device__ void VPDN::track_to_in(ParticlesData &particles, f32 xmin, f32 xmax,
                                              f32 ymin, f32 ymax, f32 zmin, f32 zmax,
                                              ui32 id) {
// std::cout<<__LINE__<< "   " <<particles.endsimu[id] <<  " F " << PARTICLE_FREEZE  << std::endl;
// std::cout<<particles<<std::endl;
    // Read position
    f64xyz pos;
    pos.x = particles.px[id];
    pos.y = particles.py[id];
    pos.z = particles.pz[id];

    // Read direction
    f64xyz dir;
    dir.x = particles.dx[id];
    dir.y = particles.dy[id];
    dir.z = particles.dz[id];

    f32 dist = hit_ray_AABB(pos, dir, xmin, xmax, ymin, ymax, zmin, zmax);

    // the particle not hitting the voxelized volume
    if (dist == FLT_MAX) {                            // TODO: Don't know why F32_MAX doesn't work...
        particles.endsimu[id] = PARTICLE_FREEZE;
        return;
    } else {
        // Check if the path of the particle cross the volume sufficiently
        f32 cross = dist_overlap_ray_AABB(pos, dir, xmin, xmax, ymin, ymax, zmin, zmax);
        if (cross < EPSILON3) {
            particles.endsimu[id] = PARTICLE_FREEZE;
            return;
        }
        // move the particle slightly inside the volume
        pos = fxyz_add(pos, fxyz_scale(pos, dist+EPSILON6));

        // TODO update tof
        // ...
    }

    // set photons
    particles.px[id] = pos.x;
    particles.py[id] = pos.y;
    particles.pz[id] = pos.z;

}

__host__ __device__ void VPDN::track_electron_to_out(ParticlesData &particles,
                                               VoxVolumeData vol,
                                               MaterialsTable materials,
                                               PhotonCrossSectionTable photon_CS_table,
                                               ElectronsCrossSectionTable electron_CS_table,
                                               GlobalSimulationParametersData parameters,
                                               DoseData dosi,
                                               f32 &randomnumbereIoni, 
                                               f32 &randomnumbereBrem, 
                                               f32 freeLength,
                                               ui32 part_id) {

                                               
        // parameters values needed to be stored in many steps
    f32 alongStepLength=0.; // Distance from the last physics interaction.
    bool lastStepisaPhysicEffect = TRUE; // To store last random number
    bool bool_loop = true; // If it is not the last step in the same voxel
    bool secondaryParticleCreated = FALSE; //If a secondary particle is created

    alongStepLength=freeLength;

    if(freeLength>0.0) lastStepisaPhysicEffect = FALSE; // Changement de voxel sans effet physique
    
    // Parameters
    int dummystep =0;
    f32 trueStepLength = 1e9f;
    f32 totalLength = 0.;
    f32 par1, par2;
    
        do
        {
            ++ dummystep;
            // Values initialisation
            f32 lengthtoVertex; // Value to store the distance from the last physics interaction.
            ui8 next_discrete_process ;
            i32 table_index; // indice de lecture de table de sections efficaces
            f32 next_interaction_distance = 1e9f;
            f32 dedxeIoni = 0;
            f32 dedxeBrem = 0;
            f32 erange = 0;
            f32 lambda = 0;
            bool significant_loss;

            f32 edep;  // Not used ?? - JB

            f32 trueGeomLength;
            
            if(lastStepisaPhysicEffect == TRUE)   // Get Random number stored until a physic interaction
            {
                randomnumbereBrem = -logf(JKISS32(particles, part_id));
                randomnumbereIoni = -logf(JKISS32(particles, part_id));
                alongStepLength = 0.f;
                lastStepisaPhysicEffect = FALSE;
                /*if(part_id==DEBUGID) PRINT_PARTICLE_STATE("");*/
            }
            
            // Read position
            f32xyz pos;
            pos.x = particles.px[part_id];
            pos.y = particles.py[part_id];
            pos.z = particles.pz[part_id];

            // Read direction
            f32xyz dir;
            dir.x = particles.dx[part_id];
            dir.y = particles.dy[part_id];
            dir.z = particles.dz[part_id];
            
            f32 energy = particles.E[part_id];
                
                                                    

            // Defined index phantom
            f32xyz ivoxsize;
            ivoxsize.x = 1.0 / vol.spacing_x;
            ivoxsize.y = 1.0 / vol.spacing_y;
            ivoxsize.z = 1.0 / vol.spacing_z;
            ui16xyzw index_phantom;
            index_phantom.x = ui16( (pos.x+vol.off_x) * ivoxsize.x );
            index_phantom.y = ui16( (pos.y+vol.off_y) * ivoxsize.y );
            index_phantom.z = ui16( (pos.z+vol.off_z) * ivoxsize.z );
            index_phantom.w = index_phantom.z*vol.nb_vox_x*vol.nb_vox_y
                            + index_phantom.y*vol.nb_vox_x
                            + index_phantom.x; // linear index
                            
            ui16 mat_id = vol.values[index_phantom.w];
            
            
            e_read_CS_table(part_id,mat_id, energy, electron_CS_table,next_discrete_process,table_index, next_interaction_distance,dedxeIoni,dedxeBrem,erange, lambda, randomnumbereBrem, randomnumbereIoni,parameters);
        
    
            lengthtoVertex = VertexLength(next_interaction_distance,alongStepLength);

        //Get cut step
            f32 cutstep = StepFunction(erange);


            if(lengthtoVertex>cutstep)
                {
                significant_loss=true;
                trueStepLength=cutstep;
                }
            else
                {
                significant_loss=false;
                trueStepLength=lengthtoVertex;
                }
        
        
            //// Get the next distance boundary volume /////////////////////////////////

            f32 vox_xmin = index_phantom.x*vol.spacing_x;
            f32 vox_ymin = index_phantom.y*vol.spacing_y;
            f32 vox_zmin = index_phantom.z*vol.spacing_z;
            f32 vox_xmax = vox_xmin + vol.spacing_x;
            f32 vox_ymax = vox_ymin + vol.spacing_y;
            f32 vox_zmax = vox_zmin + vol.spacing_z;
            
            f32 boundary_distance = hit_ray_AABB(pos, dir, vox_xmin, vox_xmax,
                                         vox_ymin, vox_ymax, vox_zmin, vox_zmax);
    
        
            if(boundary_distance<trueStepLength)
            {
                if(parameters.physics_list[ELECTRON_MSC] == ENABLED)
                {
                    trueGeomLength=gTransformToGeom(trueStepLength,erange,lambda,energy,&par1,&par2,electron_CS_table,mat_id);
                    if(trueGeomLength>boundary_distance)
                    {
                        bool_loop=false;
                    }
                }
                else
                {
                    bool_loop=false;
                }

            }



        if(bool_loop==true)
            {

            edep = eLoss(trueStepLength, particles.E[part_id], dedxeIoni, dedxeBrem, erange, electron_CS_table, mat_id, materials, particles, parameters, part_id);


            if(significant_loss == true)
                {
                    GlobalMscScattering( trueStepLength, cutstep, erange, energy, lambda,   dedxeIoni,  dedxeBrem,  electron_CS_table,  mat_id, particles,  part_id, par1, par2, materials, dosi, index_phantom,vol,parameters);
                
                /// TODO WARNING ACTIVER DOSIMETRIE
//                 dose_record(dosi, edep, particles.px[id],particles.py[id],particles.pz[id],id);
                    alongStepLength+=trueStepLength;
                    totalLength+=trueStepLength;
                    lastStepisaPhysicEffect=FALSE;
//                  Troncature(particles, id);

                }
            else     
                {
                    GlobalMscScattering( trueStepLength, lengthtoVertex, erange, energy, lambda,   dedxeIoni,  dedxeBrem,   electron_CS_table,  mat_id, particles,  part_id, par1, par2, materials, dosi, index_phantom,vol,parameters);
                    
                /// TODO WARNING ACTIVER DOSIMETRIE
//                 dose_record(dosi, edep, particles.px[id],particles.py[id],particles.pz[id],id);

                if(next_discrete_process == ELECTRON_IONISATION)
                    {
//                     
//                     eSampleSecondarieElectron( cutEnergyElectron, particles,  id, cache_secondaries, dosi);
                    lastStepisaPhysicEffect=TRUE;
                    secondaryParticleCreated = TRUE;
                    }
                else if (next_discrete_process == ELECTRON_BREMSSTRAHLUNG)
                    {
//                     
//                     eSampleSecondarieGamma( cutEnergyGamma, particles, id, materials, mat );
                    lastStepisaPhysicEffect=TRUE;
// 
                    }

                alongStepLength=0;
                freeLength=0.;


                totalLength+=trueStepLength;
//                 Troncature(particles, id);
                } // significant_loss == false

            } // bool_loop == true


//         break;
        dummystep ++ ;

    }while((particles.E[part_id]>EKINELIMIT)&&(bool_loop==true)&&(dummystep<10));



    if((particles.E[part_id]>EKINELIMIT)/*&&(secondaryParticleCreated == FALSE)*/)  //>1eV
    {

        unsigned char next_discrete_process ;
        int table_index; // indice de lecture de table de sections efficaces
        f32 next_interaction_distance = 1e9f;
        f32 dedxeIoni = 0;
        f32 dedxeBrem = 0;
        f32 erange = 0;
        f32 lambda = 0;
//         bool significant_loss;
//         f32 edep;
//         f32 trueGeomLength;
//         f32 safety;

        // Read position
        f32xyz pos; // mm
        pos.x = particles.px[part_id];
        pos.y = particles.py[part_id];
        pos.z = particles.pz[part_id];

        // Read direction
        f32xyz dir;
        dir.x = particles.dx[part_id];
        dir.y = particles.dy[part_id];
        dir.z = particles.dz[part_id];

        // Get energy
        f32 energy = particles.E[part_id];


        // Defined index phantom
        f32xyz ivoxsize;
        ivoxsize.x = 1.0 / vol.spacing_x;
        ivoxsize.y = 1.0 / vol.spacing_y;
        ivoxsize.z = 1.0 / vol.spacing_z;
        ui16xyzw index_phantom;
        index_phantom.x = ui16( (pos.x+vol.off_x) * ivoxsize.x );
        index_phantom.y = ui16( (pos.y+vol.off_y) * ivoxsize.y );
        index_phantom.z = ui16( (pos.z+vol.off_z) * ivoxsize.z );
        index_phantom.w = index_phantom.z*vol.nb_vox_x*vol.nb_vox_y
                        + index_phantom.y*vol.nb_vox_x
                        + index_phantom.x; // linear index                          
                          
                          
        //Get mat index
//         int mat = (int)(vol.data[index_phantom.w]);
        ui16 mat_id = vol.values[index_phantom.w];
        
            //// Get the next distance boundary volume /////////////////////////////////

        f32 vox_xmin = index_phantom.x*vol.spacing_x;
        f32 vox_ymin = index_phantom.y*vol.spacing_y;
        f32 vox_zmin = index_phantom.z*vol.spacing_z;
        f32 vox_xmax = vox_xmin + vol.spacing_x;
        f32 vox_ymax = vox_ymin + vol.spacing_y;
        f32 vox_zmax = vox_zmin + vol.spacing_z;
        
        f32 fragment = hit_ray_AABB(pos, dir, vox_xmin, vox_xmax,
                                        vox_ymin, vox_ymax, vox_zmin, vox_zmax);
        
        
        // Get distance to edge of voxel
//         f32 fragment = get_boundary_voxel_by_raycasting(index_phantom, pos, direction, vol.voxel_size, part_id);
        fragment+=1.E-3*mm;

        // Read Cross section table to get dedx, erange, lambda
        e_read_CS_table(part_id,mat_id, energy, electron_CS_table,next_discrete_process,table_index, next_interaction_distance,dedxeIoni,dedxeBrem,erange, lambda, randomnumbereBrem, randomnumbereIoni,parameters);


        f32 cutstep = StepFunction(erange);

        trueStepLength = GlobalMscScattering( fragment, cutstep, erange, energy, lambda,   dedxeIoni,  dedxeBrem,  electron_CS_table,  mat_id, particles,  part_id, par1, par2, materials, dosi, index_phantom,vol,parameters);


//         Troncature(particles, id);

        freeLength=alongStepLength+trueStepLength;

        totalLength+=trueStepLength;



        }
    else
        {
        particles.endsimu[part_id]=PARTICLE_DEAD;
        }        



}


__host__ __device__ void VPDN::track_photon_to_out(ParticlesData &particles,
                                               VoxVolumeData vol,
                                               MaterialsTable materials,
                                               PhotonCrossSectionTable photon_CS_table,
                                               ElectronsCrossSectionTable electron_CS_table,
                                               GlobalSimulationParametersData parameters,
                                               DoseData dosi,
                                               ui32 part_id) {

    // Read position
    f32xyz pos;
    pos.x = particles.px[part_id];
    pos.y = particles.py[part_id];
    pos.z = particles.pz[part_id];

    // Read direction
    f32xyz dir;
    dir.x = particles.dx[part_id];
    dir.y = particles.dy[part_id];
    dir.z = particles.dz[part_id];

    // Defined index phantom
    f32xyz ivoxsize;
    ivoxsize.x = 1.0 / vol.spacing_x;
    ivoxsize.y = 1.0 / vol.spacing_y;
    ivoxsize.z = 1.0 / vol.spacing_z;
    ui16xyzw index_phantom;
    index_phantom.x = ui16( (pos.x+vol.off_x) * ivoxsize.x );
    index_phantom.y = ui16( (pos.y+vol.off_y) * ivoxsize.y );
    index_phantom.z = ui16( (pos.z+vol.off_z) * ivoxsize.z );
    index_phantom.w = index_phantom.z*vol.nb_vox_x*vol.nb_vox_y
                      + index_phantom.y*vol.nb_vox_x
                      + index_phantom.x; // linear index

    // Get the material that compose this volume
    ui16 mat_id = vol.values[index_phantom.w];

    //// Find next discrete interaction ///////////////////////////////////////

    photon_get_next_interaction(particles, parameters, photon_CS_table, mat_id, part_id);

    f32 next_interaction_distance = particles.next_interaction_distance[part_id];
    ui8 next_discrete_process = particles.next_discrete_process[part_id];

    //// Get the next distance boundary volume /////////////////////////////////

    f32 vox_xmin = index_phantom.x*vol.spacing_x;
    f32 vox_ymin = index_phantom.y*vol.spacing_y;
    f32 vox_zmin = index_phantom.z*vol.spacing_z;
    f32 vox_xmax = vox_xmin + vol.spacing_x;
    f32 vox_ymax = vox_ymin + vol.spacing_y;
    f32 vox_zmax = vox_zmin + vol.spacing_z;

    f32 boundary_distance = hit_ray_AABB(pos, dir, vox_xmin, vox_xmax,
                                         vox_ymin, vox_ymax, vox_zmin, vox_zmax);

    if (boundary_distance <= next_interaction_distance) {
        next_interaction_distance = boundary_distance + EPSILON3; // Overshoot
        next_discrete_process = GEOMETRY_BOUNDARY;
    }

    //// Move particle //////////////////////////////////////////////////////

    pos = fxyz_add(pos, fxyz_scale(dir, next_interaction_distance));

    // Update TOF - TODO
    //particles.tof[part_id] += c_light * next_interaction_distance;

    particles.px[part_id] = pos.x;
    particles.py[part_id] = pos.y;
    particles.pz[part_id] = pos.z;

    // Stop simulation if out of the phantom
    if (!test_point_AABB(pos, vol.xmin, vol.xmax, vol.ymin, vol.ymax, vol.zmin, vol.zmax)) {
        particles.endsimu[part_id] = PARTICLE_FREEZE;
        return;
    }

    //// Apply discrete process //////////////////////////////////////////////////

    if (next_discrete_process != GEOMETRY_BOUNDARY) {
        // Resolve discrete process
        SecParticle electron = photon_resolve_discrete_process(particles, parameters, photon_CS_table,
                                                               materials, mat_id, part_id);

        //// Here e- are not tracked, and lost energy not drop
        
        /// TODO ADD ELECTRON NAV
        ggems_atomic_add(dosi.edep, index_phantom.w, electron.E);
        ggems_atomic_add(dosi.edep_squared, index_phantom.w, electron.E*electron.E);
        
        
    }

    //// Energy cut
    if (particles.E[part_id] <= materials.electron_energy_cut[mat_id]) {
        particles.endsimu[part_id] = PARTICLE_DEAD;
        return;
    }

}

// Device Kernel that move particles to the voxelized volume boundary
__global__ void VPDN::kernel_device_track_to_in(ParticlesData particles, f32 xmin, f32 xmax,
                                            f32 ymin, f32 ymax, f32 zmin, f32 zmax) {

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles.size) return;

    VPDN::track_to_in(particles, xmin, xmax, ymin, ymax, zmin, zmax, id);

}

// Host Kernel that move particles to the voxelized volume boundary
void VPDN::kernel_host_track_to_in(ParticlesData particles, f32 xmin, f32 xmax,
                             f32 ymin, f32 ymax, f32 zmin, f32 zmax, ui32 part_id) {

    VPDN::track_to_in(particles, xmin, xmax, ymin, ymax, zmin, zmax, part_id);

}

// Device kernel that track particles within the voxelized volume until boundary
__global__ void VPDN::kernel_device_track_to_out(ParticlesData particles,
                                           VoxVolumeData vol,
                                           MaterialsTable materials,
                                           PhotonCrossSectionTable photon_CS_table,
                                           ElectronsCrossSectionTable electron_CS_table,
                                           GlobalSimulationParametersData parameters,
                                           DoseData dosi) {

    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= particles.size) return;

    // For multivoxels navigation
    f32 randomnumbereIoni= -std::log(JKISS32(particles, id)); // -log(RN)
    f32 randomnumbereBrem= -std::log(JKISS32(particles, id)); // -log(RN)
    f32 freeLength = 0.0*mm;
    
    // Stepping loop
    while (particles.endsimu[id] != PARTICLE_DEAD && particles.endsimu[id] != PARTICLE_FREEZE) {
        
        if(particles.pname[id] == PHOTON)
        {
            VPDN::track_photon_to_out(particles, vol, materials, photon_CS_table,electron_CS_table, parameters, dosi, id);
        }
        else if(particles.pname[id] == ELECTRON)
        {
            VPDN::track_electron_to_out(particles, vol, materials, photon_CS_table,electron_CS_table, parameters, dosi,randomnumbereIoni, randomnumbereBrem, freeLength, id);
        }    
    
    }

}

// Host kernel that track particles within the voxelized volume until boundary
void VPDN::kernel_host_track_to_out(ParticlesData particles,
                              VoxVolumeData vol,
                              MaterialsTable materials,
                              PhotonCrossSectionTable photon_CS_table,
                              ElectronsCrossSectionTable electron_CS_table,
                              GlobalSimulationParametersData parameters,
                              DoseData dosi,
                              ui32 id) {

    // For multivoxels navigation
    f32 randomnumbereIoni= -std::log(JKISS32(particles, id)); // -log(RN)
    f32 randomnumbereBrem= -std::log(JKISS32(particles, id)); // -log(RN)
    f32 freeLength = 0.0*mm;

// Stepping loop
    while (particles.endsimu[id] != PARTICLE_DEAD && particles.endsimu[id] != PARTICLE_FREEZE) {

        if(particles.pname[id] == PHOTON)
        {
            VPDN::track_photon_to_out(particles, vol, materials, photon_CS_table,electron_CS_table, parameters, dosi, id);
        }
        else if(particles.pname[id] == ELECTRON)
        {
            VPDN::track_electron_to_out(particles, vol, materials, photon_CS_table,electron_CS_table, parameters, dosi, randomnumbereIoni, randomnumbereBrem, freeLength, id);
        }
//         break;
        
    }
}

////:: Privates

bool VoxPhanDosiNav::m_check_mandatory() {

    if (m_phantom.data_h.nb_vox_x == 0 || m_phantom.data_h.nb_vox_y == 0 || m_phantom.data_h.nb_vox_z == 0 ||
        m_phantom.data_h.spacing_x == 0 || m_phantom.data_h.spacing_y == 0 || m_phantom.data_h.spacing_z == 0 ||
        m_phantom.list_of_materials.size() == 0) {
        return false;
    } else {
        return true;
    }

}

////:: Main functions

void VoxPhanDosiNav::track_to_in(Particles particles) {
// std::cout<<__LINE__<<std::endl;
    if (m_params.data_h.device_target == CPU_DEVICE) {
        ui32 id=0; while (id<particles.size) {
            VPDN::kernel_host_track_to_in(particles.data_h, m_phantom.data_h.xmin, m_phantom.data_h.xmax,
                                                   m_phantom.data_h.ymin, m_phantom.data_h.ymax,
                                                   m_phantom.data_h.zmin, m_phantom.data_h.zmax,
                                                   id);
            ++id;
        }
    } else if (m_params.data_h.device_target == GPU_DEVICE) {

        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = (particles.size + m_params.data_h.gpu_block_size - 1) / m_params.data_h.gpu_block_size;

        VPDN::kernel_device_track_to_in<<<grid, threads>>>(particles.data_d, m_phantom.data_d.xmin, m_phantom.data_d.xmax,
                                                     m_phantom.data_d.ymin, m_phantom.data_d.ymax,
                                                     m_phantom.data_d.zmin, m_phantom.data_d.zmax);
        cuda_error_check("Error ", " Kernel_VoxPhanDosi (track to in)");

    }


}

void VoxPhanDosiNav::track_to_out(Particles particles) {
    
    if (m_params.data_h.device_target == CPU_DEVICE) {
                        
        ui32 id=0; while (id<particles.size) {
           
                  VPDN::kernel_host_track_to_out(particles.data_h, m_phantom.data_h,
                                                 m_materials.data_h, m_cross_sections.photon_CS.data_h, m_cross_sections.electron_CS.data_h,
                                                 m_params.data_h, m_dose_calculator.dose.data_h, id);
            ++id;
        }
    } else if (m_params.data_h.device_target == GPU_DEVICE) {

        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = (particles.size + m_params.data_h.gpu_block_size - 1) / m_params.data_h.gpu_block_size;

        VPDN::kernel_device_track_to_out<<<grid, threads>>>(particles.data_d, m_phantom.data_d, m_materials.data_d,
                                                      m_cross_sections.photon_CS.data_d,
                                                      m_cross_sections.electron_CS.data_d,
                                                      m_params.data_d, m_dose_calculator.dose.data_d);
        cuda_error_check("Error ", " Kernel_VoxPhanDosi (track to out)");

    }

}

void VoxPhanDosiNav::load_phantom_from_mhd(std::string filename, std::string range_mat_name)
{
    m_phantom.load_from_mhd(filename, range_mat_name);
}

void VoxPhanDosiNav::initialize(GlobalSimulationParameters params) {
    // Check params
    if (!m_check_mandatory()) {
        print_error("VoxPhanDosi: missing parameters.");
        exit_simulation();
    }

    // Params
    m_params = params;

    // Phantom
    m_phantom.set_name("VoxPhanDosiNav");
    m_phantom.initialize(params);

    // Materials table
    m_materials.load_elements_database();
    m_materials.load_materials_database();
    m_materials.initialize(m_phantom.list_of_materials, params);

    // Cross Sections
    m_cross_sections.initialize(m_materials, params);


    // Init dose map    
    m_dose_calculator.set_size_in_voxel(m_phantom.data_h.nb_vox_x,
                                        m_phantom.data_h.nb_vox_y,
                                        m_phantom.data_h.nb_vox_z);
    m_dose_calculator.set_voxel_size(m_phantom.data_h.spacing_x,
                                     m_phantom.data_h.spacing_y,
                                     m_phantom.data_h.spacing_z);
    m_dose_calculator.set_offset(m_phantom.data_h.off_x,
                                 m_phantom.data_h.off_y,
                                 m_phantom.data_h.off_z);
    m_dose_calculator.initialize(m_params); // CPU&GPU

}

#endif
