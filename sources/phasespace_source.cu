// GGEMS Copyright (C) 2017

/*!
 * \file phasespace_source.cu
 * \brief phasespace source class
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.2
 * \date 9 mars 2016
 *
 * v0.2: JB - Change all structs and remove CPU exec
 *
 * phasespace source class
 * 
 */

#ifndef PHASESPACE_SOURCE_CU
#define PHASESPACE_SOURCE_CU

#include "phasespace_source.cuh"

///////// GPU code ////////////////////////////////////////////////////

// Internal function that create a new particle to the buffer at the slot id
__host__ __device__ void PHSPSRC::phsp_source ( ParticlesData *particles_data,
                                                const PhaseSpaceData *phasespace,
                                                PhSpTransform transform, ui32 id )



{
    // Get a random sources (i.e. virtual sources that use the same phasespace)
    ui32 source_id = 0;
    if ( transform.nb_sources > 1 )
    {
        source_id = binary_search_left( prng_uniform( particles_data, id ), transform.cdf, transform.nb_sources );
    }

#ifdef DEBUG
    assert( source_id < transform.nb_sources );
    if ( source_id >= transform.nb_sources )
    {
        printf("   source id %i    nb sources %i\n", source_id, transform.nb_sources);
    }
#endif

    // Get a random particle from the phasespace
    ui32 phsp_id = (ui32) ( f32(phasespace->tot_particles) * prng_uniform( particles_data, id ) );

#ifdef DEBUG
    assert( phsp_id < phasespace->tot_particles );
#endif

    // Then set the mandatory field to create a new particle
    particles_data->E[id] = phasespace->energy[ phsp_id ];     // Energy in MeV

    // Apply transformation
    f32xyz pos = make_f32xyz( phasespace->pos_x[ phsp_id ], phasespace->pos_y[ phsp_id ], phasespace->pos_z[ phsp_id ] );
    f32xyz dir = make_f32xyz( phasespace->dir_x[ phsp_id ], phasespace->dir_y[ phsp_id ], phasespace->dir_z[ phsp_id ] );
    f32xyz t = make_f32xyz( transform.tx[ source_id ], transform.ty[ source_id ], transform.tz[ source_id ] );

    //printf("ID %i rot %f %f %f\n", id, transform.rx, transform.ry, transform.rz);

    pos = fxyz_add( pos, t );                                        // translate
    pos = fxyz_rotate_z_axis( pos, transform.rz[ source_id ] );      // Rotate: yaw, pitch, and roll convetion (RzRyRx)
    pos = fxyz_rotate_y_axis( pos, transform.ry[ source_id ] );      //         for Euler convention RzRyRz
    pos = fxyz_rotate_x_axis( pos, transform.rx[ source_id ] );      //         Here is a right-hand rule
    //                TODO add scaling - JB
    dir = fxyz_rotate_z_axis( dir, transform.rz[ source_id ] );
    dir = fxyz_rotate_y_axis( dir, transform.ry[ source_id ] );
    dir = fxyz_rotate_x_axis( dir, transform.rx[ source_id ] );
    dir = fxyz_unit( dir );

    particles_data->px[id] = pos.x;     // Position in mm
    particles_data->py[id] = pos.y;     //
    particles_data->pz[id] = pos.z;     //

    particles_data->dx[id] = dir.x;     // Direction (unit vector)
    particles_data->dy[id] = dir.y;     //
    particles_data->dz[id] = dir.z;     //

    particles_data->tof[id] = 0.0f;                             // Time of flight
    particles_data->status[id] = PARTICLE_ALIVE;               // Status of the particle

    particles_data->level[id] = PRIMARY;                        // It is a primary particle
    particles_data->pname[id] = phasespace->ptype[ phsp_id ];    // a photon or an electron

    particles_data->geometry_id[id] = 0;                        // Some internal variables
    particles_data->next_discrete_process[id] = NO_PROCESS;     //
    particles_data->next_interaction_distance[id] = 0.0;        //
    particles_data->scatter_order[ id ] = 0;                    //


//    printf(" ID %i E %e pos %e %e %e dir %e %e %e ptype %i\n", id, particles_data->E[id],
//           particles_data->px[id], particles_data->py[id], particles_data->pz[id],
//           particles_data->dx[id], particles_data->dy[id], particles_data->dz[id],
//           particles_data->pname[id]);
}

__global__ void PHSPSRC::phsp_point_source (ParticlesData *particles_data,
                                            const PhaseSpaceData *phasespace,
                                            PhSpTransform transform )
{
    // Get thread id
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles_data->size ) return;

    // Get a new particle
    phsp_source( particles_data, phasespace, transform, id );
}

//////// Class //////////////////////////////////////////////////////////

// Constructor
PhaseSpaceSource::PhaseSpaceSource(): GGEMSSource()
{
    // Set the name of the source
    set_name( "PhaseSpaceSource" );   

    // Max number of particles used from the phase-space file
    m_nb_part_max = -1;  // -1 mean take all particles

    // Vars
    m_phasespace_file = "";
    m_transformation_file = "";

    mh_params = nullptr;

    mh_phasespace = nullptr;
    md_phasespace = nullptr;

    // Init the struc
    mh_transform = (PhSpTransform*) malloc( sizeof(PhSpTransform) );
    mh_transform->nb_sources = 0;
    mh_transform->rx = nullptr;
    mh_transform->ry = nullptr;
    mh_transform->rz = nullptr;

    mh_transform->tx = nullptr;
    mh_transform->ty = nullptr;
    mh_transform->tz = nullptr;

    mh_transform->sx = nullptr;
    mh_transform->sy = nullptr;
    mh_transform->sz = nullptr;

    mh_transform->cdf = nullptr;

}

// Destructor
PhaseSpaceSource::~PhaseSpaceSource() {
//    // ensure all memory is deallocated
//    cudaDeviceSynchronize();
//    cudaFree(m_spectrum->energies);
//    cudaFree(m_spectrum->cdf);
//    delete m_spectrum;

    // TODO delete m_iaea
}

//========== Setting ===============================================

// Setting position of the source
void PhaseSpaceSource::set_translation( f32 tx, f32 ty, f32 tz )
{
    m_tx = tx;
    m_ty = ty;
    m_tz = tz;
}

// Setting rotation of the source
void PhaseSpaceSource::set_rotation( f32 aroundx, f32 aroundy, f32 aroundz )
{
    m_rx = aroundx;
    m_ry = aroundy;
    m_rz = aroundz;
}

// Setting scaling of the source
void PhaseSpaceSource::set_scaling( f32 sx, f32 sy, f32 sz )
{
    m_sx = sx;
    m_sy = sy;
    m_sz = sz;
}

// Setting the maximum number of particles used from the phase-space file
void PhaseSpaceSource::set_max_number_of_particles( ui32 nb_part_max )
{
    m_nb_part_max = nb_part_max;
}

void PhaseSpaceSource::set_phasespace_file( std::string filename )
{
    m_phasespace_file = filename;
}

void PhaseSpaceSource::set_transformation_file( std::string filename )
{
    m_transformation_file = filename;
}

void PhaseSpaceSource::update_phasespace_file( std::string filename )
{
    m_phasespace_file = filename;

    // Load the phasespace file (CPU & GPU)
    if ( m_phasespace_file != "" )
    {
        m_load_phasespace_file();
    }
    else
    {
        GGcerr << "Phasespace filename is not valid!" << GGendl;
        exit_simulation();
    }

    // Check if everything was set properly
    if ( !m_check_mandatory() )
    {
        GGcerr << "Phasespace source was not set properly!" << GGendl;
        exit_simulation();
    }

    // Check if the physics is set properly considering the phasespace file
    bool there_is_photon = mh_params->physics_list[PHOTON_COMPTON] ||
                           mh_params->physics_list[PHOTON_PHOTOELECTRIC] ||
                           mh_params->physics_list[PHOTON_RAYLEIGH];

    bool there_is_electron = mh_params->physics_list[ELECTRON_IONISATION] ||
                             mh_params->physics_list[ELECTRON_BREMSSTRAHLUNG] ||
                             mh_params->physics_list[ELECTRON_MSC];

    if ( !there_is_electron && mh_phasespace->nb_electrons != 0 )
    {
        GGcerr << "Phasespace file contains " << mh_phasespace->nb_electrons
               << " electrons and there are no electron physics effects enabled!"
               << GGendl;
        exit_simulation();
    }

    if ( !there_is_photon && mh_phasespace->nb_photons != 0 )
    {
        GGcerr << "Phasespace file contains " << mh_phasespace->nb_photons
               << " photons and there are no photon physics effects enabled!"
               << GGendl;
        exit_simulation();
    }

    if ( mh_phasespace->nb_positrons != 0 )
    {
        GGcerr << "Phasespace file contains " << mh_phasespace->nb_positrons
               << " positrons and there are no positron physics effects enabled!"
               << GGendl;
        exit_simulation();
    }

    // Cut off the number of particles according the user parameter
    if ( m_nb_part_max != -1 )
    {
        if ( m_nb_part_max < mh_phasespace->tot_particles )
        {
            mh_phasespace->tot_particles = m_nb_part_max;
        }
    }

    // Update data on GPU
    m_free_phasespace_to_gpu();
    m_copy_phasespace_to_gpu();

    // Some verbose if required
    if ( mh_params->display_memory_usage )
    {
        ui32 mem = 29 * mh_phasespace->tot_particles;
        GGcout_mem("PhaseSpace source", mem);
    }

}

//========= Private ============================================

// Skip comment starting with "#"
void PhaseSpaceSource::m_skip_comment(std::istream & is) {
    i8 c;
    i8 line[1024];
    if (is.eof()) return;
    is >> c;
    while (is && (c=='#')) {
        is.getline(line, 1024);
        is >> c;
        if (is.eof()) return;
    }
    is.unget();
}

// Check if everything is ok to initialize this source
bool PhaseSpaceSource::m_check_mandatory()
{
    if ( mh_phasespace->tot_particles == 0 ) return false;
    else return true;
}

// Transform data allocation
void PhaseSpaceSource::m_transform_allocation( ui32 nb_sources )
{
    /*
    // Allocation
    HANDLE_ERROR( cudaMallocManaged( &(m_transform.tx), nb_sources*sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_transform.ty), nb_sources*sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_transform.tz), nb_sources*sizeof( f32 ) ) );

    HANDLE_ERROR( cudaMallocManaged( &(m_transform.rx), nb_sources*sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_transform.ry), nb_sources*sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_transform.rz), nb_sources*sizeof( f32 ) ) );

    HANDLE_ERROR( cudaMallocManaged( &(m_transform.sx), nb_sources*sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_transform.sy), nb_sources*sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_transform.sz), nb_sources*sizeof( f32 ) ) );

    HANDLE_ERROR( cudaMallocManaged( &(m_transform.cdf), nb_sources*sizeof( f32 ) ) );
    */

    mh_transform->nb_sources = nb_sources;

    mh_transform->tx = (f32*)malloc( nb_sources*sizeof( f32 ) );
    mh_transform->ty = (f32*)malloc( nb_sources*sizeof( f32 ) );
    mh_transform->tz = (f32*)malloc( nb_sources*sizeof( f32 ) );

    mh_transform->rx = (f32*)malloc( nb_sources*sizeof( f32 ) );
    mh_transform->ry = (f32*)malloc( nb_sources*sizeof( f32 ) );
    mh_transform->rz = (f32*)malloc( nb_sources*sizeof( f32 ) );

    mh_transform->sx = (f32*)malloc( nb_sources*sizeof( f32 ) );
    mh_transform->sy = (f32*)malloc( nb_sources*sizeof( f32 ) );
    mh_transform->sz = (f32*)malloc( nb_sources*sizeof( f32 ) );

    mh_transform->cdf = (f32*)malloc( nb_sources*sizeof( f32 ) );
}

// Load phasespace file
void PhaseSpaceSource::m_load_phasespace_file()
{
    PhaseSpaceIO *reader = new PhaseSpaceIO;
    mh_phasespace = reader->read_phasespace_file( m_phasespace_file );
    delete reader;
}

// Load transformation file
void PhaseSpaceSource::m_load_transformation_file()
{
    // Open the file
    std::ifstream input( m_transformation_file.c_str(), std::ios::in );
    if( !input )
    {
        GGcerr << "Error to open the file'" << m_transformation_file << "'!" << GGendl;
        exit_simulation();
    }

    // Get the number of sources
    std::string line;
    mh_transform->nb_sources = 0;

    while( input )
    {
        m_skip_comment( input );
        std::getline( input, line );

        if ( input ) ++mh_transform->nb_sources;
    }

    // Returning to beginning of the file to read it again
    input.clear();
    input.seekg( 0, std::ios::beg );

    // Allocating buffers to store data
    m_transform_allocation( mh_transform->nb_sources );

    // Store data from file
    size_t idx = 0;
    f32 sum_act = 0;
    while( input )
    {
        m_skip_comment( input );
        std::getline( input, line );

        if ( input )
        {
            // Format
            // tx ty tz rx ry rz sx sy sz activity(prob emission)
            std::istringstream iss( line );
            iss >> mh_transform->tx[ idx ] >> mh_transform->ty[ idx ] >> mh_transform->tz[ idx ]
                >> mh_transform->rx[ idx ] >> mh_transform->ry[ idx ] >> mh_transform->rz[ idx ]
                >> mh_transform->sx[ idx ] >> mh_transform->sy[ idx ] >> mh_transform->sz[ idx ]
                >> mh_transform->cdf[ idx ];
            sum_act += mh_transform->cdf[ idx ];

            // Units
            mh_transform->tx[ idx ] *= mm;
            mh_transform->ty[ idx ] *= mm;
            mh_transform->tz[ idx ] *= mm;

            mh_transform->x[ idx ] *= deg;
            mh_transform->ry[ idx ] *= deg;
            mh_transform->rz[ idx ] *= deg;

            ++idx;
        }
    }

    // Compute CDF and normalized in same time by security
    mh_transform->cdf[ 0 ] /= sum_act;
    for( ui32 i = 1; i < mh_transform->nb_sources; ++i )
    {
        mh_transform->cdf[ i ] = mh_transform->cdf[ i ] / sum_act + mh_transform->cdf[ i - 1 ];
    }

    // Watch dog
    mh_transform->cdf[ mh_transform->nb_sources - 1 ] = 1.0;

    // Close the file
    input.close();

}

void PhaseSpaceSource::m_copy_phasespace_to_gpu()
{
    ui32 n = mh_phasespace->tot_particles;

    /// First, struct allocation
    HANDLE_ERROR( cudaMalloc( (void**) &md_phasespace, sizeof( PhaseSpaceData ) ) );

    /// Device pointers allocation
    f32 *energy;
    HANDLE_ERROR( cudaMalloc((void**) &energy, n*sizeof(f32)) );
    f32 *pos_x;
    HANDLE_ERROR( cudaMalloc((void**) &pos_x, n*sizeof(f32)) );
    f32 *pos_y;
    HANDLE_ERROR( cudaMalloc((void**) &pos_y, n*sizeof(f32)) );
    f32 *pos_z;
    HANDLE_ERROR( cudaMalloc((void**) &pos_z, n*sizeof(f32)) );

    f32 *dir_x;
    HANDLE_ERROR( cudaMalloc((void**) &dir_x, n*sizeof(f32)) );
    f32 *dir_y;
    HANDLE_ERROR( cudaMalloc((void**) &dir_y, n*sizeof(f32)) );
    f32 *dir_z;
    HANDLE_ERROR( cudaMalloc((void**) &dir_z, n*sizeof(f32)) );

    ui8 *ptype;
    HANDLE_ERROR( cudaMalloc((void**) &ptype, n*sizeof(ui8)) );

    /// Copy host data to device
    HANDLE_ERROR( cudaMemcpy( energy, mh_phasespace->energy,
                              n*sizeof(f32), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( pos_x, mh_phasespace->pos_x,
                              n*sizeof(f32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( pos_y, mh_phasespace->pos_y,
                              n*sizeof(f32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( pos_z, mh_phasespace->pos_z,
                              n*sizeof(f32), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( dir_x, mh_phasespace->dir_x,
                              n*sizeof(f32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dir_y, mh_phasespace->dir_y,
                              n*sizeof(f32), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dir_z, mh_phasespace->dir_z,
                              n*sizeof(f32), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( ptype, mh_phasespace->energy,
                              n*sizeof(ui8), cudaMemcpyHostToDevice ) );

    /// Bind data to the struct
    HANDLE_ERROR( cudaMemcpy( &(md_phasespace->energy), &energy,
                              sizeof(md_phasespace->energy), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(md_phasespace->pos_x), &pos_x,
                              sizeof(md_phasespace->pos_x), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(md_phasespace->pos_y), &pos_y,
                              sizeof(md_phasespace->pos_y), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(md_phasespace->pos_z), &pos_z,
                              sizeof(md_phasespace->pos_z), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(md_phasespace->dir_x), &dir_x,
                              sizeof(md_phasespace->dir_x), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(md_phasespace->dir_y), &dir_y,
                              sizeof(md_phasespace->dir_y), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(md_phasespace->dir_z), &dir_z,
                              sizeof(md_phasespace->dir_z), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(md_phasespace->ptype), &ptype,
                              sizeof(md_phasespace->ptype), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(md_phasespace->tot_particles), &(mh_phasespace->tot_particles),
                              sizeof(md_phasespace->tot_particles), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(md_phasespace->nb_photons), &(mh_phasespace->nb_photons),
                              sizeof(md_phasespace->nb_photons), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(md_phasespace->nb_electrons), &(mh_phasespace->nb_electrons),
                              sizeof(md_phasespace->nb_electrons), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(md_phasespace->nb_positrons), &(mh_phasespace->nb_positrons),
                              sizeof(md_phasespace->nb_positrons), cudaMemcpyHostToDevice ) );

}

void PhaseSpaceSource::m_free_phasespace_to_gpu()
{

//    ui32 n = mh_phasespace->tot_particles;

    /// Device pointers allocation
    f32 *energy;
//    HANDLE_ERROR( cudaMalloc((void**) &energy, n*sizeof(f32)) );
    f32 *pos_x;
//    HANDLE_ERROR( cudaMalloc((void**) &pos_x, n*sizeof(f32)) );
    f32 *pos_y;
//    HANDLE_ERROR( cudaMalloc((void**) &pos_y, n*sizeof(f32)) );
    f32 *pos_z;
//    HANDLE_ERROR( cudaMalloc((void**) &pos_z, n*sizeof(f32)) );

    f32 *dir_x;
//    HANDLE_ERROR( cudaMalloc((void**) &dir_x, n*sizeof(f32)) );
    f32 *dir_y;
//    HANDLE_ERROR( cudaMalloc((void**) &dir_y, n*sizeof(f32)) );
    f32 *dir_z;
//    HANDLE_ERROR( cudaMalloc((void**) &dir_z, n*sizeof(f32)) );

    ui8 *ptype;
//    HANDLE_ERROR( cudaMalloc((void**) &ptype, n*sizeof(ui8)) );

    /// Unbind

    /// Bind data to the struct
    HANDLE_ERROR( cudaMemcpy( &energy, &(md_phasespace->energy),
                              sizeof(md_phasespace->energy), cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaMemcpy( &pos_x, &(md_phasespace->pos_x),
                              sizeof(md_phasespace->pos_x), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( &pos_y, &(md_phasespace->pos_y),
                              sizeof(md_phasespace->pos_y), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( &pos_z, &(md_phasespace->pos_z),
                              sizeof(md_phasespace->pos_z), cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaMemcpy( &dir_x, &(md_phasespace->dir_x),
                              sizeof(md_phasespace->dir_x), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( &dir_y, &(md_phasespace->dir_y),
                              sizeof(md_phasespace->dir_y), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( &dir_z, &(md_phasespace->dir_z),
                              sizeof(md_phasespace->dir_z), cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaMemcpy( &ptype, &(md_phasespace->ptype),
                              sizeof(md_phasespace->ptype), cudaMemcpyDeviceToHost ) );


    /// Free memory
    cudaFree( energy );

    cudaFree( pos_x );
    cudaFree( pos_y );
    cudaFree( pos_z );

    cudaFree( dir_x );
    cudaFree( dir_y );
    cudaFree( dir_z );

    cudaFree( ptype );

    cudaFree( md_phasespace );

//    cudaFree( md_phasespace->energy );

//    cudaFree( (md_phasespace->pos_x) );
//    cudaFree( (md_phasespace->pos_y) );
//    cudaFree( (md_phasespace->pos_z) );

//    cudaFree( (md_phasespace->dir_x) );
//    cudaFree( (md_phasespace->dir_y) );
//    cudaFree( (md_phasespace->dir_z) );

//    cudaFree( (md_phasespace->ptype) );

//    cudaFree( md_phasespace );

    md_phasespace = nullptr;

}

//========= Main function ============================================

// Mandatory function, abstract from GGEMSSource. This function is called
// by GGEMS to initialize and load all necessary data into the graphic card
void PhaseSpaceSource::initialize (GlobalSimulationParametersData *h_params )
{    
    // Load the phasespace file (CPU & GPU)
    if ( m_phasespace_file != "" )
    {
        m_load_phasespace_file();
    }

    // Load transformation file (CPU & GPU)
    if ( m_transformation_file != "" )
    {
        m_load_transformation_file();
    }

    // Check if everything was set properly
    if ( !m_check_mandatory() )
    {
        GGcerr << "Phasespace source was not set properly!" << GGendl;
        exit_simulation();
    }

    // Store global parameters
    mh_params = h_params;

    // Check if the physics is set properly considering the phasespace file
    bool there_is_photon = mh_params->physics_list[PHOTON_COMPTON] ||
                           mh_params->physics_list[PHOTON_PHOTOELECTRIC] ||
                           mh_params->physics_list[PHOTON_RAYLEIGH];

    bool there_is_electron = mh_params->physics_list[ELECTRON_IONISATION] ||
                             mh_params->physics_list[ELECTRON_BREMSSTRAHLUNG] ||
                             mh_params->physics_list[ELECTRON_MSC];

    if ( !there_is_electron && mh_phasespace->nb_electrons != 0 )
    {
        GGcerr << "Phasespace file contains " << mh_phasespace->nb_electrons
               << " electrons and there are no electron physics effects enabled!"
               << GGendl;
        exit_simulation();
    }

    if ( !there_is_photon && mh_phasespace->nb_photons != 0 )
    {
        GGcerr << "Phasespace file contains " << mh_phasespace->nb_photons
               << " photons and there are no photon physics effects enabled!"
               << GGendl;
        exit_simulation();
    }

    if ( mh_phasespace->nb_positrons != 0 )
    {
        GGcerr << "Phasespace file contains " << mh_phasespace->nb_positrons
               << " positrons and there are no positron physics effects enabled!"
               << GGendl;
        exit_simulation();
    }

    // Check transformation
    if ( m_transform.nb_sources == 0 )
    {
        // Allocation of one transformation
        m_transform_allocation( 1 );

        m_transform.tx[ 0 ] = 0;
        m_transform.ty[ 0 ] = 0;
        m_transform.tz[ 0 ] = 0;

        m_transform.rx[ 0 ] = 0;
        m_transform.ry[ 0 ] = 0;
        m_transform.rz[ 0 ] = 0;

        m_transform.sx[ 0 ] = 1;
        m_transform.sy[ 0 ] = 1;
        m_transform.sz[ 0 ] = 1;

        m_transform.cdf[ 0 ] = 1;
        m_transform.nb_sources = 1;
    }

    // Cut off the number of particles according the user parameter
    if ( m_nb_part_max != -1 )
    {
        if ( m_nb_part_max < mh_phasespace->tot_particles )
        {
            mh_phasespace->tot_particles = m_nb_part_max;
        }
    }

    // Copy data on GPU
    m_copy_phasespace_to_gpu();

    // Some verbose if required
    if ( mh_params->display_memory_usage )
    {
        ui32 mem = 29 * mh_phasespace->tot_particles;
        GGcout_mem("PhaseSpace source", mem);
    }

}

// Mandatory function, abstract from GGEMSSource. This function is called
// by GGEMS to fill particle buffer of new fresh particles, which is the role
// of any source.
void PhaseSpaceSource::get_primaries_generator (ParticlesData *d_particles )
{

    // Defined threads and grid
    dim3 threads, grid;
    threads.x = mh_params->gpu_block_size;
    grid.x = ( mh_params->size_of_particles_batch + mh_params->gpu_block_size - 1 ) / mh_params->gpu_block_size;

    PHSPSRC::phsp_point_source<<<grid, threads>>>( d_particles, md_phasespace, m_transform );
    cuda_error_check( "Error ", " Kernel_phasespace_source" );
    cudaDeviceSynchronize();

}

#endif

















