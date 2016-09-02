// GGEMS Copyright (C) 2015

/*!
 * \file linac_source.cu
 * \brief Linac source
 * \author Julien Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Thursday September 1st, 2016
*/

#ifndef LINAC_SOURCE_CU
#define LINAC_SOURCE_CU

#include "linac_source.cuh"

///////// GPU code ////////////////////////////////////////////////////

// Internal function that create a new particle to the buffer at the slot id
__host__ __device__ void linac_source ( ParticlesData particles, LinacSourceData linac,
                                        f32matrix44 trans, ui32 id )
{
    // Main vars
    ui32 gbl_ind, ind;
    f32 f_ind, rnd;

    // 1. Get position

    rnd = prng_uniform( particles, id );
    ind = binary_search( rnd, linac.cdf_rho, linac.rho_nb_bins );

    if ( ind == 0 )
    {
        f_ind = ind;
    }
    else
    {
        f_ind = linear_interpolation( linac.cdf_rho[ ind - 1 ],     f32(ind - 1),
                                      linac.cdf_rho[ ind],          f32(ind), rnd );
    }

    f32 rho = f_ind * linac.rho_bin_size;    // in mm
    f32 psi = prng_uniform( particles, id ) * twopi;

    f32xyz pos = { 0.0, 0.0, 0.0 };
    pos.x = rho * cos( psi );
    pos.y = rho * sin( psi );

    pos = fxyz_local_to_global_position( trans, pos );

    particles.px[ id ] = pos.x;                        // Position in mm
    particles.py[ id ] = pos.y;                        //
    particles.pz[ id ] = pos.z;                        //

    // 2. Get energy

    ui32 rho_ind = f32(linac.E_nb_bins) * rho / linac.rho_max;  // rho / spacing:   spacing = rho_max / Nbins
    gbl_ind = rho_ind * linac.E_nb_bins;                                 // 2D array, get the right E row

    rnd = prng_uniform( particles, id );
    ind = binary_search( rnd, linac.cdf_rho_E, gbl_ind+linac.E_nb_bins, gbl_ind );

    if ( ind == (gbl_ind + linac.E_nb_bins - 1) )
    {
        f_ind = ind;
    }
    else
    {
        f_ind = linear_interpolation( linac.cdf_rho_E[ ind - 1 ],     f32(ind - 1),
                                      linac.cdf_rho_E[ ind ],         f32(ind), rnd );
    }
    f32 E = (f_ind - gbl_ind) * linac.E_bin_size;

    particles.E[ id ] = E;

    // 3. Get direction

    // 3.1 Get theta

    ui32 E_ind = f32(linac.theta_nb_bins) * E / linac.E_max;  // E / spacing:   spacing = E_max / Nbins
    rho_ind = f32(linac.theta_nb_bins) * rho / linac.rho_max;
    gbl_ind = rho_ind * linac.theta_nb_bins * linac.theta_nb_bins + E_ind * linac.theta_nb_bins;

    rnd = prng_uniform( particles, id );
    ind = binary_search( rnd, linac.cdf_rho_E_theta, gbl_ind+linac.theta_nb_bins, gbl_ind );

    if ( ind == (gbl_ind + linac.theta_nb_bins - 1) )
    {
        f_ind = ind;
    }
    else
    {
        f_ind = linear_interpolation( linac.cdf_rho_E_theta[ ind -1 ],  f32(ind - 1),
                                      linac.cdf_rho_E_theta[ ind ],     f32(ind), rnd );
    }
    f32 theta = (f_ind - gbl_ind) * linac.theta_bin_size;

    // 3.2 Get phi

    ui32 theta_ind = f32(linac.phi_nb_bins) * theta / linac.theta_max;
    rho_ind = f32(linac.phi_nb_bins) * rho / linac.rho_max;
    gbl_ind = rho_ind * linac.phi_nb_bins * linac.phi_nb_bins + theta_ind * linac.phi_nb_bins;

    rnd = prng_uniform( particles, id );
    ind = binary_search( rnd, linac.cdf_rho_theta_phi, gbl_ind+linac.phi_nb_bins, gbl_ind );

    if ( ind == (gbl_ind + linac.phi_nb_bins - 1) )
    {
        f_ind = ind;
    }
    else
    {
        f_ind = linear_interpolation( linac.cdf_rho_theta_phi[ ind ],     ind,
                                      linac.cdf_rho_theta_phi[ ind + 1 ], ind + 1, rnd );
    }
    f32 phi = (f_ind-gbl_ind) * linac.phi_bin_size;

    // 3.3 Get vector

    f32xyz dir = { 0.0, 0.0, 0.0 };
    dir.x = sin( theta ) * cos( psi+phi );
    dir.y = sin( theta ) * sin( psi+phi );
    dir.z = cos (theta );

    dir = fxyz_local_to_global_direction( trans, dir );

    particles.dx[id] = dir.x;                        // Direction (unit vector)
    particles.dy[id] = dir.y;                        //
    particles.dz[id] = dir.z;                        //

    // 4. Then set the mandatory field to create a new particle
    particles.tof[id] = 0.0f;                             // Time of flight
    particles.endsimu[id] = PARTICLE_ALIVE;               // Status of the particle

    particles.level[id] = PRIMARY;                        // It is a primary particle
    particles.pname[id] = PHOTON;                         // a photon or an electron

    particles.geometry_id[id] = 0;                        // Some internal variables
    particles.next_discrete_process[id] = NO_PROCESS;     //
    particles.next_interaction_distance[id] = 0.0;        //
    particles.scatter_order[ id ] = 0;                    //


    printf("src id %i p %f %f %f d %f %f %f E %f\n", id, pos.x, pos.y, pos.z,
                                                         dir.x, dir.y, dir.z, E);

}


// Kernel to create new particles. This kernel will only call the host/device function
// beamlet source in order to get one new particle.
__global__ void kernel_linac_source ( ParticlesData particles, LinacSourceData linac,
                                      f32matrix44 trans )
{
    // Get thread id
    const ui32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( id >= particles.size ) return;

    // Get a new particle
    linac_source( particles, linac, trans, id );
}

//////// Class //////////////////////////////////////////////////////////

// Constructor
LinacSource::LinacSource() : GGEMSSource()
{
    // Set the name of the source
    set_name( "LinacSource" );

    // Init vars    
    m_axis_trans = make_f32matrix33( 1, 0, 0,
                                     0, 1, 0,
                                     0, 0, 1 );
    m_angle = make_f32xyz( 0.0, 0.0, 0.0 );
    m_org = make_f32xyz( 0.0, 0.0, 0.0 );
    m_model_filename = "";
}

// Destructor
LinacSource::~LinacSource() {}

//========== Private ===============================================

void LinacSource::m_load_linac_model()
{
    // Get a txt reader
    TxtReader *txt_reader = new TxtReader;

    /////////////// First read the MHD file //////////////////////

    std::string line, key;

    // Watchdog
    f32 rho_max = 0;
    f32 E_max = 0;
    f32 theta_max = 0;
    f32 phi_max = 0;
    ui32 rho_nb_bins = 0;
    ui32 E_nb_bins = 0;
    ui32 theta_nb_bins = 0;
    ui32 phi_nb_bins = 0;
    std::string ElementDataFile = ""; std::string ObjectType = "";

    // Read file
    std::ifstream file( m_model_filename.c_str() );
    if ( !file ) {
        GGcerr << "Error, file '" << m_model_filename << "' not found!" << GGendl;
        exit_simulation();
    }

    while ( file ) {
        txt_reader->skip_comment( file );
        std::getline( file, line );

        if ( file ) {
            key = txt_reader->read_key(line);
            if ( key == "RhoMax" )               rho_max = txt_reader->read_key_f32_arg( line );
            if ( key == "EMax" )                 E_max = txt_reader->read_key_f32_arg( line );
            if ( key == "ThetaMax" )             theta_max = txt_reader->read_key_f32_arg( line );
            if ( key == "PhiMax" )               phi_max = txt_reader->read_key_f32_arg( line );
            if ( key == "NRhoBins" )             rho_nb_bins = txt_reader->read_key_i32_arg( line );
            if ( key == "NEBins" )               E_nb_bins = txt_reader->read_key_i32_arg( line );
            if ( key == "NThetaBins" )           theta_nb_bins = txt_reader->read_key_i32_arg( line );
            if ( key == "NPhiBins" )             phi_nb_bins = txt_reader->read_key_i32_arg( line );
            if ( key == "ElementDataFile" )      ElementDataFile = txt_reader->read_key_string_arg( line );
            if ( key == "ObjectType" )           ObjectType = txt_reader->read_key_string_arg( line );
        }

    } // read file

    // Check the header
    if ( ObjectType != "VirtualLinacSource" ) {
        GGcerr << "Linac source model header: not a virtual source model, ObjectType = " << ObjectType << " !" << GGendl;
        exit_simulation();
    }

    if ( ElementDataFile == "" ) {
        GGcerr << "Linac source model header: ElementDataFile was not specified!" << GGendl;
        exit_simulation();
    }

    if ( rho_max == 0 || E_max == 0 || theta_max == 0 || phi_max == 0 || rho_nb_bins == 0 ||
         E_nb_bins == 0 || theta_nb_bins == 0 || phi_nb_bins == 0 )
    {
        GGcerr << "Missing parameter(s) in the header of Linac source model file!" << GGendl;
        exit_simulation();
    }

    // Store data and mem allocation
    m_linac_source_data.rho_max = rho_max;
    m_linac_source_data.E_max = E_max;
    m_linac_source_data.theta_max = theta_max;
    m_linac_source_data.phi_max = phi_max;

    m_linac_source_data.rho_nb_bins = rho_nb_bins;
    m_linac_source_data.E_nb_bins = E_nb_bins;
    m_linac_source_data.theta_nb_bins = theta_nb_bins;
    m_linac_source_data.phi_nb_bins = phi_nb_bins;

    m_linac_source_data.rho_bin_size = rho_max / (f32)rho_nb_bins;
    m_linac_source_data.E_bin_size = E_max / (f32)E_nb_bins;
    m_linac_source_data.theta_bin_size = theta_max / (f32)theta_nb_bins;
    m_linac_source_data.phi_bin_size = phi_max / (f32)phi_nb_bins;

    HANDLE_ERROR( cudaMallocManaged( &(m_linac_source_data.cdf_rho), rho_nb_bins * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac_source_data.cdf_rho_E), E_nb_bins * E_nb_bins * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac_source_data.cdf_rho_E_theta),
                                     theta_nb_bins * theta_nb_bins * theta_nb_bins * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(m_linac_source_data.cdf_rho_theta_phi),
                                     phi_nb_bins * phi_nb_bins * phi_nb_bins * sizeof( f32 ) ) );

    // Read data
    FILE *pfile = fopen(ElementDataFile.c_str(), "rb");

    // Relative path?
    if (!pfile) {
        std::string nameWithRelativePath = m_model_filename;
        i32 lastindex = nameWithRelativePath.find_last_of("/");
        nameWithRelativePath = nameWithRelativePath.substr(0, lastindex);
        nameWithRelativePath += ( "/" + ElementDataFile );

        pfile = fopen(nameWithRelativePath.c_str(), "rb");
        if ( !pfile )
        {
            GGcerr << "Error, file " << ElementDataFile << " not found " << GGendl;
            exit_simulation();
        }
    }

    fread( m_linac_source_data.cdf_rho, sizeof(f32), rho_nb_bins, pfile );
    fread( m_linac_source_data.cdf_rho_E, sizeof(f32), E_nb_bins * E_nb_bins, pfile );
    fread( m_linac_source_data.cdf_rho_E_theta, sizeof(f32), theta_nb_bins * theta_nb_bins * theta_nb_bins, pfile );
    fread( m_linac_source_data.cdf_rho_theta_phi, sizeof(f32), phi_nb_bins * phi_nb_bins * phi_nb_bins, pfile );

    fclose( pfile );
}

//========== Setting ===============================================

// Setting the axis transformation matrix
void LinacSource::set_frame_axis( f32 m00, f32 m01, f32 m02,
                                    f32 m10, f32 m11, f32 m12,
                                    f32 m20, f32 m21, f32 m22 )
{
    m_axis_trans.m00 = m00;
    m_axis_trans.m01 = m01;
    m_axis_trans.m02 = m02;
    m_axis_trans.m10 = m10;
    m_axis_trans.m11 = m11;
    m_axis_trans.m12 = m12;
    m_axis_trans.m20 = m20;
    m_axis_trans.m21 = m21;
    m_axis_trans.m22 = m22;
}

// Setting orientation of the beamlet
void LinacSource::set_frame_rotation( f32 agx, f32 agy, f32 agz )
{
    m_angle = make_f32xyz( agx, agy, agz );
}

// Setting the distance between the beamlet plane and the isocenter
void LinacSource::set_frame_position( f32 posx, f32 posy, f32 posz )
{
    m_org = make_f32xyz( posx, posy, posz );
}

// Setting Linac source model
void LinacSource::set_model_filename( std::string filename )
{
    m_model_filename = filename;
}

//========== Getting ===============================================

f32matrix44 LinacSource::get_transformation_matrix()
{
    return m_transform;
}

//========= Main function ============================================

// Mandatory function, abstract from GGEMSSource. This function is called
// by GGEMS to initialize and load all necessary data into the graphic card
void LinacSource::initialize ( GlobalSimulationParameters params )
{
    // Some checking
    if ( m_model_filename == "" )
    {
        GGcerr << "No filename for the Linac source model was specified!" << GGendl;
        exit_simulation();
    }

    std::string ext = m_model_filename.substr( m_model_filename.find_last_of( "." ) + 1 );
    if ( ext != "mhd" )
    {
        GGcerr << "Linac source model file must be in Meta Header Data (.mhd)!" << GGendl;
        exit_simulation();
    }

    // Read and load data
    m_load_linac_model();

    // Store global parameters: params are provided by GGEMS and are used to
    // know different information about the simulation. For example if the targeted
    // device is a CPU or a GPU.
    m_params = params;

    // Compute the transformation matrix (Beamlet plane is set along the x-axis (angle 0))
    TransformCalculator *trans = new TransformCalculator;
    trans->set_translation( m_org );
    trans->set_rotation( m_angle );
    trans->set_axis_transformation( m_axis_trans );
    m_transform = trans->get_transformation_matrix();
    delete trans;

    // Some verbose if required
    if ( params.data_h.display_memory_usage )
    {
        ui32 mem = 4 * ( m_linac_source_data.rho_nb_bins +
                         m_linac_source_data.E_nb_bins*m_linac_source_data.E_nb_bins +
                         m_linac_source_data.theta_nb_bins*m_linac_source_data.theta_nb_bins*m_linac_source_data.theta_nb_bins +
                         m_linac_source_data.phi_nb_bins*m_linac_source_data.phi_nb_bins*m_linac_source_data.phi_nb_bins );
        GGcout_mem("Linac source", mem);
    }

}

// Mandatory function, abstract from GGEMSSource. This function is called
// by GGEMS to fill particle buffer of new fresh particles, which is the role
// of any source.
void LinacSource::get_primaries_generator ( Particles particles )
{

    // If CPU running, do it on CPU
    if ( m_params.data_h.device_target == CPU_DEVICE )
    {

        // Loop over the particle buffer
        ui32 id=0;
        while( id < particles.size )
        {
            // Call a point source that get a new particle at a time. In this case data from host (CPU)
            // is passed to the function (particles.data_h).
            linac_source( particles.data_h, m_linac_source_data,
                          m_transform,
                          id );
            ++id;
        }

    }
    // If GPU running, do it on GPU
    else if ( m_params.data_h.device_target == GPU_DEVICE )
    {

        // Defined threads and grid
        dim3 threads, grid;
        threads.x = m_params.data_h.gpu_block_size;
        grid.x = ( particles.size + m_params.data_h.gpu_block_size - 1 ) / m_params.data_h.gpu_block_size;

        // Call GPU kernel of a point source that get fill the complete particle buffer. In this case data
        // from device (GPU) is passed to the kernel (particles.data_d).
        kernel_linac_source<<<grid, threads>>>( particles.data_d, m_linac_source_data,
                                                m_transform );
        cuda_error_check( "Error ", " Kernel_beamlet_source" );
    }

}

#endif

















