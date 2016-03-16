// GGEMS Copyright (C) 2015

/*!
 * \file iaea_io.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 11/03/2016
 *
 *
 *
 */

#ifndef IAEA_IO_CU
#define IAEA_IO_CU


#include "iaea_io.cuh"

/////:: Private

// Skip comment starting with "#"
void IAEAIO::m_skip_comment(std::istream & is) {
    i8 c;
    i8 line[1024];
    if (is.eof()) return;
    is >> c;
    while (is && (c=='/')) {
        is.getline(line, 1024);
        is >> c;
        if (is.eof()) return;
    }
    is.unget();
}

// Read mhd key
std::string IAEAIO::m_read_key(std::string txt) {
    return txt.substr(0, txt.find(":"));
}

// Read the list of tokens in a txt line
std::vector< std::string > IAEAIO::m_split_txt(std::string line) {
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::copy(std::istream_iterator<std::string>(iss),
         std::istream_iterator<std::string>(),
         std::back_inserter(tokens));

    return tokens;
}

/////:: Main functions

IAEAIO::IAEAIO()
{
    m_header_ext = ".IAEAheader";
    m_file_ext   = ".IAEAphsp";
    m_filename   = "";

    m_check_sum = 0;
    m_record_length = 0;
    m_nb_photons = 0;
    m_nb_electrons = 0;
    m_nb_positrons = 0;

    m_x_flag = false;
    m_y_flag = false;
    m_z_flag = false;
    m_u_flag = false;
    m_v_flag = false;
    m_w_flag = false;
    m_weight_flag = false;
    m_exfloat_flag = false;
    m_exlongs_flag = false;
    m_genint_flag = false;
}

void IAEAIO::read_header( std::string filename )
{
    // Get filename without extension
    std::string file_wo_ext = filename.substr( 0, filename.find_last_of( "." ) );
    m_filename = file_wo_ext;

    // Read file
    std::ifstream file( file_wo_ext + m_header_ext );
    if( !file ) {
        GGcout << "Error, IAEA file '" << file_wo_ext + m_header_ext << "' not found!!" << GGendl;
        exit_simulation();
    }

    // Loop over the file
    std::string line, key;
    while( file ) {
        m_skip_comment( file );
        std::getline(file, line);

        if (file) {
            key = m_read_key(line);

            if ( key == "$CHECKSUM" )
            {
                std::getline(file, line);
                std::stringstream(line) >> m_check_sum;
            }

            if ( key == "$RECORD_LENGTH" )
            {
                std::getline(file, line);
                std::stringstream(line) >> m_record_length;
            }

            if ( key == "$PHOTONS" )
            {
                std::getline(file, line);
                std::stringstream(line) >> m_nb_photons;
            }

            if ( key == "$ELECTRONS" )
            {
                std::getline(file, line);
                std::stringstream(line) >> m_nb_electrons;
            }

            if ( key == "$POSITRONS" )
            {
                std::getline(file, line);
                std::stringstream(line) >> m_nb_positrons;
            }

            if ( key == "$RECORD_CONTENTS" )
            {
                std::vector< std::string > list;

                std::getline(file, line);
                list = m_split_txt( line );
                std::stringstream(list[ 0 ]) >> m_x_flag;

                std::getline(file, line);
                list = m_split_txt( line );
                std::stringstream(list[ 0 ]) >> m_y_flag;

                std::getline(file, line);
                list = m_split_txt( line );
                std::stringstream(list[ 0 ]) >> m_z_flag;

                std::getline(file, line);
                list = m_split_txt( line );
                std::stringstream(list[ 0 ]) >> m_u_flag;

                std::getline(file, line);
                list = m_split_txt( line );
                std::stringstream(list[ 0 ]) >> m_v_flag;

                std::getline(file, line);
                list = m_split_txt( line );
                std::stringstream(list[ 0 ]) >> m_w_flag;

                std::getline(file, line);
                list = m_split_txt( line );
                std::stringstream(list[ 0 ]) >> m_weight_flag;

                std::getline(file, line);
                list = m_split_txt( line );
                std::stringstream(list[ 0 ]) >> m_exfloat_flag;  // Only one extra float is considered - JB

                std::getline(file, line);
                list = m_split_txt( line );
                std::stringstream(list[ 0 ]) >> m_exlongs_flag;  // Only one extra long is considered - JB


                std::getline(file, line);
                list = m_split_txt( line );
                std::stringstream(list[ 0 ]) >> m_genint_flag;
            }

        }

    } // read file

    if ( !m_x_flag )
    {
        GGcerr << "IAEA header X values are missing!" << GGendl;
        exit_simulation();
    }

    if ( !m_y_flag )
    {
        GGcerr << "IAEA header Y values are missing!" << GGendl;
        exit_simulation();
    }

    if ( !m_z_flag )
    {
        GGcerr << "IAEA header Z values are missing!" << GGendl;
        exit_simulation();
    }

    if ( !m_u_flag )
    {
        GGcerr << "IAEA header U values are missing!" << GGendl;
        exit_simulation();
    }

    if ( !m_v_flag )
    {
        GGcerr << "IAEA header V values are missing!" << GGendl;
        exit_simulation();
    }

    if ( !m_w_flag )
    {
        GGcerr << "IAEA header W values are missing!" << GGendl;
        exit_simulation();
    }

/*
    GGcout << "Checksum: " << m_check_sum << GGendl;
    GGcout << "Record length: " << m_record_length << GGendl;
    GGcout << "Photons: " << m_nb_photons << GGendl;
    GGcout << "Electrons: " << m_nb_electrons << GGendl;
    GGcout << "Positrons: " << m_nb_positrons << GGendl;
    GGcout << "X is stored: " << m_x_flag << GGendl;
    GGcout << "Y is stored: " << m_y_flag << GGendl;
    GGcout << "Z is stored: " << m_z_flag << GGendl;
    GGcout << "U is stored: " << m_u_flag << GGendl;
    GGcout << "V is stored: " << m_v_flag << GGendl;
    GGcout << "W is stored: " << m_w_flag << GGendl;
    GGcout << "Weight is stored: " << m_weight_flag << GGendl;
    GGcout << "Extra float is stored: " << m_exfloat_flag << GGendl;
    GGcout << "Extra longs is stored: " << m_exlongs_flag << GGendl;
    GGcout << "Generic int is stored: " << m_genint_flag << GGendl;
*/

}


// FORMAT IAEA
// X real 4
// Y real 4
// Z real 4
// U real 4
// V real 4
// E real 4
// Weight real 4
// Type Int 2
// Sign of W Logical 1
// New history? Logical 1

// Int_extra Int 4
// or
// Float_extra Real 4
// or
// Longs extra Int 1

// Read data
IaeaType IAEAIO::read_data()
{
    if ( m_filename == "" )
    {
        GGcerr << "You need first to read the IAEA phasespace header!" << GGendl;
        exit_simulation();
    }

    // Read file
    std::string filename = m_filename + m_file_ext;
    FILE *pfile = fopen(filename.c_str(), "rb");

    if( !pfile ) {
        GGcout << "Error, IAEA file '" << filename << "' not found!!" << GGendl;
        exit_simulation();
    }

    // Total number of particles
    ui32 N = m_nb_electrons + m_nb_photons + m_nb_positrons;

    // Vars to read
    f32 U, V, W;
    i8 pType;
    i8 sign_W;
    //i8 new_history; // not used.
    f32 extra_f;
    i32 extra_l;

    // Mem allocation
    IaeaType phasespace;
    HANDLE_ERROR( cudaMallocManaged( &(phasespace.energy), N * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(phasespace.pos_x), N * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(phasespace.pos_y), N * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(phasespace.pos_z), N * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(phasespace.dir_x), N * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(phasespace.dir_y), N * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(phasespace.dir_z), N * sizeof( f32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(phasespace.ptype), N * sizeof( ui8 ) ) );
    phasespace.tot_particles = N;
    phasespace.nb_photons = m_nb_photons;
    phasespace.nb_electrons = m_nb_electrons;
    phasespace.nb_positrons = m_nb_positrons;

    // For reading data
    ui32 rec_to_read = 6;                     // energy, pos and dir are always read
    if ( m_weight_flag ) rec_to_read++;
    f32 *float_array = new f32[ rec_to_read ];

    ui32 i=0; while ( i < N )
    {
        // Particle type
        fread(&pType, sizeof(i8), 1, pfile);

        sign_W = 1;
        if( pType < 0 )
        {
            sign_W = -1;
            pType = -pType;
        }

        // ptype - photon:1 electron:2 positron:3 neutron:4 proton:5
        if ( pType == 1 ) phasespace.ptype[ i ] = PHOTON;
        if ( pType == 2 ) phasespace.ptype[ i ] = ELECTRON;
        if ( pType == 3 ) phasespace.ptype[ i ] = POSITRON;

        // Read bunch of float
        fread( float_array, sizeof(f32), rec_to_read, pfile );

        // History (not used) and energy
        // new_history = 0;
        // if( float_array[ 0 ] < 0 ) new_history = 1; // like egsnrc
        phasespace.energy[ i ] = fabs( float_array[ 0 ] ); // E in MeV

        // Pos and dir
        phasespace.pos_x[ i ] = float_array[ 1 ] *cm;   // X
        phasespace.pos_y[ i ] = float_array[ 2 ] *cm;   // Y
        phasespace.pos_z[ i ] = float_array[ 3 ] *cm;   // Z
        U = float_array[ 4 ];
        V = float_array[ 5 ];

        // Compute W
        W = 0.0f;
        f32 aux = (U*U + V*V);
        if ( aux <= 1.0 )
        {
            W = sign_W * sqrtf( 1.0f - aux);
        }
        else
        {
            aux = sqrtf( aux );
            U /= aux;
            V /= aux;
        }

        phasespace.dir_x[ i ] = U;
        phasespace.dir_y[ i ] = V;
        phasespace.dir_z[ i ] = W;

        // Extra data
        if( m_exfloat_flag ) fread(&extra_f, sizeof(f32), 1, pfile);
        if( m_exlongs_flag ) fread(&extra_l, sizeof(i32), 1, pfile);


//        printf("E %e Pos %e %e %e Dir %e %e %e Type %i\n", phasespace.energy[ i ],
//               phasespace.pos_x[ i ], phasespace.pos_y[ i ], phasespace.pos_z[ i ],
//               phasespace.dir_x[ i ], phasespace.dir_y[ i ], phasespace.dir_z[ i ],
//               phasespace.ptype[ i ]);

        ++i;

    }

    return phasespace;

}


#endif












