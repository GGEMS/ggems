// GGEMS Copyright (C) 2015

/*!
 * \file phasespace_io.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 11/03/2016
 * \date 27/04/2016 - Add binary data reader - JB
 *
 *
 */

#ifndef PHASESPACE_IO_CU
#define PHASESPACE_IO_CU


#include "phasespace_io.cuh"

/////:: Private

// Skip comment starting with "#"
void PhaseSpaceIO::m_skip_comment(std::istream & is) {
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
std::string PhaseSpaceIO::m_read_key(std::string txt) {
    return txt.substr(0, txt.find(":"));
}

// Read the list of tokens in a txt line
std::vector< std::string > PhaseSpaceIO::m_split_txt(std::string line) {
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::copy(std::istream_iterator<std::string>(iss),
         std::istream_iterator<std::string>(),
         std::back_inserter(tokens));

    return tokens;
}

/////:: Main functions

PhaseSpaceIO::PhaseSpaceIO()
{

    m_filename   = "";

    m_header_loaded = "";

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

// Read a phasespace file
PhaseSpaceData* PhaseSpaceIO::read_phasespace_file( std::string filename )
{
    std::string ext = filename.substr( filename.find_last_of( "." ) + 1 );
    if ( ext == "IAEAheader" )
    {
        m_read_IAEA_header( filename );
        return m_read_IAEA_data();
    }
    else if ( ext == "mhd" )
    {
        m_read_MHD_header( filename );
        return m_read_MHD_data();
    }
    else
    {
        GGcerr << "Phasespace source can only read data in IAEA format (.IAEAheader) or Meta Header Data (.mhd)!" << GGendl;
        exit_simulation();
    }

    return nullptr;
}

/////:: Private functions

// Read IAEA header data
void PhaseSpaceIO::m_read_IAEA_header( std::string filename )
{
    // Get filename without extension
    std::string file_wo_ext = filename.substr( 0, filename.find_last_of( "." ) );
    m_filename = file_wo_ext;

    // Read file
    std::string m_header_ext = ".IAEAheader";

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

    // Header loaded
    m_header_loaded = "iaea";

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



// IAEA Format
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

// Read data from IAEA data
PhaseSpaceData* PhaseSpaceIO::m_read_IAEA_data()
{
    if ( m_header_loaded != "iaea" )
    {
        GGcerr << "You need first to read the IAEA phasespace header!" << GGendl;
        exit_simulation();
    }

    // Read file
    std::string m_file_ext   = ".IAEAphsp";

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
    //f32 extra_f;
    //i32 extra_l;

    // Mem allocation
    PhaseSpaceData *phasespace;
    phasespace = (PhaseSpaceData*)malloc( sizeof(PhaseSpaceData) );

    phasespace->energy = (f32*)malloc( N*sizeof(f32) );
    phasespace->pos_x = (f32*)malloc( N*sizeof(f32) );
    phasespace->pos_y = (f32*)malloc( N*sizeof(f32) );
    phasespace->pos_z = (f32*)malloc( N*sizeof(f32) );
    phasespace->dir_x = (f32*)malloc( N*sizeof(f32) );
    phasespace->dir_y = (f32*)malloc( N*sizeof(f32) );
    phasespace->dir_z = (f32*)malloc( N*sizeof(f32) );
    phasespace->ptype = (ui8*)malloc( N*sizeof(ui8) );

    phasespace->tot_particles = N;
    phasespace->nb_photons = m_nb_photons;
    phasespace->nb_electrons = m_nb_electrons;
    phasespace->nb_positrons = m_nb_positrons;


    // For reading data
    ui32 rec_to_read = 6;                     // energy, pos and dir are always read
    //if ( m_weight_flag ) rec_to_read++;
    f32 *float_array = new f32[ rec_to_read ];

    // Compute extra float
    ui32 extra_read = m_record_length - 25; // 25 = Type + E + pos + dir
    i8 *garbage_array = new i8[ extra_read ];

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
        if ( pType == 1 ) phasespace->ptype[ i ] = PHOTON;
        if ( pType == 2 ) phasespace->ptype[ i ] = ELECTRON;
        if ( pType == 3 ) phasespace->ptype[ i ] = POSITRON;

        // Read bunch of float
        fread( float_array, sizeof(f32), rec_to_read, pfile );

        // History (not used) and energy
        // new_history = 0;
        // if( float_array[ 0 ] < 0 ) new_history = 1; // like egsnrc
        phasespace->energy[ i ] = fabs( float_array[ 0 ] ); // E in MeV

        // Pos and dir
        phasespace->pos_x[ i ] = float_array[ 1 ] *cm;   // X
        phasespace->pos_y[ i ] = float_array[ 2 ] *cm;   // Y
        phasespace->pos_z[ i ] = float_array[ 3 ] *cm;   // Z
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

        phasespace->dir_x[ i ] = U;
        phasespace->dir_y[ i ] = V;
        phasespace->dir_z[ i ] = W;

        // Extra data
        if ( extra_read > 0 ) fread(&garbage_array, sizeof(i8), extra_read, pfile);

        //if( m_exfloat_flag ) fread(&extra_f, sizeof(f32), 1, pfile);
        //if( m_exlongs_flag ) fread(&extra_l, sizeof(i32), 1, pfile);


//        printf("E %e Pos %e %e %e Dir %e %e %e Type %i\n", phasespace.energy[ i ],
//               phasespace.pos_x[ i ], phasespace.pos_y[ i ], phasespace.pos_z[ i ],
//               phasespace.dir_x[ i ], phasespace.dir_y[ i ], phasespace.dir_z[ i ],
//               phasespace.ptype[ i ]);

        ++i;

    }

    return phasespace;

}

// Reader header from MHD data
void PhaseSpaceIO::m_read_MHD_header( std::string filename )
{
    // Get a txt reader
    TxtReader *txt_reader = new TxtReader;    

    /////////////// First read the MHD file //////////////////////

    std::string line, key;   

    // Watchdog
    std::string ObjectType = "", ElementDataFile = "", CompressedData = "False";
    ui32 NPhotons = 0, NElectrons = 0, NPositrons = 0;

    // Read file
    std::ifstream file( filename.c_str() );
    if ( !file ) {
        GGcerr << "Error, file '" << filename << "' not found!" << GGendl;
        exit_simulation();
    }

    while ( file ) {
        txt_reader->skip_comment( file );
        std::getline( file, line );

        if ( file ) {
            key = txt_reader->read_key(line);
            if ( key == "ObjectType" )              ObjectType = txt_reader->read_key_string_arg( line );
            if ( key == "NPhotons" )                NPhotons = txt_reader->read_key_i32_arg( line );
            if ( key == "NElectrons" )              NElectrons = txt_reader->read_key_i32_arg( line );
            if ( key == "NPositrons" )              NPositrons = txt_reader->read_key_i32_arg( line );
            if ( key == "ElementDataFile" )         ElementDataFile = txt_reader->read_key_string_arg( line );
            if ( key == "CompressedData" )          CompressedData = txt_reader->read_key_string_arg( line );
        }

    } // read file

    // Check header
    if ( ObjectType != "PhaseSpace" ) {
        GGcerr << "Read phasespace header: ObjectType = " << ObjectType << " !" << GGendl;
        exit_simulation();
    }

    if ( ElementDataFile == "" ) {
        GGcerr << "Read phasespace header: ObjectType = " << ElementDataFile << " !" << GGendl;
        exit_simulation();
    }

    ui32 tot_nb_particles = NPhotons + NElectrons + NPositrons;
    if ( tot_nb_particles == 0 ) {
        GGcerr << "Read phasespace header: contains 0 particles!" << GGendl;
        exit_simulation();
    }

    // Test if relative path
    FILE *pfile = fopen(ElementDataFile.c_str(), "rb");

    // Reative path?
    if ( !pfile ) {
        std::string nameWithRelativePath = filename;
        i32 lastindex = nameWithRelativePath.find_last_of("/");
        nameWithRelativePath = nameWithRelativePath.substr(0, lastindex);
        nameWithRelativePath += ( "/" + ElementDataFile );

        m_filename = nameWithRelativePath;
    }
    else
    {
        m_filename = ElementDataFile;
        fclose( pfile );
    }

    // Store values
    m_nb_photons = NPhotons;
    m_nb_electrons = NElectrons;
    m_nb_positrons = NPositrons;
    m_compression_type = CompressedData;

    // Header loaded
    m_header_loaded = "mhd";

}


//  MHD format (not compressed)
//  type - photon:1 electron:2 positron:3 neutron:4 proton:5
//  E
//  px
//  py
//  pz
//  u
//  v
//  w

// Read data from MHD format
PhaseSpaceData* PhaseSpaceIO::m_read_MHD_data()
{

    if ( m_header_loaded != "mhd" )
    {
        GGcerr << "You need first to read the MHD phasespace header!" << GGendl;
        exit_simulation();
    }

    // Read data
    FILE *pfile = fopen(m_filename.c_str(), "rb");

    // Reative path?
    if ( !pfile ) {
        GGcerr << "Error when loading mhd file: " << m_filename << GGendl;
        exit_simulation();
    }

    // Total number of particles
    ui32 N = m_nb_electrons + m_nb_photons + m_nb_positrons;

    // Vars to read
    ui8 pType;
    f32 E;
    f32 px, py, pz;
    f32 u, v, w;

    // Mem allocation
    PhaseSpaceData *phasespace;
    phasespace = (PhaseSpaceData*)malloc( sizeof(PhaseSpaceData) );

    phasespace->energy = (f32*)malloc( N*sizeof(f32) );
    phasespace->pos_x = (f32*)malloc( N*sizeof(f32) );
    phasespace->pos_y = (f32*)malloc( N*sizeof(f32) );
    phasespace->pos_z = (f32*)malloc( N*sizeof(f32) );
    phasespace->dir_x = (f32*)malloc( N*sizeof(f32) );
    phasespace->dir_y = (f32*)malloc( N*sizeof(f32) );
    phasespace->dir_z = (f32*)malloc( N*sizeof(f32) );
    phasespace->ptype = (ui8*)malloc( N*sizeof(ui8) );

    phasespace->tot_particles = N;
    phasespace->nb_photons = m_nb_photons;
    phasespace->nb_electrons = m_nb_electrons;
    phasespace->nb_positrons = m_nb_positrons;

    // If not compressed
    if ( m_compression_type == "False" )
    {
        ui32 i = 0;
        while( i < N )
        {
            // Read a particle
            fread(&pType, sizeof(ui8), 1, pfile);
            fread(&E, sizeof(f32), 1, pfile);
            fread(&px, sizeof(f32), 1, pfile);
            fread(&py, sizeof(f32), 1, pfile);
            fread(&pz, sizeof(f32), 1, pfile);
            fread(&u, sizeof(f32), 1, pfile);
            fread(&v, sizeof(f32), 1, pfile);
            fread(&w, sizeof(f32), 1, pfile);

            // Store a particle
            if ( pType == 1 ) phasespace->ptype[ i ] = PHOTON;
            if ( pType == 2 ) phasespace->ptype[ i ] = ELECTRON;
            if ( pType == 3 ) phasespace->ptype[ i ] = POSITRON;
            phasespace->energy[ i ] = E;
            phasespace->pos_x[ i ] = px;
            phasespace->pos_y[ i ] = py;
            phasespace->pos_z[ i ] = pz;
            phasespace->dir_x[ i ] = u;
            phasespace->dir_y[ i ] = v;
            phasespace->dir_z[ i ] = w;
            ++i;
        }
    }
    else
    {
        GGcerr << "Phasespace MHD, compression method unknow: " << m_compression_type << GGendl;
        exit_simulation();
    }

    fclose( pfile );

    return phasespace;
}


#endif












