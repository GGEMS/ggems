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

    m_check_sum = 0;
    m_record_length = 0;
    m_nb_photons = 0;
    m_nb_electrons = 0;
    m_nb_positons = 0;

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
                std::stringstream(list[ 0 ]) >> m_exfloat_flag;

                std::getline(file, line);
                list = m_split_txt( line );
                std::stringstream(list[ 0 ]) >> m_exlongs_flag;

                std::getline(file, line);
                list = m_split_txt( line );
                std::stringstream(list[ 0 ]) >> m_genint_flag;
            }

        }

    } // read file

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


#endif












