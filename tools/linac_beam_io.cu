// GGEMS Copyright (C) 2017

/*!
 * \file linac_beam_io.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Tuesday April 5, 2017
 *
 * v0.1: JB - First code
 *
 */

#ifndef LINAC_BEAM_IO_CU
#define LINAC_BEAM_IO_CU

#include "linac_beam_io.cuh"

/////// Private functions /////////////////////////////////////////////////////////////////////////

// Read the list of tokens in a txt line
std::vector< std::string > LinacBeamIO::m_split_txt( std::string line ) {

    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::copy(std::istream_iterator<std::string>(iss),
         std::istream_iterator<std::string>(),
         std::back_inserter(tokens));

    return tokens;

}

/////// Getting functions //////////////////////////////////////////////////////////////////////

f32xyz LinacBeamIO::get_gantry_angles( std::string beam_filename, ui32 beam_index, ui32 control_point_index )
{

    // Open the beam file
    std::ifstream file( beam_filename.c_str(), std::ios::in );
    if( !file )
    {
        GGcerr << "Error to open the Beam file'" << beam_filename << "'!" << GGendl;
        exit_simulation();
    }

    std::string line;
    std::vector< std::string > keys;

    // Look for the beam number
    bool find_beam = false;
    while ( file )
    {
        // Read a line
        std::getline( file, line );
        keys = m_split_txt( line );

        if ( keys.size() >= 3 )
        {
            if ( keys[ 0 ] == "Beam" && std::stoi( keys[ 2 ] ) == beam_index )
            {
                find_beam = true;
                break;
            }
        }
    }

    if ( !find_beam )
    {
        GGcerr << "Beam configuration error: beam " << beam_index << " was not found!" << GGendl;
        exit_simulation();
    }
/*
    // Then look for the number of fields
    while ( file )
    {
        // Read a line
        std::getline( file, line );

        if ( line.find("Number of Fields") != std::string::npos )
        {
            break;
        }
    }

    keys = m_split_txt( line );
    ui32 nb_fields = std::stoi( keys[ 4 ] );

    if ( m_field_index >= nb_fields )
    {
        GGcerr << "Out of index for the field number, asked: " << m_field_index
               << " but a total of field of " << nb_fields << GGendl;
        exit_simulation();
    }
 */
/*
    // Look for the number of leaves
    bool find_field = false;
    while ( file )
    {
        // Read a line
        std::getline( file, line );

        if ( line.find("Number of Leaves") != std::string::npos )
        {
            find_field = true;
            break;
        }
    }

    if ( !find_field )
    {
        GGcerr << "Beam configuration error: field " << m_field_index << " was not found!" << GGendl;
        exit_simulation();
    }

    keys = m_split_txt( line );
    ui32 nb_leaves = std::stoi( keys[ 4 ] );
    if ( mh_linac->A_nb_leaves + mh_linac->B_nb_leaves != nb_leaves )
    {
        GGcerr << "Beam configuration error, " << nb_leaves
               << " leaves were found but LINAC model have " << mh_linac->A_nb_leaves + mh_linac->B_nb_leaves
               << " leaves!" << GGendl;
        exit_simulation();
    }
*/
    // Search the required field
    while ( file )
    {
        // Read a line
        std::getline( file, line );
        keys = m_split_txt( line );

        if ( keys.size() >= 3 )
        {
            if ( keys[ 0 ] == "Control" && std::stoi( keys[ 2 ] ) == control_point_index )
            {
                break;
            }
        }
    }

    // Then read the index CDF (not use at the time, so skip the line)
    std::getline( file, line );

    // Get the gantry angle
    std::getline( file, line );

    // Check
    if ( line.find( "Gantry Angle" ) == std::string::npos )
    {
        GGcerr << "Beam configuration error, no gantry angle was found!" << GGendl;
        exit_simulation();
    }

    // Read gantry angle values
    keys = m_split_txt( line );

    // if only one angle, rotate around the z-axis
    f32xyz angles;
    if ( keys.size() == 4 )
    {
        angles = make_f32xyz( 0.0, 0.0, std::stof( keys[ 3 ] ) *deg );
    }
    else if ( keys.size() == 6 ) // non-coplanar beam, or rotation on the carousel
    {
        angles = make_f32xyz( std::stof( keys[ 3 ] ) *deg,
                              std::stof( keys[ 4 ] ) *deg,
                              std::stof( keys[ 5 ] ) *deg );
    }
    else // otherwise, it seems that there is an error somewhere
    {
        GGcerr << "Beam configuration error, gantry angle must have one angle or the three rotation angles: "
               << keys.size() - 3 << " angles found!" << GGendl;
        exit_simulation();
    }

    return angles;
}


#endif












