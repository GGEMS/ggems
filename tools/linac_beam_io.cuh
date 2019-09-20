// GGEMS Copyright (C) 2017

/*!
 * \file linac_beam_io.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Tuesday April 5, 2017
 *
 * v0.1: JB - First code
 *
 */

#ifndef LINAC_BEAM_IO_CUH
#define LINAC_BEAM_IO_CUH

#include "global.cuh"
#include "vector.cuh"

// Read LINAC beam configuration file file
class LinacBeamIO {

    public:
        LinacBeamIO(){}
        ~LinacBeamIO(){}

        f32xyz get_gantry_angles( std::string beam_filename, ui32 beam_index, ui32 control_point_index );

    private:
        std::vector< std::string > m_split_txt(std::string line);


};




#endif
