// GGEMS Copyright (C) 2015

/*!
 * \file phasespace_io.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 11/03/2016
 * \date 27/04/2016 - Add binary data reader - JB
 *
 *
 */

#ifndef PHASESPACE_IO_CUH
#define PHASESPACE_IO_CUH

#include "global.cuh"
#include "txt_reader.cuh"

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

struct PhaseSpaceData
{
    f32 *energy;

    f32 *pos_x;
    f32 *pos_y;
    f32 *pos_z;

    f32 *dir_x;
    f32 *dir_y;
    f32 *dir_z;

    ui8 *ptype;

    ui32 tot_particles;
    ui32 nb_photons;
    ui32 nb_electrons;
    ui32 nb_positrons;
};

// Read PhaseSpace file
class PhaseSpaceIO {

    public:
        PhaseSpaceIO();
        ~PhaseSpaceIO(){}

        PhaseSpaceData read_phasespace_file( std::string filename );

    private:
        // IAEA format
        void m_read_IAEA_header( std::string filename );
        PhaseSpaceData m_read_IAEA_data();
        // MHD format
        void m_read_MHD_header( std::string filename );
        PhaseSpaceData m_read_MHD_data();

    private:
        std::string m_filename;

        ui64 m_check_sum;
        ui32 m_record_length;
        ui32 m_nb_photons;
        ui32 m_nb_electrons;
        ui32 m_nb_positrons;

        bool m_x_flag;
        bool m_y_flag;
        bool m_z_flag;
        bool m_u_flag;
        bool m_v_flag;
        bool m_w_flag;
        bool m_weight_flag;
        bool m_exfloat_flag;
        bool m_exlongs_flag;
        bool m_genint_flag;

        std::string m_header_loaded;     // Specify which header format was loaded
        std::string m_compression_type;  // If comprsesed data which method

        void m_skip_comment(std::istream & is);
        std::string m_read_key(std::string txt);
        std::vector< std::string > m_split_txt(std::string line);


};




#endif
