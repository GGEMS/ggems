// GGEMS Copyright (C) 2015

/*!
 * \file txt_reader.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 19 novembre 2015
 *
 *
 *
 */

#ifndef TXT_READER_CUH
#define TXT_READER_CUH

#include "global.cuh"

// Voxelized phantom
class TxtReader {

    public:
        TxtReader(){};
        ~TxtReader(){};

        void skip_comment(std::istream &);

        // Use by materials and activities range file
        f32 read_start_range(std::string);
        f32 read_stop_range(std::string);
        std::string read_mat_range(std::string);

        // Use by mhd file or other   (key = arg arg arg)
        std::string read_key(std::string);
        std::string read_key_string_arg(std::string);
        i32 read_key_i32_arg(std::string);
        f32 read_key_f32_arg(std::string);
        i32 read_key_i32_arg_atpos(std::string, i32);
        f32 read_key_f32_arg_atpos(std::string, i32);

        // Use by elements and materials data file
        std::string read_element_name(std::string);
        i32 read_element_Z(std::string);
        f32 read_element_A(std::string);

        std::string read_material_name(std::string);
        f32 read_material_density(std::string);
        ui16 read_material_nb_elements(std::string);
        std::string read_material_element(std::string);
        f32 read_material_fraction(std::string);

        // General functions
        i32 read_i32_atpos(std::string, i32);
        f32 read_f32_atpos(std::string, i32);

    private:
        std::string m_remove_white_space(std::string);
        // TODO: Carefull this function was never test !!!
        std::vector< std::string > m_split_txt(std::string);

};

std::vector<std::string> split_vector(std::string str, std::string split);

template < typename T > std::string to_string ( const T& n )
{
    std::ostringstream stm ;
    stm << n ;
    return stm.str() ;
}

//std::string insert_string_before_format(std::string filename, std::string insert);


#endif
