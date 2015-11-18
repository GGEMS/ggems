// GGEMS Copyright (C) 2015

/*!
 * \file voxelized.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */

#ifndef VOXELIZED_CUH
#define VOXELIZED_CUH

#include "global.cuh"
#include "base_object.cuh"

// Voxelized phantom
class Voxelized : public BaseObject {
    public:
        Voxelized();
        ~Voxelized() {}

        void load_from_raw(std::string volume_name, std::string range_name,
                           i32 nx, i32 ny, i32 nz, f32 sx, f32 sy, f32 sz);

        void load_from_mhd(std::string volume_name, std::string range_name);

        //void set_color_map(std::string matname, Color col, f32 alpha);

        f32 *data;

        ui16 nb_vox_x, nb_vox_y, nb_vox_z;
        ui32 number_of_voxels;
        f32 spacing_x, spacing_y, spacing_z;
        f32 offset_x, offset_y, offset_z;

        std::vector<std::string> list_of_materials;

        // Only for display purpose
        //std::vector<std::string> show_mat;
        //std::vector<Color> show_colors;
        //std::vector<f32> show_transparencies;

    private:
        void m_define_materials_from_range(f32 *raw_data, std::string range_name);
        void m_define_materials_from_range(ui16 *raw_data, std::string range_name);
        
        // TODO this can be moved to a dedicated class TxtReader
        void m_skip_comment(std::istream &);
        std::string m_remove_white_space(std::string);

        // for range file
        f32 m_read_start_range(std::string);
        f32 m_read_stop_range(std::string);
        std::string m_read_mat_range(std::string);

        // for mhd
        std::string m_read_mhd_key(std::string);
        std::string m_read_mhd_string_arg(std::string);
        i32 m_read_mhd_int(std::string);
        i32 m_read_mhd_int_atpos(std::string, i32);
        f32 m_read_mhd_f32_atpos(std::string, i32);
};




#endif
