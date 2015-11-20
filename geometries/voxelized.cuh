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
#include "txt_reader.cuh"
#include "base_object.cuh"

// Table containing every definition of the materials used in the world
struct VoxVolume {
    ui16 nb_vox_x, nb_vox_y, nb_vox_z;
    ui32 number_of_voxels;
    f32 spacing_x, spacing_y, spacing_z;
    f32 org_x, org_y, org_z;
    f32 xmin, xmax, ymin, ymax, zmin, zmax;

    ui16 *data;
};

// Voxelized phantom
class Voxelized : public BaseObject {
    public:
        Voxelized();
        ~Voxelized() {}

        void load_from_raw(std::string volume_name, std::string range_name,
                           i32 nx, i32 ny, i32 nz, f32 sx, f32 sy, f32 sz);

        void load_from_mhd(std::string volume_name, std::string range_name);

        void set_origin(f32 x, f32 y, f32 z);

        //void set_color_map(std::string matname, Color col, f32 alpha);

        VoxVolume volume;

        std::vector<std::string> list_of_materials;

        // Only for display purpose
        //std::vector<std::string> show_mat;
        //std::vector<Color> show_colors;
        //std::vector<f32> show_transparencies;

    private:
        void m_define_materials_from_range(f32 *raw_data, std::string range_name);
        void m_define_materials_from_range(ui16 *raw_data, std::string range_name);
        TxtReader m_txt_reader;
                
};




#endif
