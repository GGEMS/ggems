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
#include "materials.cuh"

// Struct that handle Vox Volume data
template <typename aType>
struct VoxVolumeData {
    ui16 nb_vox_x, nb_vox_y, nb_vox_z;
    ui32 number_of_voxels;
    f32 spacing_x, spacing_y, spacing_z;
    f32 off_x, off_y, off_z;
    f32 xmin, xmax, ymin, ymax, zmin, zmax;

    aType *values;
};

// Voxelized phantom
class VoxelizedPhantom : public BaseObject {
    public:
        VoxelizedPhantom();
        ~VoxelizedPhantom() {}

        void load_from_raw(std::string volume_name, std::string range_name,
                           i32 nx, i32 ny, i32 nz, f32 sx, f32 sy, f32 sz);

        void load_from_mhd(std::string volume_name, std::string range_name);

        void initialize(GlobalSimulationParameters parameters);

        void set_offset(f32 x, f32 y, f32 z);

        VoxVolumeData<ui16> data_h;
        VoxVolumeData<ui16> data_d;
        std::vector<std::string> list_of_materials;    

    private:
        template<typename Type> void m_define_materials_from_range(Type *raw_data, std::string range_name);
        //void m_define_materials_from_range(ui16 *raw_data, std::string range_name);
        void m_copy_phantom_cpu2gpu();
        bool m_check_mandatory();
        TxtReader m_txt_reader;
                
};




#endif
