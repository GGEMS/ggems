// GGEMS Copyright (C) 2015

/*!
 * \file voxelized.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */

#ifndef VOXELIZED_CU
#define VOXELIZED_CU

#include "voxelized.cuh"

VoxelizedPhantom::VoxelizedPhantom() {
    volume.data_h.xmin = 0.0f;
    volume.data_h.xmax = 0.0f;
    volume.data_h.ymin = 0.0f;
    volume.data_h.ymax = 0.0f;
    volume.data_h.zmin = 0.0f;
    volume.data_h.zmax = 0.0f;

    // Init pointer
    volume.data_h.values = NULL;
}

///:: Private



// Convert range data into material ID
void VoxelizedPhantom::m_define_materials_from_range(ui16 *raw_data, std::string range_name) {

    ui16 start, stop;
    std::string mat_name, line;
    ui32 i;
    ui16 val;
    ui16 mat_index = 0;
    
    // Data allocation
    volume.data_h.values = (ui16*)malloc(volume.data_h.number_of_voxels * sizeof(ui16));

    // Read range file
    std::ifstream file(range_name.c_str());
    if(!file) {
        printf("Error, file %s not found \n", range_name.c_str());
        exit_simulation();
    }
    while (file) {
        m_txt_reader.skip_comment(file);
        std::getline(file, line);

        if (file) {
            start = m_txt_reader.read_start_range(line);
            stop  = m_txt_reader.read_stop_range(line);
            mat_name = m_txt_reader.read_mat_range(line);
            list_of_materials.push_back(mat_name);            

            // build labeled phantom according range data
            i=0; while (i < volume.data_h.number_of_voxels) {
                val = raw_data[i];
                if ((val==start && val==stop) || (val>=start && val<stop)) {
                    volume.data_h.values[i] = mat_index;
                }
                ++i;
            } // over the volume

        } // new material range
        ++mat_index;

    } // read file

}

// Convert range data into material ID
void VoxelizedPhantom::m_define_materials_from_range(f32 *raw_data, std::string range_name) {

    f32 start, stop;
    std::string mat_name, line;
    ui32 i;
    f32 val;
    ui16 mat_index = 0;

    // Data allocation
    volume.data_h.values = (ui16*)malloc(volume.data_h.number_of_voxels * sizeof(ui16));

    // Read range file
    std::ifstream file(range_name.c_str());
    if(!file) {
        printf("Error, file %s not found \n", range_name.c_str());
        exit_simulation();
    }
    while (file) {
        m_txt_reader.skip_comment(file);
        std::getline(file, line);

        if (file) {
            start = m_txt_reader.read_start_range(line);
            stop  = m_txt_reader.read_stop_range(line);
            mat_name = m_txt_reader.read_mat_range(line);
            list_of_materials.push_back(mat_name);
            //printf("IND %i MAT %s \n", mat_index, mat_name.c_str());

            // build labeled phantom according range data
            i=0; while (i < volume.data_h.number_of_voxels) {
                val = raw_data[i];
                if ((val==start && val==stop) || (val>=start && val<stop)) {
                    volume.data_h.values[i] = mat_index;
                }
                ++i;
            } // over the volume

        } // new material range
        ++mat_index;

    } // read file

}

///:: Main functions

// Load phantom from binary data (f32)
void VoxelizedPhantom::load_from_raw(std::string volume_name, std::string range_name,
                              i32 nx, i32 ny, i32 nz, f32 sx, f32 sy, f32 sz) {

    /////////////// First read the raw data from the phantom ////////////////////////

    volume.data_h.number_of_voxels = nx*ny*nz;
    volume.data_h.nb_vox_x = nx;
    volume.data_h.nb_vox_y = ny;
    volume.data_h.nb_vox_z = nz;
    volume.data_h.spacing_x = sx;
    volume.data_h.spacing_y = sy;
    volume.data_h.spacing_z = sz;
    ui32 mem_size = sizeof(f32) * volume.data_h.number_of_voxels;

    FILE *pfile = fopen(volume_name.c_str(), "rb");
    if (!pfile) {
        printf("Error when loading raw data file: %s\n", volume_name.c_str());
        exit_simulation();
    }

    f32 *raw_data = (f32*)malloc(mem_size);
    fread(raw_data, sizeof(f32), volume.data_h.number_of_voxels, pfile);

    fclose(pfile);

    /////////////// Then, convert the raw data into material id //////////////////////

    m_define_materials_from_range(raw_data, range_name);

    // Free memory
    free(raw_data);

    ///////////// Define a bounding box for this phantom //////////////////////////////

    f32 h_lengthx = volume.data_h.nb_vox_x * volume.data_h.spacing_x * 0.5f;
    f32 h_lengthy = volume.data_h.nb_vox_y * volume.data_h.spacing_y * 0.5f;
    f32 h_lengthz = volume.data_h.nb_vox_z * volume.data_h.spacing_z * 0.5f;

    volume.data_h.xmin = -h_lengthx; volume.data_h.xmax = h_lengthx;
    volume.data_h.ymin = -h_lengthy; volume.data_h.ymax = h_lengthy;
    volume.data_h.zmin = -h_lengthz; volume.data_h.zmax = h_lengthz;

    volume.data_h.off_x = h_lengthx;
    volume.data_h.off_y = h_lengthy;
    volume.data_h.off_z = h_lengthz;

}

// Load phantom from mhd file (only f32 data)
void VoxelizedPhantom::load_from_mhd(std::string volume_name, std::string range_name) {

    /////////////// First read the MHD file //////////////////////

    std::string line, key;
    i32 nx=-1, ny=-1, nz=-1;
    f32 sx=0, sy=0, sz=0;
    f32 ox=-1, oy=-1, oz=-1;

    // Watchdog
    std::string ObjectType="", BinaryData="", BinaryDataByteOrderMSB="", CompressedData="",
                ElementType="", ElementDataFile="";
    i32 NDims=0;

    // Read range file
    std::ifstream file(volume_name.c_str());
    if(!file) {
        printf("Error, file %s not found \n", volume_name.c_str());
        exit_simulation();
    }
    while (file) {
        m_txt_reader.skip_comment(file);
        std::getline(file, line);

        if (file) {
            key = m_txt_reader.read_key(line);
            if (key=="ObjectType")              ObjectType = m_txt_reader.read_key_string_arg(line);
            if (key=="NDims")                   NDims = m_txt_reader.read_key_i32_arg(line);
            if (key=="BinaryData")              BinaryData = m_txt_reader.read_key_string_arg(line);
            if (key=="BinaryDataByteOrderMSB")  BinaryDataByteOrderMSB = m_txt_reader.read_key_string_arg(line);
            if (key=="CompressedData")          CompressedData = m_txt_reader.read_key_string_arg(line);
            //if (key=="TransformMatrix") printf("Matrix\n");
            if (key=="Offset")                  {
                                                ox = m_txt_reader.read_key_f32_arg_atpos(line, 0);
                                                oy = m_txt_reader.read_key_f32_arg_atpos(line, 1);
                                                oz = m_txt_reader.read_key_f32_arg_atpos(line, 2);
            }
            //if (key=="CenterOfRotation") printf("CoR\n");
            if (key=="ElementSpacing") {
                                                sx = m_txt_reader.read_key_f32_arg_atpos(line, 0);
                                                sy = m_txt_reader.read_key_f32_arg_atpos(line, 1);
                                                sz = m_txt_reader.read_key_f32_arg_atpos(line, 2);
            }
            if (key=="DimSize") {
                                                nx = m_txt_reader.read_key_i32_arg_atpos(line, 0);
                                                ny = m_txt_reader.read_key_i32_arg_atpos(line, 1);
                                                nz = m_txt_reader.read_key_i32_arg_atpos(line, 2);
            }

            //if (key=="AnatomicalOrientation") printf("Anato\n");
            if (key=="ElementType")             ElementType = m_txt_reader.read_key_string_arg(line);
            if (key=="ElementDataFile")         ElementDataFile = m_txt_reader.read_key_string_arg(line);
        }

    } // read file

    // Check header
    if (ObjectType != "Image") {
        printf("Error, mhd header: ObjectType = %s\n", ObjectType.c_str());
        exit_simulation();
    }
    if (BinaryData != "True") {
        printf("Error, mhd header: BinaryData = %s\n", BinaryData.c_str());
        exit_simulation();
    }
    if (BinaryDataByteOrderMSB != "False") {
        printf("Error, mhd header: BinaryDataByteOrderMSB = %s\n", BinaryDataByteOrderMSB.c_str());
        exit_simulation();
    }
    if (CompressedData != "False") {
        printf("Error, mhd header: CompressedData = %s\n", CompressedData.c_str());
        exit_simulation();
    }
    if (ElementType != "MET_FLOAT" && ElementType != "MET_USHORT") {
        printf("Error, mhd header: ElementType = %s\n", ElementType.c_str());
        exit_simulation();
    }
    if (ElementDataFile == "") {
        printf("Error, mhd header: ElementDataFile = %s\n", ElementDataFile.c_str());
        exit_simulation();
    }
    if (NDims != 3) {
        printf("Error, mhd header: NDims = %i\n", NDims);
        exit_simulation();
    }

    if (nx == -1 || ny == -1 || nz == -1 || sx == 0 || sy == 0 || sz == 0) {
        printf("Error when loading mhd file (unknown dimension and spacing)\n");
        printf("   => dim %i %i %i - spacing %f %f %f\n", nx, ny, nz, sx, sy, sz);
        exit_simulation();
    }
    // Read data
    FILE *pfile = fopen(ElementDataFile.c_str(), "rb");
    if (!pfile) {
        std::string nameWithRelativePath = volume_name;
        i32 lastindex = nameWithRelativePath.find_last_of(".");
        nameWithRelativePath = nameWithRelativePath.substr(0, lastindex);
        nameWithRelativePath+=".raw";
        pfile = fopen(nameWithRelativePath.c_str(), "rb");
        if (!pfile) {
            printf("Error when loading mhd file: %s\n", ElementDataFile.c_str());
            exit_simulation();
        }
    }

    volume.data_h.number_of_voxels = nx*ny*nz;
    volume.data_h.nb_vox_x = nx;
    volume.data_h.nb_vox_y = ny;
    volume.data_h.nb_vox_z = nz;
    volume.data_h.spacing_x = sx;
    volume.data_h.spacing_y = sy;
    volume.data_h.spacing_z = sz;
    
    if(ElementType == "MET_FLOAT") {
      ui32 mem_size = sizeof(f32) * volume.data_h.number_of_voxels;

      f32 *raw_data = (f32*)malloc(mem_size);
      fread(raw_data, sizeof(f32), volume.data_h.number_of_voxels, pfile);
      fclose(pfile);
      
      /////////////// Then, convert the raw data into material id //////////////////////

      m_define_materials_from_range(raw_data, range_name);

      // Free memory
      free(raw_data);
    }
    
    if(ElementType == "MET_USHORT") {
      ui32 mem_size = sizeof(ui16) * volume.data_h.number_of_voxels;

      ui16 *raw_data = (ui16*)malloc(mem_size);
      fread(raw_data, sizeof(ui16), volume.data_h.number_of_voxels, pfile);
      fclose(pfile);
      /////////////// Then, convert the raw data into material id //////////////////////
      m_define_materials_from_range(raw_data, range_name);

      // Free memory
      free(raw_data);
    }  

    ///////////// Define a bounding box for this phantom //////////////////////////////

    f32 h_lengthx = volume.data_h.nb_vox_x * volume.data_h.spacing_x * 0.5f;
    f32 h_lengthy = volume.data_h.nb_vox_y * volume.data_h.spacing_y * 0.5f;
    f32 h_lengthz = volume.data_h.nb_vox_z * volume.data_h.spacing_z * 0.5f;

    // If the offset is not defined, chose the volume center
    if (ox == -1 || oy == -1 || oz == -1) {
        volume.data_h.off_x = h_lengthx;
        volume.data_h.off_y = h_lengthy;
        volume.data_h.off_z = h_lengthz;

        volume.data_h.xmin = -h_lengthx; volume.data_h.xmax = h_lengthx;
        volume.data_h.ymin = -h_lengthy; volume.data_h.ymax = h_lengthy;
        volume.data_h.zmin = -h_lengthz; volume.data_h.zmax = h_lengthz;

    } else {
        volume.data_h.off_x = ox;
        volume.data_h.off_y = oy;
        volume.data_h.off_z = oz;

        volume.data_h.xmin = -ox; volume.data_h.xmax = volume.data_h.xmin + (volume.data_h.nb_vox_x * volume.data_h.spacing_x);
        volume.data_h.ymin = -oy; volume.data_h.ymax = volume.data_h.ymin + (volume.data_h.nb_vox_y * volume.data_h.spacing_y);
        volume.data_h.zmin = -oz; volume.data_h.zmax = volume.data_h.zmin + (volume.data_h.nb_vox_z * volume.data_h.spacing_z);
    }

}

void VoxelizedPhantom::set_offset(f32 x, f32 y, f32 z) {
    volume.data_h.off_x = x;
    volume.data_h.off_y = y;
    volume.data_h.off_z = z;

    volume.data_h.xmin = -x; volume.data_h.xmax = volume.data_h.xmin + (volume.data_h.nb_vox_x * volume.data_h.spacing_x);
    volume.data_h.ymin = -y; volume.data_h.ymax = volume.data_h.ymin + (volume.data_h.nb_vox_y * volume.data_h.spacing_y);
    volume.data_h.zmin = -z; volume.data_h.zmax = volume.data_h.zmin + (volume.data_h.nb_vox_z * volume.data_h.spacing_z);
}

/*
// Define which material to display in the viewer (VRML) and how
void Voxelized::set_color_map(std::string matname, Color col, f32 alpha) {
    show_mat.push_back(matname);
    show_colors.push_back(col);
    show_transparencies.push_back(alpha);
}
*/

#endif






















