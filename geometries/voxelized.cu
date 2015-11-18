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

Voxelized::Voxelized() {
    xmin = 0.0f;
    xmax = 0.0f;
    ymin = 0.0f;
    ymax = 0.0f;
    zmin = 0.0f;
    zmax = 0.0f;

    // Init pointer
    data = NULL;
}

///:: Private

// Skip comment starting with "#"
void Voxelized::m_skip_comment(std::istream & is) {
    i8 c;
    i8 line[1024];
    if (is.eof()) return;
    is >> c;
    while (is && (c=='#')) {
        is.getline(line, 1024);
        is >> c;
        if (is.eof()) return;
    }
    is.unget();
}

// Remove all white space
std::string Voxelized::m_remove_white_space(std::string txt) {
    txt.erase(remove_if(txt.begin(), txt.end(), isspace), txt.end());
    return txt;
}

// Read start range
f32 Voxelized::m_read_start_range(std::string txt) {
    f32 res;
    txt = txt.substr(0, txt.find(" "));
    std::stringstream(txt) >> res;
    return res;
}

// Read stop range
f32 Voxelized::m_read_stop_range(std::string txt) {
    f32 res;
    txt = txt.substr(txt.find(" ")+1);
    txt = txt.substr(0, txt.find(" "));
    std::stringstream(txt) >> res;
    return res;
}

// Read material range
std::string Voxelized::m_read_mat_range(std::string txt) {
    txt = txt.substr(txt.find(" ")+1);
    txt = txt.substr(txt.find(" ")+1);
    return txt.substr(0, txt.find(" "));
}

// Read mhd key
std::string Voxelized::m_read_mhd_key(std::string txt) {
    txt = txt.substr(0, txt.find("="));
    return remove_white_space(txt);
}

// Read string mhd arg
std::string Voxelized::m_read_mhd_string_arg(std::string txt) {
    txt = txt.substr(txt.find("=")+1);
    return remove_white_space(txt);
}

// Read i32 mhd arg
i32 Voxelized::m_read_mhd_int(std::string txt) {
    i32 res;
    txt = txt.substr(txt.find("=")+1);
    txt = remove_white_space(txt);
    std::stringstream(txt) >> res;
    return res;
}

// Read int mhd arg
i32 Voxelized::m_read_mhd_int_atpos(std::string txt, i32 pos) {
    i32 res;
    txt = txt.substr(txt.find("=")+2);
    if (pos==0) {
        txt = txt.substr(0, txt.find(" "));
    }
    if (pos==1) {
        txt = txt.substr(txt.find(" ")+1);
        txt = txt.substr(0, txt.find(" "));
    }
    if (pos==2) {
        txt = txt.substr(txt.find(" ")+1);
        txt = txt.substr(txt.find(" ")+1);
    }
    std::stringstream(txt) >> res;
    return res;
}

// Read f32 mhd arg
f32 Voxelized::m_read_mhd_f32_atpos(std::string txt, i32 pos) {
    f32 res;
    txt = txt.substr(txt.find("=")+2);
    if (pos==0) {
        txt = txt.substr(0, txt.find(" "));
    }
    if (pos==1) {
        txt = txt.substr(txt.find(" ")+1);
        txt = txt.substr(0, txt.find(" "));
    }
    if (pos==2) {
        txt = txt.substr(txt.find(" ")+1);
        txt = txt.substr(txt.find(" ")+1);
    }
    std::stringstream(txt) >> res;
    return res;
}

// Convert range data into material ID
void Voxelized::m_define_materials_from_range(ui16 *raw_data, std::string range_name) {

    ui16 start, stop;
    std::string mat_name, line;
    ui32 i;
    ui16 val;
    ui32 mat_index = 0;
    
    // Data allocation
    data = (f32*)malloc(number_of_voxels * sizeof(f32));

    // Read range file
    std::ifstream file(range_name.c_str());
    if(!file) {
        printf("Error, file %s not found \n", range_name.c_str());
        exit_simulation();
    }
    while (file) {
        m_skip_comment(file);
        std::getline(file, line);

        if (file) {
            start = m_read_start_range(line);
            stop  = m_read_stop_range(line);
            mat_name = m_read_mat_range(line);
            list_of_materials.push_back(mat_name);            

            // build labeled phantom according range data
            i=0; while (i < number_of_voxels) {
                val = raw_data[i];
                if ((val==start && val==stop) || (val>=start && val<stop)) {
                    data[i] = mat_index;
                }
                ++i;
            } // over the volume

        } // new material range
        ++mat_index;

    } // read file

}

// Convert range data into material ID
void Voxelized::m_define_materials_from_range(f32 *raw_data, std::string range_name) {

    f32 start, stop;
    std::string mat_name, line;
    ui32 i;
    f32 val;
    ui32 mat_index = 0;

    // Data allocation
    data = (f32*)malloc(number_of_voxels * sizeof(f32));

    // Read range file
    std::ifstream file(range_name.c_str());
    if(!file) {
        printf("Error, file %s not found \n", range_name.c_str());
        exit_simulation();
    }
    while (file) {
        m_skip_comment(file);
        std::getline(file, line);

        if (file) {
            start = m_read_start_range(line);
            stop  = m_read_stop_range(line);
            mat_name = m_read_mat_range(line);
            list_of_materials.push_back(mat_name);
            //printf("IND %i MAT %s \n", mat_index, mat_name.c_str());

            // build labeled phantom according range data
            i=0; while (i < number_of_voxels) {
                val = raw_data[i];
                if ((val==start && val==stop) || (val>=start && val<stop)) {
                    data[i] = mat_index;
                }
                ++i;
            } // over the volume

        } // new material range
        ++mat_index;

    } // read file

}

///:: Main functions

// Load phantom from binary data (f32)
void Voxelized::load_from_raw(std::string volume_name, std::string range_name,
                              i32 nx, i32 ny, i32 nz, f32 sx, f32 sy, f32 sz) {

    /////////////// First read the raw data from the phantom ////////////////////////

    number_of_voxels = nx*ny*nz;
    nb_vox_x = nx;
    nb_vox_y = ny;
    nb_vox_z = nz;
    spacing_x = sx;
    spacing_y = sy;
    spacing_z = sz;
    mem_size = sizeof(f32) * number_of_voxels;

    FILE *pfile = fopen(volume_name.c_str(), "rb");
    if (!pfile) {
        printf("Error when loading raw data file: %s\n", volume_name.c_str());
        exit_simulation();
    }

    f32 *raw_data = (f32*)malloc(mem_size);
    fread(raw_data, sizeof(f32), number_of_voxels, pfile);

    fclose(pfile);

    /////////////// Then, convert the raw data into material id //////////////////////

    m_define_materials_from_range(raw_data, range_name);

    // Free memory
    free(raw_data);

    ///////////// Define a bounding box for this phantom //////////////////////////////

    f32 h_lengthx = nb_vox_x * spacing_x * 0.5f;
    f32 h_lengthy = nb_vox_y * spacing_y * 0.5f;
    f32 h_lengthz = nb_vox_z * spacing_z * 0.5f;

    xmin = -h_lengthx; xmax = h_lengthx;
    ymin = -h_lengthy; ymax = h_lengthy;
    zmin = -h_lengthz; zmax = h_lengthz;

    offset_x = h_lengthx;
    offset_y = h_lengthy;
    offset_z = h_lengthz;

}

// Load phantom from mhd file (only f32 data)
void Voxelized::load_from_mhd(std::string volume_name, std::string range_name) {

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
        m_skip_comment(file);
        std::getline(file, line);

        if (file) {
            key = m_read_mhd_key(line);
            if (key=="ObjectType")              ObjectType = m_read_mhd_string_arg(line);
            if (key=="NDims")                   NDims = m_read_mhd_int(line);
            if (key=="BinaryData")              BinaryData = m_read_mhd_string_arg(line);
            if (key=="BinaryDataByteOrderMSB")  BinaryDataByteOrderMSB = m_read_mhd_string_arg(line);
            if (key=="CompressedData")          CompressedData = m_read_mhd_string_arg(line);
            //if (key=="TransformMatrix") printf("Matrix\n");
            if (key=="Offset")                  {
                                                ox = m_read_mhd_f32_atpos(line, 0);
                                                oy = m_read_mhd_f32_atpos(line, 1);
                                                oz = m_read_mhd_f32_atpos(line, 2);
            }
            //if (key=="CenterOfRotation") printf("CoR\n");
            if (key=="ElementSpacing") {
                                                sx = m_read_mhd_f32_atpos(line, 0);
                                                sy = m_read_mhd_f32_atpos(line, 1);
                                                sz = m_read_mhd_f32_atpos(line, 2);
            }
            if (key=="DimSize") {
                                                nx = m_read_mhd_int_atpos(line, 0);
                                                ny = m_read_mhd_int_atpos(line, 1);
                                                nz = m_read_mhd_int_atpos(line, 2);
            }

            //if (key=="AnatomicalOrientation") printf("Anato\n");
            if (key=="ElementType")             ElementType = m_read_mhd_string_arg(line);
            if (key=="ElementDataFile")         ElementDataFile = m_read_mhd_string_arg(line);
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

    number_of_voxels = nx*ny*nz;
    nb_vox_x = nx;
    nb_vox_y = ny;
    nb_vox_z = nz;
    spacing_x = sx;
    spacing_y = sy;
    spacing_z = sz;   
    
    if(ElementType == "MET_FLOAT") {
      mem_size = sizeof(f32) * number_of_voxels;

      f32 *raw_data = (f32*)malloc(mem_size);
      fread(raw_data, sizeof(f32), number_of_voxels, pfile);
      fclose(pfile);
      
      /////////////// Then, convert the raw data into material id //////////////////////

      m_define_materials_from_range(raw_data, range_name);

      // Free memory
      free(raw_data);
    }
    
    if(ElementType == "MET_USHORT") {
      mem_size = sizeof(ui16) * number_of_voxels;

      ui16 *raw_data = (ui16*)malloc(mem_size);
      fread(raw_data, sizeof(ui16), number_of_voxels, pfile);
      fclose(pfile);
      /////////////// Then, convert the raw data into material id //////////////////////
      m_define_materials_from_range(raw_data, range_name);

      // Free memory
      free(raw_data);
    }  

    ///////////// Define a bounding box for this phantom //////////////////////////////

    f32 h_lengthx = nb_vox_x * spacing_x * 0.5f;
    f32 h_lengthy = nb_vox_y * spacing_y * 0.5f;
    f32 h_lengthz = nb_vox_z * spacing_z * 0.5f;

    // If the offset is not defined, chose the volume center
    if (ox == -1 || oy == -1 || oz == -1) {
        offset_x = h_lengthx;
        offset_y = h_lengthy;
        offset_z = h_lengthz;

        xmin = -h_lengthx; xmax = h_lengthx;
        ymin = -h_lengthy; ymax = h_lengthy;
        zmin = -h_lengthz; zmax = h_lengthz;

    } else {
        offset_x = ox;
        offset_y = oy;
        offset_z = oz;

        xmin = -ox; xmax = xmin + (nb_vox_x * spacing_x);
        ymin = -oy; ymax = ymin + (nb_vox_y * spacing_y);
        zmin = -oz; zmax = zmin + (nb_vox_z * spacing_z);
    }

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






















