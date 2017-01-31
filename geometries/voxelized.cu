// GGEMS Copyright (C) 2017

/*!
 * \file voxelized.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.2
 * \date 18 novembre 2015
 *
 * v0.2: JB - Change all structs and remove CPU exec
 *
 */

#ifndef VOXELIZED_CU
#define VOXELIZED_CU

#include "voxelized.cuh"

VoxelizedPhantom::VoxelizedPhantom() {
    h_volume = (VoxVolumeData<ui16>*)malloc( sizeof(VoxVolumeData<ui16>) );

    h_volume->xmin = 0.0f;
    h_volume->xmax = 0.0f;
    h_volume->ymin = 0.0f;
    h_volume->ymax = 0.0f;
    h_volume->zmin = 0.0f;
    h_volume->zmax = 0.0f;

    // Init pointer
    h_volume->values = nullptr;
}

///:: Private

// Check
bool VoxelizedPhantom::m_check_mandatory() {
    if (h_volume->number_of_voxels == 0) return false;
    else return true;
}

// Copy the phantom to the GPU
void VoxelizedPhantom::m_copy_phantom_cpu2gpu() {

    ui32 n = h_volume->number_of_voxels;

    /// First, struct allocation
    HANDLE_ERROR( cudaMalloc( (void**) &d_volume, sizeof( VoxVolumeData<ui16> ) ) );

    /// Device pointers allocation
    ui16 *values;
    HANDLE_ERROR( cudaMalloc((void**) &values, n*sizeof(ui16)) );

    /// Copy host data to device
    HANDLE_ERROR( cudaMemcpy( values, h_volume->values,
                              n*sizeof(ui16), cudaMemcpyHostToDevice ) );

    /// Bind data to the struct
    HANDLE_ERROR( cudaMemcpy( &(d_volume->values), &values,
                              sizeof(d_volume->values), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_volume->nb_vox_x), &(h_volume->nb_vox_x),
                                sizeof(d_volume->nb_vox_x), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_volume->nb_vox_y), &(h_volume->nb_vox_y),
                                sizeof(d_volume->nb_vox_y), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_volume->nb_vox_z), &(h_volume->nb_vox_z),
                                sizeof(d_volume->nb_vox_z), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_volume->number_of_voxels), &(h_volume->number_of_voxels),
                                sizeof(d_volume->number_of_voxels), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_volume->spacing_x), &(h_volume->spacing_x),
                                sizeof(d_volume->spacing_x), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_volume->spacing_y), &(h_volume->spacing_y),
                                sizeof(d_volume->spacing_y), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_volume->spacing_z), &(h_volume->spacing_z),
                                sizeof(d_volume->spacing_z), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_volume->off_x), &(h_volume->off_x),
                                sizeof(d_volume->off_x), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_volume->off_y), &(h_volume->off_y),
                                sizeof(d_volume->off_y), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_volume->off_z), &(h_volume->off_z),
                                sizeof(d_volume->off_z), cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( &(d_volume->xmin), &(h_volume->xmin),
                                sizeof(d_volume->xmin), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_volume->xmax), &(h_volume->xmax),
                                sizeof(d_volume->xmax), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_volume->ymin), &(h_volume->ymin),
                                sizeof(d_volume->ymin), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_volume->ymax), &(h_volume->ymax),
                                sizeof(d_volume->ymax), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_volume->zmin), &(h_volume->zmin),
                                sizeof(d_volume->zmin), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( &(d_volume->zmax), &(h_volume->zmax),
                                sizeof(d_volume->zmax), cudaMemcpyHostToDevice ) );

}

// Convert range data into material ID
template<typename Type>
void VoxelizedPhantom::m_define_materials_from_range(Type *raw_data, std::string range_name) {

    f32 start, stop;
    std::string mat_name, line;
    ui32 i;
    f32 val;
    ui16 mat_index = 0;
    
    // Data allocation
    h_volume->values = (ui16*)malloc(h_volume->number_of_voxels * sizeof(ui16));

    // Read range file
    std::ifstream file(range_name.c_str());
    if(!file) {
        printf("Error, file %s not found \n", range_name.c_str());
        exit_simulation();
    }
    f32 min_val =  FLT_MAX;
    f32 max_val = -FLT_MAX;
    while (file) {
        m_txt_reader.skip_comment(file);
        std::getline(file, line);

        if (file) {
            start = m_txt_reader.read_start_range(line);
            stop  = m_txt_reader.read_stop_range(line);
            mat_name = m_txt_reader.read_mat_range(line);
            list_of_materials.push_back(mat_name);            

            // Store min and max values
            if ( start < min_val ) min_val = start;
            if ( stop  > max_val ) max_val = stop;

            // build labeled phantom according range data
            i=0; while (i < h_volume->number_of_voxels) {
                val = (f32) raw_data[i];
                if ((val==start && val==stop) || (val>=start && val<stop)) {
                    h_volume->values[i] = mat_index;
                }
                ++i;
            } // over the volume

        } // new material range
        ++mat_index;

    } // read file

    // Check if everything was converted
    i=0; while (i < h_volume->number_of_voxels) {
        val = (f32) raw_data[i];

        if ( val < min_val )
        {
            GGcerr << "A phantom raw value is out of the material range, phantom: "
                   << val << " min range " << min_val << GGendl;
            exit_simulation();
        }

        if ( val > max_val )
        {
            GGcerr << "A phantom raw value is out of the material range, phantom: "
                   << val << " max range " << max_val << GGendl;
            exit_simulation();
        }

        ++i;
    } // over the volume

}

///:: Main functions

// Init
void VoxelizedPhantom::initialize() {
    // Check if everything was set properly
    if ( !m_check_mandatory() ) {
        print_error("Missing parameters for the voxelized phantom!");
        exit_simulation();
    }

    m_copy_phantom_cpu2gpu();
}

// Load phantom from binary data (f32)
void VoxelizedPhantom::load_from_raw(std::string volume_name, std::string range_name,
                              i32 nx, i32 ny, i32 nz, f32 sx, f32 sy, f32 sz) {

    /////////////// First read the raw data from the phantom ////////////////////////

    h_volume->number_of_voxels = nx*ny*nz;
    h_volume->nb_vox_x = nx;
    h_volume->nb_vox_y = ny;
    h_volume->nb_vox_z = nz;
    h_volume->spacing_x = sx;
    h_volume->spacing_y = sy;
    h_volume->spacing_z = sz;
    ui32 mem_size = sizeof(f32) * h_volume->number_of_voxels;

    FILE *pfile = fopen(volume_name.c_str(), "rb");
    if (!pfile) {
        printf("Error when loading raw data file: %s\n", volume_name.c_str());
        exit_simulation();
    }

    f32 *raw_data = (f32*)malloc(mem_size);
    fread(raw_data, sizeof(f32), h_volume->number_of_voxels, pfile);

    fclose(pfile);

    /////////////// Then, convert the raw data into material id //////////////////////

    m_define_materials_from_range(raw_data, range_name);

    // Free memory
    free(raw_data);

    ///////////// Define a bounding box for this phantom //////////////////////////////

    f32 h_lengthx = h_volume->nb_vox_x * h_volume->spacing_x * 0.5f;
    f32 h_lengthy = h_volume->nb_vox_y * h_volume->spacing_y * 0.5f;
    f32 h_lengthz = h_volume->nb_vox_z * h_volume->spacing_z * 0.5f;

    h_volume->xmin = -h_lengthx; h_volume->xmax = h_lengthx;
    h_volume->ymin = -h_lengthy; h_volume->ymax = h_lengthy;
    h_volume->zmin = -h_lengthz; h_volume->zmax = h_lengthz;

    h_volume->off_x = h_lengthx;
    h_volume->off_y = h_lengthy;
    h_volume->off_z = h_lengthz;

}

// Load phantom from mhd file (only f32 data)
void VoxelizedPhantom::load_from_mhd(std::string volume_name, std::string range_name) {

    /////////////// First read the MHD file //////////////////////

    std::string line, key;
    i32 nx=-1, ny=-1, nz=-1;
    f32 sx=0, sy=0, sz=0;
    f32 ox=0, oy=0, oz=0;

    bool flag_offset = false;

    // Watchdog
    std::string ObjectType="", BinaryData="", BinaryDataByteOrderMSB="", CompressedData="",
                ElementType="", ElementDataFile="";
    i32 NDims=0;

    // Read file
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
                                                flag_offset = true;
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
    if (ElementType != "MET_FLOAT" && ElementType != "MET_SHORT" && ElementType != "MET_USHORT" &&
        ElementType != "MET_UCHAR" && ElementType != "MET_UINT") {
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

    // Reative path?
    if (!pfile) {
        std::string nameWithRelativePath = volume_name;
        i32 lastindex = nameWithRelativePath.find_last_of("/");
        nameWithRelativePath = nameWithRelativePath.substr(0, lastindex);       
        nameWithRelativePath += ( "/" + ElementDataFile );

        pfile = fopen(nameWithRelativePath.c_str(), "rb");
        if (!pfile) {
            printf("Error when loading mhd file: %s\n", ElementDataFile.c_str());

            exit_simulation();
        }
    }

    h_volume->number_of_voxels = nx*ny*nz;
    h_volume->nb_vox_x = nx;
    h_volume->nb_vox_y = ny;
    h_volume->nb_vox_z = nz;
    h_volume->spacing_x = sx;
    h_volume->spacing_y = sy;
    h_volume->spacing_z = sz;
    
    if(ElementType == "MET_FLOAT") {
      ui32 mem_size = sizeof(f32) * h_volume->number_of_voxels;

      f32 *raw_data = (f32*)malloc(mem_size);
      fread(raw_data, sizeof(f32), h_volume->number_of_voxels, pfile);
      fclose(pfile);
      
      /////////////// Then, convert the raw data into material id //////////////////////

      m_define_materials_from_range(raw_data, range_name);

      // Free memory
      free(raw_data);
    }
    
    if(ElementType == "MET_USHORT") {
      ui32 mem_size = sizeof(ui16) * h_volume->number_of_voxels;

      ui16 *raw_data = (ui16*)malloc(mem_size);
      fread(raw_data, sizeof(ui16), h_volume->number_of_voxels, pfile);
      fclose(pfile);
      /////////////// Then, convert the raw data into material id //////////////////////
      m_define_materials_from_range(raw_data, range_name);

      // Free memory
      free(raw_data);
    }  

    if(ElementType == "MET_SHORT") {
      ui32 mem_size = sizeof(i16) * h_volume->number_of_voxels;

      i16 *raw_data = (i16*)malloc(mem_size);
      fread(raw_data, sizeof(i16), h_volume->number_of_voxels, pfile);
      fclose(pfile);
      /////////////// Then, convert the raw data into material id //////////////////////
      m_define_materials_from_range(raw_data, range_name);

      // Free memory
      free(raw_data);
    }

    if(ElementType == "MET_UCHAR") {
      ui32 mem_size = sizeof(ui8) * h_volume->number_of_voxels;

      ui8 *raw_data = (ui8*)malloc(mem_size);
      fread(raw_data, sizeof(ui8), h_volume->number_of_voxels, pfile);
      fclose(pfile);
      /////////////// Then, convert the raw data into material id //////////////////////
      m_define_materials_from_range(raw_data, range_name);

      // Free memory
      free(raw_data);
    }

    if(ElementType == "MET_UINT") {
      ui32 mem_size = sizeof(ui32) * h_volume->number_of_voxels;

      ui32 *raw_data = (ui32*)malloc(mem_size);
      fread(raw_data, sizeof(ui32), h_volume->number_of_voxels, pfile);
      fclose(pfile);
      /////////////// Then, convert the raw data into material id //////////////////////
      m_define_materials_from_range(raw_data, range_name);

      // Free memory
      free(raw_data);
    }

    ///////////// Define a bounding box for this phantom //////////////////////////////

    f32 h_lengthx = h_volume->nb_vox_x * h_volume->spacing_x * 0.5f;
    f32 h_lengthy = h_volume->nb_vox_y * h_volume->spacing_y * 0.5f;
    f32 h_lengthz = h_volume->nb_vox_z * h_volume->spacing_z * 0.5f;

    // If the offset is not defined, chose the volume center
    if ( !flag_offset ) {
        h_volume->off_x = h_lengthx;
        h_volume->off_y = h_lengthy;
        h_volume->off_z = h_lengthz;

        h_volume->xmin = -h_lengthx; h_volume->xmax = h_lengthx;
        h_volume->ymin = -h_lengthy; h_volume->ymax = h_lengthy;
        h_volume->zmin = -h_lengthz; h_volume->zmax = h_lengthz;

    } else {
        h_volume->off_x = ox;
        h_volume->off_y = oy;
        h_volume->off_z = oz;

        h_volume->xmin = -ox; h_volume->xmax = h_volume->xmin + (h_volume->nb_vox_x * h_volume->spacing_x);
        h_volume->ymin = -oy; h_volume->ymax = h_volume->ymin + (h_volume->nb_vox_y * h_volume->spacing_y);
        h_volume->zmin = -oz; h_volume->zmax = h_volume->zmin + (h_volume->nb_vox_z * h_volume->spacing_z);
    }

}

void VoxelizedPhantom::set_offset(f32 x, f32 y, f32 z) {
    h_volume->off_x = x;
    h_volume->off_y = y;
    h_volume->off_z = z;

    h_volume->xmin = -x; h_volume->xmax = h_volume->xmin + (h_volume->nb_vox_x * h_volume->spacing_x);
    h_volume->ymin = -y; h_volume->ymax = h_volume->ymin + (h_volume->nb_vox_y * h_volume->spacing_y);
    h_volume->zmin = -z; h_volume->zmax = h_volume->zmin + (h_volume->nb_vox_z * h_volume->spacing_z);
}

#endif






















