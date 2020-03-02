
/*!
  \file GGEMSSolidPhantom.cc

  \brief GGEMS class for solid phantom informations

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 2, 2020
*/

#include "GGEMS/geometries/GGEMSSolidPhantom.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolidPhantom::GGEMSSolidPhantom(void)
{
  GGcout("GGEMSSolidPhantom", "GGEMSSolidPhantom", 3) << "Allocation of GGEMSSolidPhantom..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolidPhantom::~GGEMSSolidPhantom(void)
{
  GGcout("GGEMSSolidPhantom", "~GGEMSSolidPhantom", 3) << "Deallocation of GGEMSSolidPhantom..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void GGEMSSolidPhantom::LoadPhantomImage(std::string const& phantom_filename)
{
  GGcout("GGEMSSolidPhantom", "LoadPhantomImage", 3) << "Loading image phantom from mhd file..." << GGendl;

  // Read MHD input file
  GGEMSMHDImage mhd_input_phantom;
  mhd_input_phantom.Read(phantom_filename);

/*  std::string line, key;
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
    }*/
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void GGEMSSolidPhantom::LoadRangeToMaterialData(std::string const& range_data_filename)
{
  GGcout("GGEMSSolidPhantom", "LoadRangeToMaterialData", 3) << "Loading range to material data and label..." << GGendl;
}
