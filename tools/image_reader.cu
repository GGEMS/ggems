#ifndef SAVE_DATA_CU
#define SAVE_DATA_CU

#include "image_reader.cuh"

/*

using namespace std;


void ImageReader::create_directory(std::string dirname)
{
    std::string command = "mkdir -p " + dirname;
    system(command.c_str());
}

void ImageReader::create_directory_tree(std::string dirname)
{
    std::string tmp = dirname;;
    std::string directory = "";
    
    for(;;)
    {
        if(tmp.substr (0, tmp.find_first_of ( "/" ) +1) == "") break;
        directory += tmp.substr (0, tmp.find_first_of ( "/" ) +1);
        tmp = tmp.substr (tmp.find_first_of ( "/" ) +1 );
        
        create_directory(directory);
    }
}       

void ImageReader::record3Dimage ( string histname,  f32 *data, f32xyz offset, f32xyz spacing, ui32xyz size, bool sparse_compression )
{

//         printf("Image Parameters : %d %g %g %d %g %g %d %g %g \n",
//                                   size.x,  offset.x, offset.x+size.x*spacing.x,
//                                   size.y,  offset.y, offset.y+size.y*spacing.y,
//                                   size.z,  offset.z, offset.z+size.z*spacing.z );

   GGcout << "Write image " << histname << " ... " << GGendl;
    // Check format
    string format = get_format ( histname );
    histname = get_filename_without_format ( histname );

    create_directory_tree(histname);

    if ( format == "mhd" )
    {
        string pathnamemhd = histname + ".mhd";
        string pathnameraw = histname + ".raw";

//         cout<<"Save file : "<<pathnamemhd << endl;

        // MHD file
        std::ofstream myfile;
        myfile.open ( pathnamemhd.c_str() );
        myfile << "ObjectType = Image\n";
        myfile << "NDims = 3\n";
        myfile << "BinaryData = True\n";
        myfile << "BinaryDataByteOrderMSB = False\n";

        myfile << "CompressedData = "<<  ( ( sparse_compression ) ? "COO" : "False" )  <<"\n";

        myfile << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n";
        myfile << "Offset = "<<offset.x<<" "<<offset.y<<" "<<offset.z<<"\n";
        myfile << "CenterOfRotation = 0 0 0\n";
        myfile << "ElementSpacing = "<<spacing.x<<" "<<spacing.y<<" "<<spacing.z<<"\n";
        myfile << "DimSize = "<<size.x<<" "<<size.y<<" "<<size.z<<"\n";
        myfile << "AnatomicalOrientation = ???\n";
        myfile << "ElementType = MET_FLOAT\n";
        myfile << "ElementDataFile = "<<remove_path ( pathnameraw ).c_str() <<"\n";

        myfile.close();



        // RAW File
        FILE *pFile_mhd;
        pFile_mhd = fopen ( pathnameraw.c_str(),"wb" );

        // Compressed data in COO format
        if ( sparse_compression )
        {

            // First get the number of non-zero
            ui32 index = 0;
            ui32 ct_nz = 0;
            while ( index < size.x*size.y*size.z )
            {
                if ( data[index] != 0.0 ) ++ct_nz;
                ++index;
            }

            // Write the previous value as the first binary element
            fwrite ( &ct_nz, sizeof ( ui32 ), 1, pFile_mhd );

            // Some vars
            ui16 ix, iy, iz;
//             unsigned int jump;
//             jump = size.x*size.y;
            index = 0;

            // Loop over every element
            iz = 0;
            while ( iz<size.z )
            {
                iy = 0;
                while ( iy<size.y )
                {
                    ix = 0;
                    while ( ix<size.x )
                    {

                        // Export only non-zero value in COO format
                        if ( data[index] != 0.0 )
                        {
                            // xyz coordinate
                            fwrite ( &ix, sizeof ( ui16 ), 1, pFile_mhd );
                            fwrite ( &iy, sizeof ( ui16 ), 1, pFile_mhd );
                            fwrite ( &iz, sizeof ( ui16 ), 1, pFile_mhd );
                            // Then the corresponding value
                            float val = data[index];
                            fwrite ( &val, sizeof ( f32 ), 1, pFile_mhd );
                        }

                        ++ix;
                        ++index;
                    } // ix
                    ++iy;
                } // iy
                ++iz;
            } // iz

        }
        else
        {
            // Export uncompressed raw data
            ui32 i=0;
            while ( i<size.x*size.y*size.z )
            {
                f32 val = data[i];
                fwrite ( &val, sizeof ( f32 ), 1, pFile_mhd );
                ++i;
            }
        }

        fclose ( pFile_mhd );


    } // Fin mhd
    else  if ( ( format == "ASCII" ) || ( format == "txt" ) )
    {
        string pathname = histname + ".txt";

//         std::cout<<"saving "<<pathname.c_str() <<std::endl;
        std::ofstream ofs ( pathname.c_str(),  std::ofstream::out );

        int xdim = size.x;
        int ydim = size.y;
        int zdim = size.z;

        for ( int i=0; i<xdim; ++i )
        {
            for ( int j=0; j<ydim; ++j )
            {
                for ( int k=0; k<zdim; ++k )
                {
                    if ( data[i + j*xdim + k*xdim*ydim]!=0. )
                        ofs << i <<"\t"<<j<<"\t"<<k<<"\t"<<data[i + j*xdim + k*xdim*ydim]<<std::endl;
                }
            }
        }
    }

    else
    {

        GGcout << " Unknown format ... " << GGendl;

    }

}

#ifndef SINGLE_PRECISION
void ImageReader::record3Dimage ( string histname,  f64 *data, f32xyz offset, f32xyz spacing, ui32xyz size, bool sparse_compression )
{

//         printf("Image Parameters : %d %g %g %d %g %g %d %g %g \n",
//                                   size.x,  offset.x, offset.x+size.x*spacing.x,
//                                   size.y,  offset.y, offset.y+size.y*spacing.y,
//                                   size.z,  offset.z, offset.z+size.z*spacing.z );

   GGcout << "Write image " << histname << " ... " << GGendl;
    // Check format
    string format = get_format ( histname );
    histname = get_filename_without_format ( histname );
 
    create_directory_tree(histname);
    
    if ( format == "mhd" )
    {
        string pathnamemhd = histname + ".mhd";
        string pathnameraw = histname + ".raw";

//         cout<<"Save file : "<<pathnamemhd << endl;

        // MHD file
        std::ofstream myfile;
        myfile.open ( pathnamemhd.c_str() );
        myfile << "ObjectType = Image\n";
        myfile << "NDims = 3\n";
        myfile << "BinaryData = True\n";
        myfile << "BinaryDataByteOrderMSB = False\n";

        myfile << "CompressedData = "<<  ( ( sparse_compression ) ? "COO" : "False" )  <<"\n";

        myfile << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n";
        myfile << "Offset = "<<offset.x<<" "<<offset.y<<" "<<offset.z<<"\n";
        myfile << "CenterOfRotation = 0 0 0\n";
        myfile << "ElementSpacing = "<<spacing.x<<" "<<spacing.y<<" "<<spacing.z<<"\n";
        myfile << "DimSize = "<<size.x<<" "<<size.y<<" "<<size.z<<"\n";
        myfile << "AnatomicalOrientation = ???\n";
        myfile << "ElementType = MET_FLOAT\n";
        myfile << "ElementDataFile = "<<remove_path ( pathnameraw ).c_str() <<"\n";

        myfile.close();



        // RAW File
        FILE *pFile_mhd;
        pFile_mhd = fopen ( pathnameraw.c_str(),"wb" );

        // Compressed data in COO format
        if ( sparse_compression )
        {

            // First get the number of non-zero
            ui32 index = 0;
            ui32 ct_nz = 0;
            while ( index < size.x*size.y*size.z )
            {
                if ( data[index] != 0.0 ) ++ct_nz;
                ++index;
            }

            // Write the previous value as the first binary element
            fwrite ( &ct_nz, sizeof ( ui32 ), 1, pFile_mhd );

            // Some vars
            ui16 ix, iy, iz;
//             unsigned int jump;
//             jump = size.x*size.y;
            index = 0;

            // Loop over every element
            iz = 0;
            while ( iz<size.z )
            {
                iy = 0;
                while ( iy<size.y )
                {
                    ix = 0;
                    while ( ix<size.x )
                    {

                        // Export only non-zero value in COO format
                        if ( data[index] != 0.0 )
                        {
                            // xyz coordinate
                            fwrite ( &ix, sizeof ( ui16 ), 1, pFile_mhd );
                            fwrite ( &iy, sizeof ( ui16 ), 1, pFile_mhd );
                            fwrite ( &iz, sizeof ( ui16 ), 1, pFile_mhd );
                            // Then the corresponding value
                            float val = data[index];
                            fwrite ( &val, sizeof ( f32 ), 1, pFile_mhd );
                        }

                        ++ix;
                        ++index;
                    } // ix
                    ++iy;
                } // iy
                ++iz;
            } // iz

        }
        else
        {
            // Export uncompressed raw data
            ui32 i=0;
            while ( i<size.x*size.y*size.z )
            {
                f32 val = data[i];
                fwrite ( &val, sizeof ( f32 ), 1, pFile_mhd );
                ++i;
            }
        }

        fclose ( pFile_mhd );


    } // Fin mhd
    else  if ( ( format == "ASCII" ) || ( format == "txt" ) )
    {
        string pathname = histname + ".txt";

//         std::cout<<"saving "<<pathname.c_str() <<std::endl;
        std::ofstream ofs ( pathname.c_str(),  std::ofstream::out );

        int xdim = size.x;
        int ydim = size.y;
        int zdim = size.z;

        for ( int i=0; i<xdim; ++i )
        {
            for ( int j=0; j<ydim; ++j )
            {
                for ( int k=0; k<zdim; ++k )
                {
                    if ( data[i + j*xdim + k*xdim*ydim]!=0. )
                        ofs << i <<"\t"<<j<<"\t"<<k<<"\t"<<data[i + j*xdim + k*xdim*ydim]<<std::endl;
                }
            }
        }
    }

    else
    {

        GGcout << " Unknown format ... " << GGendl;

    }

}
#endif

void ImageReader::record3Dimage ( string histname,  ui32 *data, f32xyz offset, f32xyz spacing, ui32xyz size, bool sparse_compression )
{




    // Check format
    string format = get_format ( histname );
    histname = get_filename_without_format ( histname );

    create_directory_tree(histname);
    
    if ( format == "mhd" )
    {
        string pathnamemhd = histname + ".mhd";
        string pathnameraw = histname + ".raw";

        GGcout << "Write image " << histname << " ... " << GGendl;

        // MHD file
        std::ofstream myfile;
        myfile.open ( pathnamemhd.c_str() );
        myfile << "ObjectType = Image\n";
        myfile << "NDims = 3\n";
        myfile << "BinaryData = True\n";
        myfile << "BinaryDataByteOrderMSB = False\n";

        myfile << "CompressedData = "<<  ( ( sparse_compression ) ? "COO" : "False" )  <<"\n";

        myfile << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n";
        myfile << "Offset = "<<offset.x<<" "<<offset.y<<" "<<offset.z<<"\n";
        myfile << "CenterOfRotation = 0 0 0\n";
        myfile << "ElementSpacing = "<<spacing.x<<" "<<spacing.y<<" "<<spacing.z<<"\n";
        myfile << "DimSize = "<<size.x<<" "<<size.y<<" "<<size.z<<"\n";
        myfile << "AnatomicalOrientation = ???\n";
        myfile << "ElementType = MET_UINT\n";
        myfile << "ElementDataFile = "<<remove_path ( pathnameraw ).c_str() <<"\n";

        myfile.close();

        // RAW File
        FILE *pFile_mhd;
        pFile_mhd = fopen ( pathnameraw.c_str(),"wb" );

        // Compressed data in COO format
        if ( sparse_compression )
        {

            // First get the number of non-zero
            ui32 index = 0;
            ui32 ct_nz = 0;
            while ( index < size.x*size.y*size.z )
            {
                if ( data[index] != 0.0 ) ++ct_nz;
                ++index;
            }

            // Write the previous value as the first binary element
            fwrite ( &ct_nz, sizeof ( ui32 ), 1, pFile_mhd );

            // Some vars
            ui16 ix, iy, iz;
//             unsigned int jump;
//             jump = size.x*size.y;
            index = 0;

            // Loop over every element
            iz = 0;
            while ( iz<size.z )
            {
                iy = 0;
                while ( iy<size.y )
                {
                    ix = 0;
                    while ( ix<size.x )
                    {

                        // Export only non-zero value in COO format
                        if ( data[index] != 0.0 )
                        {
                            // xyz coordinate
                            fwrite ( &ix, sizeof ( ui16 ), 1, pFile_mhd );
                            fwrite ( &iy, sizeof ( ui16 ), 1, pFile_mhd );
                            fwrite ( &iz, sizeof ( ui16 ), 1, pFile_mhd );
                            // Then the corresponding value
                            float val = data[index];
                            fwrite ( &val, sizeof ( ui32 ), 1, pFile_mhd );
                        }

                        ++ix;
                        ++index;
                    } // ix
                    ++iy;
                } // iy
                ++iz;
            } // iz

        }
        else
        {
            // Export uncompressed raw data
            int i=0;
            while ( i<size.x*size.y*size.z )
            {
                ui32 val = data[i];            
                fwrite ( &val, sizeof ( ui32 ), 1, pFile_mhd );
                ++i;
            }
        }

        fclose ( pFile_mhd );


    } // Fin mhd
    else  if ( ( format == "ASCII" ) || ( format == "txt" ) )
    {
        string pathname = histname + ".txt";

        std::cout<<"saving "<<pathname.c_str() <<std::endl;
        std::ofstream ofs ( pathname.c_str(),  std::ofstream::out );

        int xdim = size.x;
        int ydim = size.y;
        int zdim = size.z;

        for ( int i=0; i<xdim; ++i )
        {
            for ( int j=0; j<ydim; ++j )
            {
                for ( int k=0; k<zdim; ++k )
                {
                    if ( data[i + j*xdim + k*xdim*ydim]!=0. )
                        ofs << i <<"\t"<<j<<"\t"<<k<<"\t"<<data[i + j*xdim + k*xdim*ydim]<<std::endl;
                }
            }
        }
    }

    else
    {

        cout << " Unknown format ... "<<endl;

    }

}

void ImageReader::record3Dimage (string histname,  ui16 *data, f32xyz offset, f32xyz spacing, ui32xyz size, bool sparse_compression )
{



    // Check format
    string format = get_format ( histname );
    histname = get_filename_without_format ( histname );

    create_directory_tree(histname);
    
    if ( format == "mhd" )
    {
        string pathnamemhd = histname + ".mhd";
        string pathnameraw = histname + ".raw";

        GGcout << "Write image " << histname << " ... " << GGendl;

        // MHD file
        std::ofstream myfile;
        myfile.open ( pathnamemhd.c_str() );
        myfile << "ObjectType = Image\n";
        myfile << "NDims = 3\n";
        myfile << "BinaryData = True\n";
        myfile << "BinaryDataByteOrderMSB = False\n";

        myfile << "CompressedData = "<<  ( ( sparse_compression ) ? "COO" : "False" )  <<"\n";

        myfile << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n";
        myfile << "Offset = "<<offset.x<<" "<<offset.y<<" "<<offset.z<<"\n";
        myfile << "CenterOfRotation = 0 0 0\n";
        myfile << "ElementSpacing = "<<spacing.x<<" "<<spacing.y<<" "<<spacing.z<<"\n";
        myfile << "DimSize = "<<size.x<<" "<<size.y<<" "<<size.z<<"\n";
        myfile << "AnatomicalOrientation = ???\n";
        myfile << "ElementType = MET_USHORT\n";
        myfile << "ElementDataFile = "<<remove_path ( pathnameraw ).c_str() <<"\n";

        myfile.close();

        // RAW File
        FILE *pFile_mhd;
        pFile_mhd = fopen ( pathnameraw.c_str(),"wb" );

        // Compressed data in COO format
        if ( sparse_compression )
        {

            // First get the number of non-zero
            ui32 index = 0;
            ui32 ct_nz = 0;
            while ( index < size.x*size.y*size.z )
            {
                if ( data[index] != 0.0 ) ++ct_nz;
                ++index;
            }

            // Write the previous value as the first binary element
            fwrite ( &ct_nz, sizeof ( ui16 ), 1, pFile_mhd );

            // Some vars
            ui16 ix, iy, iz;
//             unsigned int jump;
//             jump = size.x*size.y;
            index = 0;

            // Loop over every element
            iz = 0;
            while ( iz<size.z )
            {
                iy = 0;
                while ( iy<size.y )
                {
                    ix = 0;
                    while ( ix<size.x )
                    {

                        // Export only non-zero value in COO format
                        if ( data[index] != 0.0 )
                        {
                            // xyz coordinate
                            fwrite ( &ix, sizeof ( ui16 ), 1, pFile_mhd );
                            fwrite ( &iy, sizeof ( ui16 ), 1, pFile_mhd );
                            fwrite ( &iz, sizeof ( ui16 ), 1, pFile_mhd );
                            // Then the corresponding value
                            float val = data[index];
                            fwrite ( &val, sizeof ( ui16 ), 1, pFile_mhd );
                        }

                        ++ix;
                        ++index;
                    } // ix
                    ++iy;
                } // iy
                ++iz;
            } // iz

        }
        else
        {
            // Export uncompressed raw data
            int i=0;
            while ( i<size.x*size.y*size.z )
            {
                ui16 val = data[i];            
                fwrite ( &val, sizeof ( ui16 ), 1, pFile_mhd );
                ++i;
            }
        }

        fclose ( pFile_mhd );


    } // Fin mhd
    else  if ( ( format == "ASCII" ) || ( format == "txt" ) )
    {
        string pathname = histname + ".txt";

        std::cout<<"saving "<<pathname.c_str() <<std::endl;
        std::ofstream ofs ( pathname.c_str(),  std::ofstream::out );

        int xdim = size.x;
        int ydim = size.y;
        int zdim = size.z;

        for ( int i=0; i<xdim; ++i )
        {
            for ( int j=0; j<ydim; ++j )
            {
                for ( int k=0; k<zdim; ++k )
                {
                    if ( data[i + j*xdim + k*xdim*ydim]!=0. )
                        ofs << i <<"\t"<<j<<"\t"<<k<<"\t"<<data[i + j*xdim + k*xdim*ydim]<<std::endl;
                }
            }
        }
    }

    else
    {

        cout << " Unknown format ... " <<endl;

    }

}


// Load mhd image
f32* ImageReader::load_mhd_image( string filename, f32xyz &offset, f32xyz &voxsize, ui32xyz &nbvox )
{
    /////////////// First read the MHD file //////////////////////

    TxtReader m_txt_reader;

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
    std::ifstream file( filename.c_str() );

    if ( !file )
    {
        GGcerr << "Error, file " << filename << " not found " << GGendl;
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
        std::string nameWithRelativePath = filename;
        i32 lastindex = nameWithRelativePath.find_last_of("/");
        nameWithRelativePath = nameWithRelativePath.substr(0, lastindex);
        nameWithRelativePath += ( "/" + ElementDataFile );

        pfile = fopen(nameWithRelativePath.c_str(), "rb");
        if ( !pfile )
        {
            GGcerr << "Error, file " << ElementDataFile << " not found " << GGendl;
            exit_simulation();
        }
    }

    ui32 number_of_voxels = nx*ny*nz;
    ui32 mem_size = sizeof(f32) * number_of_voxels;
    f32 *data = (f32*)malloc(mem_size);

    nbvox.x = nx;
    nbvox.y = ny;
    nbvox.z = nz;

    voxsize.x = sx;
    voxsize.y = sy;
    voxsize.z = sz;

    if ( ElementType == "MET_FLOAT" ) {
        fread(data, sizeof(f32), number_of_voxels, pfile);
        fclose(pfile);
    }

    if(ElementType == "MET_USHORT") {
        mem_size = sizeof(ui16) * number_of_voxels;

        ui16 *raw_data = (ui16*)malloc(mem_size);
        fread(raw_data, sizeof(ui16), number_of_voxels, pfile);
        fclose(pfile);

        // Convert
        ui32 i=0; while ( i < number_of_voxels )
        {
            data[ i ] = (f32) raw_data[ i ];
            ++i;
        }
        // Free memory
        free(raw_data);
    }

    if(ElementType == "MET_SHORT") {
        mem_size = sizeof(i16) * number_of_voxels;

        i16 *raw_data = (i16*)malloc(mem_size);
        fread(raw_data, sizeof(i16), number_of_voxels, pfile);
        fclose(pfile);

        // Convert
        ui32 i=0; while ( i < number_of_voxels )
        {
            data[ i ] = (f32) raw_data[ i ];
            ++i;
        }
        // Free memory
        free(raw_data);
    }

    if(ElementType == "MET_UCHAR") {
        mem_size = sizeof(ui8) * number_of_voxels;

        ui8 *raw_data = (ui8*)malloc(mem_size);
        fread(raw_data, sizeof(ui8), number_of_voxels, pfile);
        fclose(pfile);

        // Convert
        ui32 i=0; while ( i < number_of_voxels )
        {
            data[ i ] = (f32) raw_data[ i ];
            ++i;
        }
        // Free memory
        free(raw_data);
    }

    if(ElementType == "MET_UINT") {
        mem_size = sizeof(ui32) * number_of_voxels;

        ui32 *raw_data = (ui32*)malloc(mem_size);
        fread(raw_data, sizeof(ui32), number_of_voxels, pfile);
        fclose(pfile);

        // Convert
        ui32 i=0; while ( i < number_of_voxels )
        {
            data[ i ] = (f32) raw_data[ i ];
            ++i;
        }
        // Free memory
        free(raw_data);
    }

    f32 h_lengthx = nbvox.x * voxsize.x * 0.5f;
    f32 h_lengthy = nbvox.y * voxsize.y * 0.5f;
    f32 h_lengthz = nbvox.z * voxsize.z * 0.5f;

    // If the offset is not defined, chose the volume center
    if ( !flag_offset )
    {
        offset.x = h_lengthx;
        offset.y = h_lengthy;
        offset.z = h_lengthz;
    }
    else
    {
        offset.x = ox;
        offset.y = oy;
        offset.z = oz;
    }

    return data;
}

*/

#endif
