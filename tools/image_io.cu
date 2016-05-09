// GGEMS Copyright (C) 2015

/*!
 * \file image_io.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 03/05/2016 - Must replace the awful image_reader
 *
 *
 */

#ifndef IMAGE_IO_CU
#define IMAGE_IO_CU

#include "image_io.cuh"

/// Classe ///////////////////////////////////////////////////////

ImageIO::ImageIO()
{
    // Init params
    m_dim = 2;     // 2D
    m_type = "";
    m_offset = make_f32xyz(0.0, 0.0, 0.0);
    m_spacing = make_f32xyz(1.0, 1.0, 1.0);
    m_size = make_ui32xyz(0, 0, 0);
    m_nb_data = 0;

    m_f32_data = NULL;
    m_i32_data = NULL;
    m_ui32_data = NULL;
    m_i16_data = NULL;
    m_ui16_data = NULL;
    m_i8_data = NULL;
    m_ui8_data = NULL;

    m_image_loaded = false;
}

/// Private functions ////////////////////////////////////////////

std::string ImageIO::m_remove_path( std::string filename, std::string separator )
{
    return filename.substr( filename.find_last_of ( separator.c_str() ) +1 );
}

void ImageIO::m_create_directory( std::string dirname )
{
    std::string command = "mkdir -p " + dirname;
    system(command.c_str());
}

void ImageIO::m_create_directory_tree( std::string dirname )
{
    std::string tmp = dirname;;
    std::string directory = "";

    for(;;)
    {
        if(tmp.substr (0, tmp.find_first_of ( "/" ) +1) == "") break;
        directory += tmp.substr (0, tmp.find_first_of ( "/" ) +1);
        tmp = tmp.substr (tmp.find_first_of ( "/" ) +1 );

        m_create_directory( directory );
    }
}

template<typename Type2D>
void ImageIO::m_write_2D( std::string filename, Type2D *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression )
{
    GGcout << "Write image " << filename << " ... " << GGendl;

    // Check format
    std::string format = get_extension( filename );
    filename = get_filename_without_extension( filename );

    if ( format != "mhd" )
    {
        GGcerr << "Image must exported in MHD format (.mhd): ." << format << " given!" << GGendl;
        exit_simulation();
    }

    // Build directory if need
    m_create_directory_tree(filename);

    // Define new extension names
    std::string pathnamemhd = filename + ".mhd";
    std::string pathnameraw = filename + ".raw";

    // MHD file
    std::ofstream myfile;
    myfile.open ( pathnamemhd.c_str() );
    myfile << "ObjectType = Image\n";
    myfile << "NDims = 2\n";
    myfile << "BinaryData = True\n";
    myfile << "BinaryDataByteOrderMSB = False\n";

    myfile << "CompressedData = " << ( ( sparse_compression ) ? "COO" : "False" )  <<"\n";

    myfile << "TransformMatrix = 1 0 0 1\n";
    myfile << "Offset = " << offset.x << " " << offset.y << "\n";
    myfile << "CenterOfRotation = 0 0\n";
    myfile << "ElementSpacing = " << spacing.x << " " << spacing.y << "\n";
    myfile << "DimSize = " << size.x << " " << size.y << "\n";
    myfile << "AnatomicalOrientation = ???\n";

    // Select the right type
    if ( std::is_same< Type2D, f32>::value )
    {
        myfile << "ElementType = MET_FLOAT\n";
    }
    else if ( std::is_same< Type2D, i32>::value )
    {
        myfile << "ElementType = MET_INT\n";
    }
    else if ( std::is_same< Type2D, ui32>::value )
    {
        myfile << "ElementType = MET_UINT\n";
    }
    else if ( std::is_same< Type2D, i16>::value )
    {
        myfile << "ElementType = MET_SHORT\n";
    }
    else if ( std::is_same< Type2D, ui16>::value )
    {
        myfile << "ElementType = MET_USHORT\n";
    }
    else if ( std::is_same< Type2D, i8>::value )
    {
        myfile << "ElementType = MET_CHAR\n";
    }
    else if ( std::is_same< Type2D, ui8>::value )
    {
        myfile << "ElementType = MET_UCHAR\n";
    }
    else
    {
        GGcerr << "Image data type not recongnized!" << GGendl;
        exit_simulation();
    }


    myfile << "ElementDataFile = " << m_remove_path( pathnameraw ).c_str() <<"\n";
    myfile.close();

    // RAW File
    FILE *pFile_mhd;
    pFile_mhd = fopen ( pathnameraw.c_str(), "wb" );

    // Compressed data in COO format
    if ( sparse_compression )
    {
        // First get the number of non-zero
        ui32 index = 0;
        ui32 ct_nz = 0;
        while ( index < size.x*size.y )
        {
            if ( data[index] != 0.0 ) ++ct_nz;
            ++index;
        }

        // Write the previous value as the first binary element
        fwrite ( &ct_nz, sizeof ( ui32 ), 1, pFile_mhd );

        // Some vars
        ui16 ix, iy;
        index = 0;

        // Loop over every element
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
                    // Then the corresponding value
                    Type2D val = data[index];
                    fwrite ( &val, sizeof ( Type2D ), 1, pFile_mhd );
                }

                ++ix;
                ++index;
            } // ix
            ++iy;
        } // iy

    }
    else
    {
        // Export uncompressed raw data
        fwrite( data, sizeof(Type2D), size.x*size.y, pFile_mhd );

//        ui32 i=0;
//        while ( i<size.x*size.y )
//        {
//            Type val = data[i];
//            fwrite( &val, sizeof ( f32 ), 1, pFile_mhd );
//            ++i;
//        }
    }

    fclose ( pFile_mhd );
}

template<typename Type3D>
void ImageIO::m_write_3D( std::string filename, Type3D *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression )
{

    GGcout << "Write image " << filename << " ... " << GGendl;

    // Check format
    std::string format = get_extension( filename );
    filename = get_filename_without_extension( filename );

    if ( format != "mhd" )
    {
        GGcerr << "Image must exported in MHD format (.mhd): ." << format << " given!" << GGendl;
        exit_simulation();
    }

    // Build directory if need
    m_create_directory_tree(filename);

    // Define new extension names
    std::string pathnamemhd = filename + ".mhd";
    std::string pathnameraw = filename + ".raw";

    // MHD file
    std::ofstream myfile;
    myfile.open ( pathnamemhd.c_str() );
    myfile << "ObjectType = Image\n";
    myfile << "NDims = 3\n";
    myfile << "BinaryData = True\n";
    myfile << "BinaryDataByteOrderMSB = False\n";

    myfile << "CompressedData = " << ( ( sparse_compression ) ? "COO" : "False" )  <<"\n";

    myfile << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n";
    myfile << "Offset = " << offset.x << " " << offset.y << " " << offset.z << "\n";
    myfile << "CenterOfRotation = 0 0 0\n";
    myfile << "ElementSpacing = " << spacing.x << " " << spacing.y << " " << spacing.z << "\n";
    myfile << "DimSize = " << size.x << " " << size.y << " " << size.z << "\n";
    myfile << "AnatomicalOrientation = ???\n";

    // Select the right type
    if ( std::is_same< Type3D, f32>::value )
    {
        myfile << "ElementType = MET_FLOAT\n";
    }
    else if ( std::is_same< Type3D, i32>::value )
    {
        myfile << "ElementType = MET_INT\n";
    }
    else if ( std::is_same< Type3D, ui32>::value )
    {
        myfile << "ElementType = MET_UINT\n";
    }
    else if ( std::is_same< Type3D, i16>::value )
    {
        myfile << "ElementType = MET_SHORT\n";
    }
    else if ( std::is_same< Type3D, ui16>::value )
    {
        myfile << "ElementType = MET_USHORT\n";
    }
    else if ( std::is_same< Type3D, i8>::value )
    {
        myfile << "ElementType = MET_CHAR\n";
    }
    else if ( std::is_same< Type3D, ui8>::value )
    {
        myfile << "ElementType = MET_UCHAR\n";
    }
    else
    {
        GGcerr << "Image data type not recongnized!" << GGendl;
        exit_simulation();
    }

    myfile << "ElementDataFile = " << m_remove_path( pathnameraw ).c_str() <<"\n";
    myfile.close();

    // RAW File
    FILE *pFile_mhd;
    pFile_mhd = fopen ( pathnameraw.c_str(), "wb" );

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
                        Type3D val = data[index];
                        fwrite ( &val, sizeof ( Type3D ), 1, pFile_mhd );
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
        fwrite( data, sizeof(Type3D), size.x*size.y*size.z, pFile_mhd );

//        ui32 i=0;
//        while ( i<size.x*size.y )
//        {
//            Type val = data[i];
//            fwrite( &val, sizeof ( f32 ), 1, pFile_mhd );
//            ++i;
//        }
    }

    fclose ( pFile_mhd );
}

/// Publics functions ////////////////////////////////////////////

std::string ImageIO::get_extension( std::string filename )
{
    // Get ext
    std::string ext = filename.substr( filename.find_last_of( "." ) + 1 );

    // Get lower case
    std::transform( ext.begin(), ext.end(), ext.begin(), ::tolower );

    return ext;
}

std::string ImageIO::get_filename_without_extension( std::string filename, std::string separator )
{
    return filename.substr( 0, filename.find_last_of ( separator.c_str() ) );
}

ui8 ImageIO::get_dim()
{
    return m_dim;
}

std::string ImageIO::get_type()
{
    return m_type;
}

f32xyz ImageIO::get_offset()
{
    return m_offset;
}

f32xyz ImageIO::get_spacing()
{
    return m_spacing;
}

ui32xyz ImageIO::get_size()
{
    return m_size;
}

f32* ImageIO::get_image_in_f32()
{
    // Same type that the one required
    if ( m_type == "f32" )
    {
        return m_f32_data;
    }
    else
    {
        f32 *tmp = new f32[ m_nb_data ];
        ui32 i;

        i=0; while (i < m_nb_data )
        {
            if ( m_type == "i32" )  tmp[ i ] = (f32) m_i32_data[ i ];
            if ( m_type == "ui32" ) tmp[ i ] = (f32) m_ui32_data[ i ];
            if ( m_type == "i16" )  tmp[ i ] = (f32) m_i16_data[ i ];
            if ( m_type == "ui16" ) tmp[ i ] = (f32) m_ui16_data[ i ];
            if ( m_type == "i8" )   tmp[ i ] = (f32) m_i8_data[ i ];
            if ( m_type == "ui8" )  tmp[ i ] = (f32) m_ui8_data[ i ];

            ++i;
        }
        return tmp;
    }
}

i32* ImageIO::get_image_in_i32()
{
    // Same type that the one required
    if ( m_type == "i32" )
    {
        return m_i32_data;
    }
    else
    {
        i32 *tmp = new i32[ m_nb_data ];
        ui32 i;

        i=0; while (i < m_nb_data )
        {
            if ( m_type == "f32" )  tmp[ i ] = (i32) m_f32_data[ i ];
            if ( m_type == "ui32" ) tmp[ i ] = (i32) m_ui32_data[ i ];
            if ( m_type == "i16" )  tmp[ i ] = (i32) m_i16_data[ i ];
            if ( m_type == "ui16" ) tmp[ i ] = (i32) m_ui16_data[ i ];
            if ( m_type == "i8" )   tmp[ i ] = (i32) m_i8_data[ i ];
            if ( m_type == "ui8" )  tmp[ i ] = (i32) m_ui8_data[ i ];

            ++i;
        }
        return tmp;
    }
}

ui32* ImageIO::get_image_in_ui32()
{
    // Same type that the one required
    if ( m_type == "ui32" )
    {
        return m_ui32_data;
    }
    else
    {
        ui32 *tmp = new ui32[ m_nb_data ];
        ui32 i;

        i=0; while (i < m_nb_data )
        {
            if ( m_type == "f32" )  tmp[ i ] = (ui32) m_f32_data[ i ];
            if ( m_type == "i32" )  tmp[ i ] = (ui32) m_i32_data[ i ];
            if ( m_type == "i16" )  tmp[ i ] = (ui32) m_i16_data[ i ];
            if ( m_type == "ui16" ) tmp[ i ] = (ui32) m_ui16_data[ i ];
            if ( m_type == "i8" )   tmp[ i ] = (ui32) m_i8_data[ i ];
            if ( m_type == "ui8" )  tmp[ i ] = (ui32) m_ui8_data[ i ];

            ++i;
        }
        return tmp;
    }
}

i16* ImageIO::get_image_in_i16()
{
    // Same type that the one required
    if ( m_type == "i16" )
    {
        return m_i16_data;
    }
    else
    {
        i16 *tmp = new i16[ m_nb_data ];
        ui32 i;

        i=0; while (i < m_nb_data )
        {
            if ( m_type == "f32" )  tmp[ i ] = (i16) m_f32_data[ i ];
            if ( m_type == "i32" )  tmp[ i ] = (i16) m_i32_data[ i ];
            if ( m_type == "ui32" ) tmp[ i ] = (i16) m_ui32_data[ i ];
            if ( m_type == "ui16" ) tmp[ i ] = (i16) m_ui16_data[ i ];
            if ( m_type == "i8" )   tmp[ i ] = (i16) m_i8_data[ i ];
            if ( m_type == "ui8" )  tmp[ i ] = (i16) m_ui8_data[ i ];

            ++i;
        }
        return tmp;
    }
}

ui16* ImageIO::get_image_in_ui16()
{
    // Same type that the one required
    if ( m_type == "ui16" )
    {
        return m_ui16_data;
    }
    else
    {
        ui16 *tmp = new ui16[ m_nb_data ];
        ui32 i;

        i=0; while (i < m_nb_data )
        {
            if ( m_type == "f32" )  tmp[ i ] = (ui16) m_f32_data[ i ];
            if ( m_type == "i32" )  tmp[ i ] = (ui16) m_i32_data[ i ];
            if ( m_type == "ui32" ) tmp[ i ] = (ui16) m_ui32_data[ i ];
            if ( m_type == "i16" )  tmp[ i ] = (ui16) m_i16_data[ i ];
            if ( m_type == "i8" )   tmp[ i ] = (ui16) m_i8_data[ i ];
            if ( m_type == "ui8" )  tmp[ i ] = (ui16) m_ui8_data[ i ];

            ++i;
        }
        return tmp;
    }
}

i8* ImageIO::get_image_in_i8()
{
    // Same type that the one required
    if ( m_type == "i8" )
    {
        return m_i8_data;
    }
    else
    {
        i8 *tmp = new i8[ m_nb_data ];
        ui32 i;

        i=0; while (i < m_nb_data )
        {
            if ( m_type == "f32" )  tmp[ i ] = (i8) m_f32_data[ i ];
            if ( m_type == "i32" )  tmp[ i ] = (i8) m_i32_data[ i ];
            if ( m_type == "ui32" ) tmp[ i ] = (i8) m_ui32_data[ i ];
            if ( m_type == "i16" )  tmp[ i ] = (i8) m_i16_data[ i ];
            if ( m_type == "ui16" ) tmp[ i ] = (i8) m_ui16_data[ i ];
            if ( m_type == "ui8" )  tmp[ i ] = (i8) m_ui8_data[ i ];

            ++i;
        }
        return tmp;
    }
}

ui8* ImageIO::get_image_in_ui8()
{
    // Same type that the one required
    if ( m_type == "ui8" )
    {
        return m_ui8_data;
    }
    else
    {
        ui8 *tmp = new ui8[ m_nb_data ];
        ui32 i;

        i=0; while (i < m_nb_data )
        {
            if ( m_type == "f32" )  tmp[ i ] = (ui8) m_f32_data[ i ];
            if ( m_type == "i32" )  tmp[ i ] = (ui8) m_i32_data[ i ];
            if ( m_type == "ui32" ) tmp[ i ] = (ui8) m_ui32_data[ i ];
            if ( m_type == "i16" )  tmp[ i ] = (ui8) m_i16_data[ i ];
            if ( m_type == "ui16" ) tmp[ i ] = (ui8) m_ui16_data[ i ];
            if ( m_type == "i8" )   tmp[ i ] = (ui8) m_i8_data[ i ];

            ++i;
        }
        return tmp;
    }
}

/// Publics functions - IO ////////////////////////////////////////////

// 2D //////////////////////////////

void ImageIO::write_2D( std::string filename, f32 *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression )
{
    m_write_2D( filename, data, size, offset, spacing, sparse_compression );
}

void ImageIO::write_2D( std::string filename, i32 *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression )
{
    m_write_2D( filename, data, size, offset, spacing, sparse_compression );
}

void ImageIO::write_2D( std::string filename, ui32 *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression )
{
    m_write_2D( filename, data, size, offset, spacing, sparse_compression );
}

void ImageIO::write_2D( std::string filename, i16 *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression )
{
    m_write_2D( filename, data, size, offset, spacing, sparse_compression );
}

void ImageIO::write_2D( std::string filename, ui16 *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression )
{
    m_write_2D( filename, data, size, offset, spacing, sparse_compression );
}

void ImageIO::write_2D( std::string filename, i8 *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression )
{
    m_write_2D( filename, data, size, offset, spacing, sparse_compression );
}

void ImageIO::write_2D( std::string filename, ui8 *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression )
{
    m_write_2D( filename, data, size, offset, spacing, sparse_compression );
}

// 3D ///////////////////////////////

void ImageIO::write_3D( std::string filename, f32 *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression )
{
    m_write_3D( filename, data, size, offset, spacing, sparse_compression );
}

void ImageIO::write_3D( std::string filename, i32 *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression )
{
    m_write_3D( filename, data, size, offset, spacing, sparse_compression );
}

void ImageIO::write_3D( std::string filename, ui32 *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression )
{
    m_write_3D( filename, data, size, offset, spacing, sparse_compression );
}

void ImageIO::write_3D( std::string filename, i16 *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression )
{
    m_write_3D( filename, data, size, offset, spacing, sparse_compression );
}

void ImageIO::write_3D( std::string filename, ui16 *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression )
{
    m_write_3D( filename, data, size, offset, spacing, sparse_compression );
}

void ImageIO::write_3D( std::string filename, i8 *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression )
{
    m_write_3D( filename, data, size, offset, spacing, sparse_compression );
}

void ImageIO::write_3D( std::string filename, ui8 *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression )
{
    m_write_3D( filename, data, size, offset, spacing, sparse_compression );
}

// Open

void ImageIO::open( std::string filename )
{
    // Check if MHD file
    std::string ext = get_extension( filename );
    if ( ext != "mhd" )
    {
        GGcerr << "GGEMS can only open MHD image file format!" << GGendl;
        exit_simulation();;
    }

    TxtReader txt_reader;

    std::string line, key;
    i32 nx=-1, ny=-1, nz=-1;
    f32 sx=0, sy=0, sz=0;
    f32 ox=0, oy=0, oz=0;

    bool flag_offset = false;

    // Watchdog
    std::string ObjectType="", BinaryData="", BinaryDataByteOrderMSB="", CompressedData="",
                ElementType="", ElementDataFile="";
    i32 NDims = 0;

    // Read file
    std::ifstream file( filename.c_str() );

    if ( !file )
    {
        GGcerr << "Error, file " << filename << " not found " << GGendl;
        exit_simulation();
    }

    // First reading
    while (file) {
        txt_reader.skip_comment(file);
        std::getline(file, line);

        if (file) {
            key = txt_reader.read_key(line);
            if ( key == "ObjectType" )              ObjectType = txt_reader.read_key_string_arg(line);
            if ( key == "NDims" )                   NDims = txt_reader.read_key_i32_arg(line);
            if ( key == "BinaryData" )              BinaryData = txt_reader.read_key_string_arg(line);
            if ( key == "BinaryDataByteOrderMSB" )  BinaryDataByteOrderMSB = txt_reader.read_key_string_arg(line);
            if ( key == "CompressedData" )          CompressedData = txt_reader.read_key_string_arg(line);
            if ( key == "ElementType" )             ElementType = txt_reader.read_key_string_arg(line);
            if ( key == "ElementDataFile" )         ElementDataFile = txt_reader.read_key_string_arg(line);
        }

    } // read file

    // Check Header
    if ( ObjectType == "" ) {
        GGcerr << "Open MHD image: ObjectType was not specified!" << GGendl;
        exit_simulation();
    }
    if ( ObjectType != "Image" ) {
        GGcerr << "Open MHD image: Not an image, ObjectType = " << ObjectType << GGendl;
        exit_simulation();
    }

    if ( NDims == 0 ) {
        GGcerr << "Open MHD image: NDims was not specified!" << GGendl;
        exit_simulation();
    }
    if ( NDims == 0 ) {
        GGcerr << "Open MHD image: NDims was not specified!" << GGendl;
        exit_simulation();
    }

    if ( BinaryData != "True" ) {
        GGcerr << "Open MHD image: should binary data, BinaryData = " << BinaryData << GGendl;
        exit_simulation();
    }

    if ( BinaryDataByteOrderMSB != "False" ) {
        GGcerr << "Open MHD image: byte order should be not in MSB, BinaryDataByteOrderMSB = " << BinaryDataByteOrderMSB << GGendl;
        exit_simulation();
    }

    if ( CompressedData != "False" ) {
        GGcerr << "Open MHD image: cannot open compressed data yet, CompressedData = " << CompressedData << GGendl;
        exit_simulation();
    }

    if ( ElementType == "" ) {
        GGcerr << "Open MHD image: ElementType was not specified!" << GGendl;
        exit_simulation();
    }
    if (ElementType != "MET_FLOAT" &&
        ElementType != "MET_INT"   && ElementType != "MET_UINT" &&
        ElementType != "MET_SHORT" && ElementType != "MET_USHORT" &&
        ElementType != "MET_CHAR"  && ElementType != "MET_UCHAR" ) {
        GGcerr << "Open MHD image: Data Type not recognized, ElementType = " << ElementType << GGendl;
        exit_simulation();
    }

    if ( ElementDataFile == "" ) {
        GGcerr << "Open MHD image: ElementDataFile was not specified!" << GGendl;
        exit_simulation();
    }

    // Second reading
    file.seekg( 0 );
    while (file) {
        txt_reader.skip_comment(file);
        std::getline(file, line);

        if (file) {
            key = txt_reader.read_key(line);

            if (key == "Offset") {
                if ( NDims >= 1 ) ox = txt_reader.read_key_f32_arg_atpos(line, 0);
                if ( NDims >= 2 ) oy = txt_reader.read_key_f32_arg_atpos(line, 1);
                if ( NDims >= 3 ) oz = txt_reader.read_key_f32_arg_atpos(line, 2);
                flag_offset = true;
            }

            if (key=="ElementSpacing") {
                if ( NDims >= 1 ) sx = txt_reader.read_key_f32_arg_atpos(line, 0);
                if ( NDims >= 2 ) sy = txt_reader.read_key_f32_arg_atpos(line, 1);
                if ( NDims >= 3 ) sz = txt_reader.read_key_f32_arg_atpos(line, 2);
            }

            if (key=="DimSize") {
                if ( NDims >= 1 ) nx = txt_reader.read_key_i32_arg_atpos(line, 0);
                if ( NDims >= 2 ) ny = txt_reader.read_key_i32_arg_atpos(line, 1);
                if ( NDims >= 3 ) nz = txt_reader.read_key_i32_arg_atpos(line, 2);
            }
        }

    } // read file

    // Check data
    if ( NDims >= 1 )
    {
        if ( nx == -1  || sx == 0)
        {
            GGcerr << "Open MHD image: unknown dimension and spacing along x-axis!" << GGendl;
            exit_simulation();
        }
    }

    if ( NDims >= 2 )
    {
        if ( ny == -1  || sy == 0)
        {
            GGcerr << "Open MHD image: unknown dimension and spacing along y-axis!" << GGendl;
            exit_simulation();
        }
    }

    if ( NDims >= 3 )
    {
        if ( nz == -1  || sz == 0)
        {
            GGcerr << "Open MHD image: unknown dimension and spacing along z-axis!" << GGendl;
            exit_simulation();
        }
    }

    // Read data
    FILE *pfile = fopen(ElementDataFile.c_str(), "rb");

    // Relative path?
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

    // Store the current information
    if ( NDims >= 1 )
    {
        if ( !flag_offset ) m_offset.x = nx * sx * 0.5;
        else                m_offset.x = ox;
        m_size.x = nx;
        m_spacing.x = sx;
        m_nb_data = nx;
    }
    if ( NDims >= 2 )
    {
        if ( !flag_offset ) m_offset.y = ny * sy * 0.5;
        else                m_offset.y = oy;
        m_size.y = ny;
        m_spacing.y = sy;
        m_nb_data *= ny;
    }
    if ( NDims >= 3 )
    {
        if ( !flag_offset ) m_offset.z = nz * sz * 0.5;
        else                m_offset.z = oz;
        m_size.z = nz;
        m_spacing.z = sz;
        m_nb_data *= nz;
    }
    m_dim = NDims;

    // Allocation and reading
    if ( ElementType == "MET_FLOAT" )
    {
        m_f32_data = new f32( m_nb_data );
        fread( m_f32_data, sizeof( f32 ), m_nb_data, pfile );
        m_type = "f32";

    } else if ( ElementType == "MET_INT" )
    {
        m_i32_data = new i32( m_nb_data );
        fread( m_i32_data, sizeof( i32 ), m_nb_data, pfile );
        m_type = "i32";

    } else if ( ElementType == "MET_UINT" )
    {
        m_ui32_data = new ui32( m_nb_data );
        fread( m_ui32_data, sizeof( ui32 ), m_nb_data, pfile );
        m_type = "ui32";

    } else if ( ElementType == "MET_SHORT" )
    {
        m_i16_data = new i16( m_nb_data );
        fread( m_i16_data, sizeof( i16 ), m_nb_data, pfile );
        m_type = "i16";

    } else if ( ElementType == "MET_USHORT" )
    {
        m_ui16_data = new ui16( m_nb_data );
        fread( m_ui16_data, sizeof( ui16 ), m_nb_data, pfile );
        m_type = "ui16";

    } else if ( ElementType == "MET_CHAR" )
    {
        m_i8_data = new i8( m_nb_data );
        fread( m_i8_data, sizeof( i8 ), m_nb_data, pfile );
        m_type = "i8";

    } else if ( ElementType == "MET_UCHAR" )
    {
        m_ui8_data = new ui8( m_nb_data );
        fread( m_ui8_data, sizeof( ui8 ), m_nb_data, pfile );
        m_type = "ui8";
    }

    // Close the file
    fclose(pfile);

    // flag it
    m_image_loaded = true;

}




#endif












