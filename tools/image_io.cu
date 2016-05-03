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
    std::string format = get_format ( filename );
    filename = get_filename_without_format ( filename );

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

    // std::is_same


    GGcout << "Write image " << filename << " ... " << GGendl;

    // Check format
    std::string format = get_format ( filename );
    filename = get_filename_without_format ( filename );

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

std::string ImageIO::get_format( std::string filename )
{
    return filename.substr( filename.find_last_of( "." ) + 1 );
}

std::string ImageIO::get_filename_without_format( std::string filename, std::string separator )
{
    return filename.substr( 0, filename.find_last_of ( separator.c_str() ) );
}

/// 2D

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

/// 3D

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

#endif












