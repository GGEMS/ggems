// GGEMS Copyright (C) 2015

/*!
 * \file image_io.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 03/05/2016 - Must replace the awful image_reader
 *
 *
 */

#ifndef IMAGE_IO_CUH
#define IMAGE_IO_CUH

#include "global.cuh"

// Handle image (1D, 2D or 3D)
class ImageIO {

    public:
        ImageIO(){}
        ~ImageIO(){}

    public:
        void write_2D( std::string filename,  f32 *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression = false );
        void write_2D( std::string filename,  i32 *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression = false );
        void write_2D( std::string filename, ui32 *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression = false );
        void write_2D( std::string filename,  i16 *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression = false );
        void write_2D( std::string filename, ui16 *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression = false );
        void write_2D( std::string filename,   i8 *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression = false );
        void write_2D( std::string filename,  ui8 *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression = false );

        void write_3D( std::string filename,  f32 *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression = false );
        void write_3D( std::string filename,  i32 *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression = false );
        void write_3D( std::string filename, ui32 *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression = false );
        void write_3D( std::string filename,  i16 *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression = false );
        void write_3D( std::string filename, ui16 *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression = false );
        void write_3D( std::string filename,   i8 *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression = false );
        void write_3D( std::string filename,  ui8 *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression = false );

        std::string get_format( std::string filename );
        std::string get_filename_without_format( std::string filename, std::string separator= "." );

    private:
        template<typename Type2D>
        void m_write_2D( std::string filename, Type2D *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression );
        template<typename Type3D>
        void m_write_3D( std::string filename, Type3D *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression );

        std::string m_remove_path( std::string filename, std::string separator= "/" );
        void m_create_directory( std::string dirname );
        void m_create_directory_tree( std::string dirname );

};




#endif
