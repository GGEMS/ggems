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
#include "vector.cuh"
#include "txt_reader.cuh"

// Handle image (1D, 2D or 3D)
class ImageIO {

    public:
        ImageIO();
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

        void open( std::string filename );

    public:
        std::string get_extension( std::string filename );
        std::string get_filename_without_extension( std::string filename, std::string separator= "." );
        ui8 get_dim();
        std::string get_type();
        f32xyz get_offset();
        f32xyz get_spacing();
        ui32xyz get_size();

        f32*  get_image_in_f32();
        i32*  get_image_in_i32();
        ui32* get_image_in_ui32();
        i16*  get_image_in_i16();
        ui16* get_image_in_ui16();
        i8*   get_image_in_i8();
        ui8*  get_image_in_ui8();

    private:
        template<typename Type2D>
        void m_write_2D( std::string filename, Type2D *data, ui32xy size, f32xy offset, f32xy spacing, bool sparse_compression );
        template<typename Type3D>
        void m_write_3D( std::string filename, Type3D *data, ui32xyz size, f32xyz offset, f32xyz spacing, bool sparse_compression );

        std::string m_remove_path( std::string filename, std::string separator= "/" );
        void m_create_directory( std::string dirname );
        void m_create_directory_tree( std::string dirname );

        // Image params
        ui8 m_dim;           // 1D, 2D or 3D
        std::string m_type;  // f32, ui32, i32, etc.
        f32xyz m_offset, m_spacing;
        ui32xyz m_size;
        ui32    m_nb_data;
        f32    *m_f32_data;
        i32    *m_i32_data;
        ui32   *m_ui32_data;
        i16    *m_i16_data;
        ui16   *m_ui16_data;
        i8     *m_i8_data;
        ui8    *m_ui8_data;

        bool m_image_loaded;

};




#endif
