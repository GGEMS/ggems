#ifndef IMAGE_READER_CUH
#define IMAGE_READER_CUH

#include <iostream>
#include "global.cuh"
#include "fun.cuh"

namespace ImageReader
{
    template<typename T>
    void recordTables1 ( std::ofstream &outputFile , int i,T *t ) // recursive variadic function
    {
        outputFile << t[i] << "\t";

    }


    template<typename T, typename... Args>
    void recordTables1 ( std::ofstream &outputFile , int i,T *t, Args... args ) // recursive variadic function
    {

        outputFile << t[i] << "\t";
        recordTables1 ( outputFile, i, args... ) ;


    }

    template<typename T, typename... Args >
    void recordTables ( std::string filename, unsigned int size ,T *t, Args... args ) // recursive variadic function
    {

        std::ofstream outputFile ( filename.c_str(), std::ios::out | std::ios::trunc );

        for ( int i= 0; i < size ; i++ )
        {
            outputFile << t[i] << "\t";
            recordTables1 ( outputFile, i, args... ) ;

            outputFile << std::endl;
        }

        outputFile.close();
    }


    template<typename T, typename... Args >
    void recordTables ( std::string filename, unsigned int begin, unsigned int stop ,T *t, Args... args ) // recursive variadic function
    {

        std::ofstream outputFile ( filename.c_str(), std::ios::out | std::ios::trunc );

        for ( int i= begin; i < stop ; i++ )
        {
            outputFile << t[i] << "\t";
            recordTables1 ( outputFile, i, args... ) ;

            outputFile << std::endl;
        }

        outputFile.close();
    }


    template < typename T >
    void recordTable ( std::string filename, T* data,int size )
    {

        std::ofstream outputFile ( filename.c_str(), std::ios::out | std::ios::trunc );

        for ( int i = 0; i< size; i++ )
        {
            outputFile << data[i] <<std::endl;

        }

        outputFile.close();
    }


    void recordImage();

    void record2DImage();

    // Record dose map
    //     void record3Dimage( std::string histname,  f32 *data, f32xyz offset, f32xyz spacing, i32xyz size, bool sparce_compression = false);

    void record3Dimage ( std::string histname,  f64 *data, f32xyz offset, f32xyz spacing, i32xyz size, bool sparce_compression = false );
    void record3Dimage ( std::string histname,  ui32 *data, f32xyz offset, f32xyz spacing, i32xyz size, bool sparce_compression = false );
    void record3Dimage ( std::string histname,  ui16 *data, f32xyz offset, f32xyz spacing, i32xyz size, bool sparce_compression = false );

    inline std::string get_format ( std::string filename )
    {
        return filename.substr ( filename.find_last_of ( "." ) + 1 );
    }

    inline std::string get_filename_without_format ( std::string filename, std::string separator= "." )
    {
        return filename.substr ( 0,filename.find_last_of ( separator.c_str() ) );
    }

    inline std::string remove_path ( std::string filename, std::string separator= "/" )
    {
        return filename.substr ( filename.find_last_of ( separator.c_str() ) +1 );
    }

    void create_directory(std::string dirname);

    void create_directory_tree(std::string dirname);
//         
};
#endif
