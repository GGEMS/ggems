#ifndef SAVE_DATA_CUH
#define SAVE_DATA_CUH

#include <iostream>
#include "global.cuh"

namespace GGEMSutils
{
    

    // Record dose map 
    void record_dose_map( std::string histname,  f32 *data, f32xyz offset, f32xyz spacing, i32xyz size, bool sparce_compression = false);
    
    inline std::string get_format(std::string filename);

    inline std::string get_filename_without_format(std::string filename);
    
};
    
    
#endif