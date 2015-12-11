#ifndef DOSIMETRY_MANAGER_CUH
#define DOSIMETRY_MANAGER_CUH

#include "global.cuh"
#include "particles.cuh"
#include "vector.cuh"
#include "image_reader.cuh"
#include <ostream> 
#include <iomanip>
#ifndef DOSIMETRY
#define DOSIMETRY
/**
 * \struct Dosimetry
 * \brief Dosimetry structure
 *
 * Structure where dosimetry parameters are store during the simulation (edep and edep squared). The size of the dosimetry volume is the same as the voxelised volume
 * \param edep Energy deposited inside the volume
 * \param dose Dose deposited inside the volume
 * \param edep_Squared_value Energy squared deposited inside the volume
 * \param uncertainty Uncertainty associated with the Energy deposited inside the volume
 * \param nb_of_voxels Number of voxels inside the volume, and also the size of the dosimetrics array
 **/
struct DosimetryTable
    {
    // Dose storage
    f32 * edep;
    f32 * dose;
    f32 * edep_squared;
    ui32 *number_of_hits;
    
    // Not used in GPU
    f32 * uncertainty;
    
    // Number of voxels per dimension
    ui32 nx;
    ui32 ny;
    ui32 nz;
    
    // Voxel size per dimension
    f32 spacing_x;
    f32 spacing_y;
    f32 spacing_z;
    
    // Offset
    f32 x0;
    f32 y0;
    f32 z0;
    
    ui32 nb_of_voxels;
    };
#endif


class DoseCalculator {

//     std::ostream& operator<<(std::ostream& os, const DoseCalculator& v);


    public:
        DoseCalculator();
        ~DoseCalculator();
        
        void initialize(ui32xyz,f32xyz, f32xyz);
        
//         __host__ __device__ void store_energy_and_energy2(ui32, f32);
        
        void m_copy_dosi_cpu2gpu();
        DosimetryTable dose_h;
        DosimetryTable dose_d;
        
        void write_dosi(std::string);
        
        friend std::ostream& operator<<(std::ostream& os, const DoseCalculator& v)
        {
            os  << std::fixed << std::setprecision(2);
            os  << "Dosemap parameters  : " << std::endl;

            os  << "\t"  <<  "+"  << std::setfill('-') << std::setw(30) << "+" << std::endl;
            os  << std::setfill(' ');
            
            os  << "\t"   << "|" 
                << std::left  << std::setw(9) << "" 
                << std::right << std::setw(5) << "X"
                << std::right << std::setw(7) << "Y" 
                << std::right << std::setw(7) << "Z"
                << std::setw(2)<< "|" << std::endl;
            
            os  << "\t"   << "|" 
                << std::left  << std::setw(9) << "Offset" 
                << std::right << std::setw(5) << v.dose_h.x0 
                << std::right << std::setw(7) << v.dose_h.y0  
                << std::right << std::setw(7) << v.dose_h.z0 
                << std::setw(2)<< "|" << std::endl;
               
            os  << "\t"   << "|" 
                << std::left  << std::setw(9) << "Spacing" 
                << std::right << std::setw(5) << v.dose_h.spacing_x 
                << std::right << std::setw(7) << v.dose_h.spacing_y  
                << std::right << std::setw(7) << v.dose_h.spacing_z 
                << std::setw(2)<< "|" << std::endl;
                
            os << "\t"   << "|" 
                << std::left  << std::setw(9) << "Size" 
                << std::right << std::setw(5) << v.dose_h.nx 
                << std::right << std::setw(7)<< v.dose_h.ny 
                << std::right << std::setw(7)<< v.dose_h.nz 
                << std::setw(2)<< "|" << std::endl;
                
            os << "\t"   <<  "+"  << std::setfill('-') << std::setw(30) << "+" << std::endl;

            return os;
        }   
        

        
    private :
        
        
};




#endif