// GGEMS Copyright (C) 2015

/*!
 * \file vox_phan_dosi.cuh
 * \brief
 * \author Y. Lemaréchal
 * \version 0.1
 * \date 18 novembre 2015
 *
 *
 *
 */

#ifndef VOX_PHAN_DOSI_CUH
#define VOX_PHAN_DOSI_CUH

#include "global.cuh"
#include "voxelized.cuh"
#include "raytracing.cuh"
#include "vector.cuh"
#include "materials.cuh"
#include "photon.cuh"
#include "photon_navigator.cuh"
#include "image_reader.cuh"
#include "dosimetry_actor.cuh"

class VoxPhanDosi {
    public:
        VoxPhanDosi() {m_dose_calculator = new DoseCalculator;}
        ~VoxPhanDosi() {}

        // Tracking from outside to the phantom broder
        void track_to_in(Particles particles);
        // Tracking inside the phantom until the phantom border
        void track_to_out(Particles particles, Materials materials, PhotonCrossSection photon_CS);
        
        // Check format phantom file
        void load_phantom(std::string phantomfile, std::string materialfile);  
        
        // Init
        void initialize(GlobalSimulationParameters params);

        // Get list of materials
        std::vector<std::string> get_materials_list();
        // Get data that contains materials index
        ui16* get_data_materials_indices();
        // Get the size of data (nb of voxels)
        ui32 get_data_size();

        inline std::string get_name(){return "VoxPhanDosi";};
        
        // Dosimetry map functions
        void add_dosimetry_map(); // Without arg dosemap = phan size and position
        void add_dosimetry_map(ui32xyz,f32xyz,f32xyz);
        void add_dosimetry_map(ui32,ui32,ui32,f32,f32,f32,f32,f32,f32);
        
        void write(std::string filename = "dosimetry.mhd"){
            m_dose_calculator->write_dosi(filename);
        }
        

        void print_dosimetry();
    private:
        void load_phantom_from_mhd(std::string, std::string);  
        
        VoxelizedPhantom phantom;
    
        bool m_check_mandatory();       
        void m_copy_phantom_cpu2gpu();

        GlobalSimulationParameters m_params;

        DoseCalculator *m_dose_calculator;
        
        
        friend std::ostream& operator<<(std::ostream& os, VoxPhanDosi& v)
        {
            os  << std::fixed << std::setprecision(2);
            os  << "VoxPhanDosi parameters  : " << std::endl;

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
                << std::right << std::setw(5) << v.phantom.volume.data_h.org_x 
                << std::right << std::setw(7) << v.phantom.volume.data_h.org_y
                << std::right << std::setw(7) << v.phantom.volume.data_h.org_z
                << std::setw(2)<< "|" << std::endl;
               
            os  << "\t"   << "|" 
                << std::left  << std::setw(9) << "Spacing" 
                << std::right << std::setw(5) << v.phantom.volume.data_h.spacing_x 
                << std::right << std::setw(7) << v.phantom.volume.data_h.spacing_y  
                << std::right << std::setw(7) << v.phantom.volume.data_h.spacing_z 
                << std::setw(2)<< "|" << std::endl;
                
            os << "\t"   << "|" 
                << std::left  << std::setw(9) << "Size" 
                << std::right << std::setw(5) << v.phantom.volume.data_h.nb_vox_x 
                << std::right << std::setw(7) << v.phantom.volume.data_h.nb_vox_y 
                << std::right << std::setw(7) << v.phantom.volume.data_h.nb_vox_z 
                << std::setw(2)<< "|" << std::endl;
                
            os << "\t"   <<  "+"  << std::setfill('-') << std::setw(30) << "+" << std::endl;

            std::cout<<"  Associated ";
            v.print_dosimetry();
            
            return os;
            
            
        }   
        
        
//         
};

#endif
