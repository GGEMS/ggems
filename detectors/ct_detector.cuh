// GGEMS Copyright (C) 2015

/*!
 * \file ct_detector.cuh
 * \brief CT detector (flatpanel)
 * \author Didier Benoit <didier.benoit13@gmail.com>
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.3
 * \date december 2, 2015
 *
 * v0.4: JB - Change all structs and remove CPU exec
 * v0.3: JB - Handle transformation (local frame to global frame) and add unified mem
 * v0.2: JB - Add digitizer
 * v0.1: DB - First code
 */

#ifndef CT_DETECTOR_CUH
#define CT_DETECTOR_CUH

#include "global.cuh"
#include "raytracing.cuh"
#include "particles.cuh"
#include "primitives.cuh"
#include "fun.cuh"
#include "image_io.cuh"
#include "ggems_detector.cuh"

#define MAX_SCATTER_ORDER 3
#define GET_HIT 1
#define GET_ENERGY 2

class GGEMSDetector;

class CTDetector : public GGEMSDetector
{
    public:
        CTDetector();
        ~CTDetector();

        // Setting
        //void set_dimension(f32 sizex, f32 sizey , f32 sizez );
        void set_number_of_pixels( ui32 nx, ui32 ny, ui32 nz );
        void set_pixel_size( f32 sx, f32 sy, f32 sz );
        void set_position( f32 x, f32 y, f32 z );
        void set_rotation( f32 rx, f32 ry, f32 rz );
        void set_projection_axis( f32 m00, f32 m01, f32 m02,
                                  f32 m10, f32 m11, f32 m12,
                                  f32 m20, f32 m21, f32 m22 );
        void set_record_option( std::string opt );
        void set_record_scatter( bool flag );

        void set_threshold( f32 threshold );

        f32matrix44 get_transformation();
        ObbData get_bounding_box();

        // Tracking from outside to the detector
        void track_to_in( ParticlesData *d_particles );
        void track_to_out( ParticlesData *d_particles ){}

        // Init
        void initialize( GlobalSimulationParametersData *h_params );
        void digitizer( ParticlesData *d_particles );

        void save_projection( std::string filename , std::string format = "f32" );
        void save_scatter( std::string filename );

        void print_info_scatter();

    private:
        ui32 get_detected_particles();
        ui32 get_scatter_number( ui32 scatter_order );

    private:

        // Image Projection
        f32xyz m_pixel_size;
        ui32xyz m_nb_pixel;

        f32 *m_projection;
        ui32 *m_scatter;

        // Flat panel
        f32xyz m_dim;
        f32xyz m_pos;
        f32xyz m_angle;
        f32matrix33 m_proj_axis;
        f32 m_threshold;        
        ObbData m_detector_volume;

        ui8 m_record_option;
        ui8 m_record_scatter;
        f32matrix44 m_transform;
        GlobalSimulationParametersData *mh_params;
};

#endif

