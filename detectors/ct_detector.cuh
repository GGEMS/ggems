// GGEMS Copyright (C) 2015

/*!
 * \file ct_detector.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 2 december 2015
 *
 *
 *
 */

#ifndef CT_DETECTOR_CUH
#define CT_DETECTOR_CUH

#include "global.cuh"
#include "raytracing.cuh"
#include "particles.cuh"
#include "obb.cuh"

class GGEMSDetector;

class CTDetector : public GGEMSDetector
{
    public:
        CTDetector();
        ~CTDetector();

        // Setting
        void set_dimension( f32 w, f32 h, f32 d );

        void set_pixel_size( f32 sx, f32 sy, f32 sz );
        void set_position( f32 x, f32 y, f32 z );
        void set_threshold( f32 threshold );
        void set_orbiting( f32 orbiting_angle );

        // Tracking from outside to the detector
        void track_to_in( Particles particles );
        void track_to_out( Particles particles ){}

        // Init
        void initialize( GlobalSimulationParameters params );

        void digitizer(){}
        void save_projection( std::string filename );

        void save_scatter( std::string basename );

        void printInfoDetection();

    private:
        ui32 getDetectedParticles();
        ui32 getScatterNumber( ui32 scatter_order );

    private:
        bool m_check_mandatory();
        void m_copy_detector_cpu2gpu();

        Obb m_detector_volume;
        f32 m_pixel_size_x, m_pixel_size_y, m_pixel_size_z;
        ui32 m_nb_pixel_x, m_nb_pixel_y, m_nb_pixel_z;
        f32 m_posx, m_posy, m_posz;
        f32 m_threshold;
        f32 m_orbiting_angle;

        ui32 *m_projection_h;
        ui32 *m_projection_d;
        ui32 *m_scatter_order_h;
        ui32 *m_scatter_order_d;

        GlobalSimulationParameters m_params;
};

#endif

