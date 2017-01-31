// GGEMS Copyright (C) 2017

/*!
 * \file template_detector.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.2
 * \date 02/03/2016
 *
 * v0.2: JB - Change all structs and remove CPU exec
 *
 */

#ifndef TEMPLATE_DETECTOR_CUH
#define TEMPLATE_DETECTOR_CUH

#include "global.cuh"
#include "raytracing.cuh"
#include "particles.cuh"
#include "ggems_detector.cuh"

class GGEMSDetector;

class TemplateDetector : public GGEMSDetector
{
    public:
        TemplateDetector();
        ~TemplateDetector();

        // Setting
        void set_dimension( f32 width, f32 height, f32 depth );

        // Mandatory functions from abstract class GGEMSDetector
        void initialize( GlobalSimulationParametersData *h_params );   // Initialisation
        void track_to_in( ParticlesData *d_particles );                // Navigation until the detector
        void track_to_out( ParticlesData *d_particles );               // Navigation within the detector
        void digitizer( ParticlesData *d_particles );                  // Hits processing into data (histo, image, etc.)

        // Save data
        void save_data( std::string filename );

    private:
        bool m_check_mandatory();
        f32 m_width, m_height, m_depth;
        GlobalSimulationParametersData *mh_params;

};

#endif
