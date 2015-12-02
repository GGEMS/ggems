// GGEMS Copyright (C) 2015

/*!
 * \file flatpanel_detector.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 2 december 2015
 *
 *
 *
 */

#ifndef FLATPANEL_DETECTOR_CUH
#define FLATPANEL_DETECTOR_CUH

#include "global.cuh"
#include "raytracing.cuh"
#include "particles.cuh"
#include "obb.cuh"

class FlatpanelDetector {
    public:
        FlatpanelDetector() {}
        ~FlatpanelDetector() {}

        // Setting


        // Tracking from outside to the detector
        void track_to_in(Particles particles);
        // Tracking inside the detector
        void track_to_out(Particles particles);

        // Init
        void initialize(GlobalSimulationParameters params);

        void digitizer();
        void save_data(std::string filename);

    private:
        bool m_check_mandatory();       
        void m_copy_detector_cpu2gpu();

        Obb phantom;

        GlobalSimulationParameters m_params;

};

#endif
