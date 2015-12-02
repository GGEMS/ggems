// GGEMS Copyright (C) 2015

/*!
 * \file detector_manager.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 2 december 2015
 *
 *
 *
 */

#ifndef DETECTORMANAGER_CUH
#define DETECTORMANAGER_CUH

#include "global.cuh"
#include "particles.cuh"

class DetectorManager {
    public:
        DetectorManager();
        ~DetectorManager() {}

        //void set_detector(VoxPhanImgNav &aPhantom);
        void initialize(GlobalSimulationParameters params);

        void track_to_in(Particles particles);
        void track_to_out(Particles particles);

        void digitizer();
        void save_data(std::string filename);

        std::string get_detector_name();

    private:
        //VoxPhanImgNav m_vox_phan_img;
        std::string m_detector_name;

};

#endif
