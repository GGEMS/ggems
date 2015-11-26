// GGEMS Copyright (C) 2015

#ifndef SOURCEMANAGER_CUH
#define SOURCEMANAGER_CUH

#include "global.cuh"
#include "point_source.cuh"

class SourcesManager {
    public:
        SourcesManager();
        ~SourcesManager() {}

        void set_source(PointSource &aSource);
        void initialize(GlobalSimulationParameters params);

        void get_primaries_generator(Particles particles);
        std::string get_source_name();

    private:
        PointSource m_point_source;
        std::string m_source_name;

};

#endif
