#ifndef GUARD_GGEMS_PHYSICS_GGEMSMUDATA_HH
#define GUARD_GGEMS_PHYSICS_GGEMSMUDATA_HH

#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
//#include "GGEMS/physics/GGEMSMuDataConstants.hh"
#include "GGEMS/physics/GGEMSProcessConstants.hh"

// Mu and Mu_en table used by TLE
typedef struct GGEMSMuMuEnData_t {
    GGfloat E_bins[MAX_CROSS_SECTION_TABLE_NUMBER_BINS];      // n CROSS_SECTION_TABLE_NUMBER_BINS
    GGfloat mu[256*MAX_CROSS_SECTION_TABLE_NUMBER_BINS];          // n*k GGEMSMuDataConstants::kMuNbEnergies
    GGfloat mu_en[256*MAX_CROSS_SECTION_TABLE_NUMBER_BINS];       // n*k GGEMSMuDataConstants::kMuNbEnergies

    GGint nb_mat;      // k
    GGint nb_bins;     // n

    GGfloat E_min;
    GGfloat E_max;
}GGEMSMuMuEnData;

#endif
