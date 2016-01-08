// GGEMS Copyright (C) 2015

#ifndef ELECTRON_H
#define ELECTRON_H

#include "global.cuh"
#include "materials.cuh"
#include "image_reader.cuh"
#include "fun.cuh"
#include "particles.cuh"
#include "prng.cuh"
#include "dose_calculator.cuh"
#include "voxelized.cuh"
#include "raytracing.cuh"

#ifndef CROSSSECTIONTABLEELECTRONS
#define CROSSSECTIONTABLEELECTRONS

#define DDR3(a)  printf("[\033[34;01mDEBUG\033[00m] %s %e %e %e\n",#a,a.x,a.y,a.z);
#define DDR(a)  printf("[\033[34;01mDEBUG\033[00m] %s %e \n",#a,a);
#define DDI(a)  printf("[\033[34;01mDEBUG\033[00m] %s %d \n",#a,a);

// Cross section table for electron and positrons
struct ElectronsCrossSectionTable
{
    f32* E;                   // n*k
    f32* eIonisationdedx;     // n*k
    f32* eIonisationCS;       // n*k
    f32* eBremdedx;           // n*k
    f32* eBremCS;             // n*k
    f32* eMSC;                // n*k
    f32* eRange;              // n*k
    f32* pIonisationdedx;     // n*k
    f32* pIonisationCS;       // n*k
    f32* pBremdedx;           // n*k
    f32* pBremCS;             // n*k
    f32* pMSC;                // n*k
    f32* pRange;              // n*k
    f32* pAnni_CS_tab;        // n*k For positrons
    f32 E_min;
    f32 E_max;
    ui32 nb_bins;       // n
    ui32 nb_mat;        // k
    f32 cutEnergyElectron;
    f32 cutEnergyGamma;

    void operator= ( ElectronsCrossSectionTable tab1 )
    {
        E_min = tab1.E_min;
        E_max = tab1.E_max;
        nb_bins = tab1.nb_bins;
        nb_mat = tab1.nb_mat;
        cutEnergyElectron = tab1.cutEnergyElectron;
        cutEnergyGamma = tab1.cutEnergyGamma;


        E = new f32[nb_mat *nb_bins];  //Mandatories tables
        eRange = new f32[nb_mat *nb_bins];

        eIonisationCS = new f32[nb_mat * nb_bins];
        eIonisationdedx= new f32[nb_mat * nb_bins];

        eBremCS = new f32[nb_mat * nb_bins];
        eBremdedx= new f32[nb_mat * nb_bins];

        eMSC = new f32[nb_mat * nb_bins];

        pAnni_CS_tab = new f32[nb_mat * nb_bins];

        for ( ui32 i = 0; i < nb_bins * nb_mat; i++ )
        {
            E[i] = tab1.E[i];
            eRange[i] = tab1.eRange[i];

            eIonisationCS[i] = tab1.eIonisationCS[i];
            eIonisationdedx[i] = tab1.eIonisationdedx[i];

            eBremCS[i] = tab1.eBremCS[i];
            eBremdedx[i] = tab1.eBremdedx[i];

            eMSC[i] = tab1.eMSC[i];

            pAnni_CS_tab[i] = tab1.pAnni_CS_tab[i];
        }


    }

};

struct ElectronsCrossSection
{
    ElectronsCrossSectionTable data_h;
    ElectronsCrossSectionTable data_d;

};

#endif




// // Struct that handle CPU&GPU CS data
// struct ElectronCrossSection {
//     ElectronsCrossSectionTable data_h;
//     ElectronsCrossSectionTable data_d;
//
//     ui32 nb_bins;         // n
//     ui32 nb_mat;          // k
// };

class ElectronCrossSection
{
public :
    ElectronCrossSection() {}
    ~ElectronCrossSection() {}

    void initialize ( GlobalSimulationParameters params,MaterialsTable materials );

    void generateTable();

    void printElectronTables ( std::string );
    
    void m_copy_cs_table_cpu2gpu();

    inline ElectronsCrossSectionTable get_data_h()
    {
        return data_h;
    }
    inline ElectronsCrossSectionTable get_data_d()
    {
        return data_d;
    }

    ElectronsCrossSectionTable data_h;
    ElectronsCrossSectionTable data_d;

private :

    void Energy_table();
    void Range_table ( int id_mat );
    f32 GetDedx ( f32 Energy,int material );


    // eIoni functions
    void eIoni_DEDX_table ( int id_mat );
    f32 eIoniDEDX ( f32 Ekine,int id_mat );
    f32 DensCorrection ( f32 x, int id_mat );
    void eIoni_CrossSection_table ( int id_mat );
    f32 eIoniCrossSection ( int id_mat, f32 Ekine );
    f32 eIoniCrossSectionPerAtom ( int index, f32 Ekine );

    // Bremsthralung functions
    void eBrem_DEDX_table ( int id_mat );
    f32 eBremDEDX ( f32 Ekine,int id_mat );
    f32 eBremLoss ( f32 Z,f32 T,f32 Cut );
    void eBrem_CrossSection_table ( int );
    f32 eBremCrossSection ( f32 ,int );
    f32 eBremCrossSectionPerVolume ( f32 , int );
    f32 eBremCrossSectionPerAtom ( f32, f32, f32 );

    // MSC functions
    void eMSC_CrossSection_table ( int id_mat );
    f32 eMscCrossSection ( f32 Ekine, int id_mat );
    f32 eMscCrossSectionPerAtom ( f32 Ekine,  unsigned short int AtomNumber );



    ui32 nb_bins;         // n
    ui32 nb_mat;          // k
    f32 cutEnergyElectron;
    f32 cutEnergyGamma;

    f32 MaxKinEnergy;
    f32 MinKinEnergy;

    GlobalSimulationParameters parameters;
    MaterialsTable myMaterials;


    // Constant parameters for bremstrahlung table
    const f32 ZZ[ 8 ] = { 2., 4., 6., 14., 26., 50., 82., 92. };
    const f32 coefloss[ 8 ][ 11 ] =
    {
        { .98916, .47564, -.2505, -.45186, .14462, .21307, -.013738, -.045689, -.0042914, .0034429, .00064189 },
        { 1.0626, .37662, -.23646, -.45188, .14295, .22906, -.011041, -.051398, -.0055123, .0039919, .00078003 },
        { 1.0954, .315, -.24011, -.43849, .15017, .23001, -.012846, -.052555, -.0055114, .0041283, .00080318 },
        { 1.1649, .18976, -.24972, -.30124, .1555, .13565, -.024765, -.027047, -.00059821, .0019373, .00027647 },
        { 1.2261, .14272, -.25672, -.28407, .13874, .13586, -.020562, -.026722, -.00089557, .0018665, .00026981 },
        { 1.3147, .020049, -.35543, -.13927, .17666, .073746, -.036076, -.013407, .0025727, .00084005, -1.4082e-05 },
        { 1.3986, -.10586, -.49187, -.0048846, .23621, .031652, -.052938, -.0076639, .0048181, .00056486, -.00011995 },
        { 1.4217, -.116, -.55497, -.044075, .27506, .081364, -.058143, -.023402, .0031322, .0020201, .00017519 }
    };

    const f32 coefsig[ 8 ][ 11 ] =
    {
        { .4638, .37748, .32249, -.060362, -.065004, -.033457, -.004583, .011954, .0030404, -.0010077, -.00028131},
        { .50008, .33483, .34364, -.086262, -.055361, -.028168, -.0056172, .011129, .0027528, -.00092265, -.00024348},
        { .51587, .31095, .34996, -.11623, -.056167, -.0087154, .00053943, .0054092, .00077685, -.00039635, -6.7818e-05},
        { .55058, .25629, .35854, -.080656, -.054308, -.049933, -.00064246, .016597, .0021789, -.001327, -.00025983},
        { .5791, .26152, .38953, -.17104, -.099172, .024596, .023718, -.0039205, -.0036658, .00041749, .00023408},
        { .62085, .27045, .39073, -.37916, -.18878, .23905, .095028, -.068744, -.023809, .0062408, .0020407},
        { .66053, .24513, .35404, -.47275, -.22837, .35647, .13203, -.1049, -.034851, .0095046, .0030535},
        { .67143, .23079, .32256, -.46248, -.20013, .3506, .11779, -.1024, -.032013, .0092279, .0028592}
    };

    // constants for eMSC
    const f32 Zdat[ 15 ] = { 4., 6., 13., 20., 26., 29., 32., 38., 47., 50., 56., 64., 74., 79., 82. };

    const f32 Tdat[ 22 ] =
    {
        100.*eV, 200.*eV, 400.*eV, 700.*eV, 1.*keV, 2.*keV, 4.*keV, 7.*keV,
        10.*keV, 20.*keV, 40.*keV, 70.*keV, 100.*keV, 200.*keV, 400.*keV, 700.*keV,
        1.*MeV, 2.*MeV, 4.*MeV, 7.*MeV, 10.*MeV, 20.*MeV
    };

    const f32 celectron[ 15 ][ 22 ] =
    {
        { 1.125, 1.072, 1.051, 1.047, 1.047, 1.050, 1.052, 1.054, 1.054, 1.057, 1.062, 1.069, 1.075, 1.090, 1.105, 1.111, 1.112, 1.108, 1.100, 1.093, 1.089, 1.087 },
        { 1.408, 1.246, 1.143, 1.096, 1.077, 1.059, 1.053, 1.051, 1.052, 1.053, 1.058, 1.065, 1.072, 1.087, 1.101, 1.108, 1.109, 1.105, 1.097, 1.090, 1.086, 1.082 },
        { 2.833, 2.268, 1.861, 1.612, 1.486, 1.309, 1.204, 1.156, 1.136, 1.114, 1.106, 1.106, 1.109, 1.119, 1.129, 1.132, 1.131, 1.124, 1.113, 1.104, 1.099, 1.098 },
        { 3.879, 3.016, 2.380, 2.007, 1.818, 1.535, 1.340, 1.236, 1.190, 1.133, 1.107, 1.099, 1.098, 1.103, 1.110, 1.113, 1.112, 1.105, 1.096, 1.089, 1.085, 1.098 },
        { 6.937, 4.330, 2.886, 2.256, 1.987, 1.628, 1.395, 1.265, 1.203, 1.122, 1.080, 1.065, 1.061, 1.063, 1.070, 1.073, 1.073, 1.070, 1.064, 1.059, 1.056, 1.056 },
        { 9.616, 5.708, 3.424, 2.551, 2.204, 1.762, 1.485, 1.330, 1.256, 1.155, 1.099, 1.077, 1.070, 1.068, 1.072, 1.074, 1.074, 1.070, 1.063, 1.059, 1.056, 1.052 },
        { 11.72, 6.364, 3.811, 2.806, 2.401, 1.884, 1.564, 1.386, 1.300, 1.180, 1.112, 1.082, 1.073, 1.066, 1.068, 1.069, 1.068, 1.064, 1.059, 1.054, 1.051, 1.050 },
        { 18.08, 8.601, 4.569, 3.183, 2.662, 2.025, 1.646, 1.439, 1.339, 1.195, 1.108, 1.068, 1.053, 1.040, 1.039, 1.039, 1.039, 1.037, 1.034, 1.031, 1.030, 1.036 },
        { 18.22, 1.48, 5.333, 3.713, 3.115, 2.367, 1.898, 1.631, 1.498, 1.301, 1.171, 1.105, 1.077, 1.048, 1.036, 1.033, 1.031, 1.028, 1.024, 1.022, 1.021, 1.024 },
        { 14.14, 10.65, 5.710, 3.929, 3.266, 2.453, 1.951, 1.669, 1.528, 1.319, 1.178, 1.106, 1.075, 1.040, 1.027, 1.022, 1.020, 1.017, 1.015, 1.013, 1.013, 1.020 },
        { 14.11, 11.73, 6.312, 4.240, 3.478, 2.566, 2.022, 1.720, 1.569, 1.342, 1.186, 1.102, 1.065, 1.022, 1.003, 0.997, 0.995, 0.993, 0.993, 0.993, 0.993, 1.011 },
        { 22.76, 20.01, 8.835, 5.287, 4.144, 2.901, 2.219, 1.855, 1.677, 1.410, 1.224, 1.121, 1.073, 1.014, 0.986, 0.976, 0.974, 0.972, 0.973, 0.974, 0.975, 0.987 },
        { 50.77, 40.85, 14.13, 7.184, 5.284, 3.435, 2.520, 2.059, 1.837, 1.512, 1.283, 1.153, 1.091, 1.010, 0.969, 0.954, 0.950, 0.947, 0.949, 0.952, 0.954, 0.963 },
        { 65.87, 59.06, 15.87, 7.570, 5.567, 3.650, 2.682, 2.182, 1.939, 1.579, 1.325, 1.178, 1.108, 1.014, 0.965, 0.947, 0.941, 0.938, 0.940, 0.944, 0.946, 0.954 },
        { 55.60, 47.34, 15.92, 7.810, 5.755, 3.767, 2.760, 2.239, 1.985, 1.609, 1.343, 1.188, 1.113, 1.013, 0.960, 0.939, 0.933, 0.930, 0.933, 0.936, 0.939, 0.949 }
    };

    const f32 sig0[ 15 ] =
    {
        .2672*barn, .5922*barn, 2.653*barn, 6.235*barn, 11.69*barn, 13.24*barn, 16.12*barn, 23.00*barn,
        35.13*barn, 39.95*barn, 50.85*barn, 67.19*barn, 91.15*barn, 104.4*barn, 113.1*barn
    };

    const f32 hecorr[ 15 ] =
    {
        120.70, 117.50, 105.00, 92.92, 79.23, 74.510, 68.29, 57.39, 41.97, 36.14, 24.53, 10.21, -7.855, -16.84, -22.30
    };


};

#endif
