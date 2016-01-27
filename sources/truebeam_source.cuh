#ifndef TRUEBEAM_SOURCE_CUH
#define TRUEBEAM_SOURCE_CUH

// GGEMS Copyright (C) 2015

/*!
 * \file truebeam_source.cuh
 * \brief Novalis Truebeam source model
 * \author Yannick Lemaréchal <yannick.lemarechal@gmail.com>
 * \version 0.1
 * \date Friday January 26, 2015
*/

#include "global.cuh"
#include "particles.cuh"
#include "ggems_source.cuh"

#ifndef TRUEBEAMSOURCE
#define TRUEBEAMSOURCE
struct TrueBeamSourceData
    {  
    i32 max_ind_rho;
    i32 max_ind_rhoet;
    i32 max_ind_rhotp;
    i32 max_ind_rhotp_511;
    i32 max_ind_position;
    i32 max_ind_position_511;
    bool mFlatteningFilter;
    


    f32 pas_pos;
    f32 pas_ener;
    f32 pas_theta;
    f32 pas_phi;
    f32 pas_pos_511;
    f32 pas_theta_511;
    f32 pas_phi_511;

    f32 Tx_photon;
    f32 Tx_511;

    // For vector size
    ui32 Vecteur_Position_size;
    ui32 Vecteur_Rho_Energy_size;
    ui32 Vecteur_RhoET_size;
    ui32 Vecteur_RhoTP_size;
    ui32 Vecteur_Position_511_size;
    ui32 Vecteur_RTP_511_size;
    ui32 Vecteur_Rho_indice_size;
    ui32 Vecteur_RhoET_indice_size;
    ui32 Vecteur_RhoET2_indice_size;
    ui32 Vecteur_RhoTP_indice_size;
    ui32 Vecteur_RhoTP2_indice_size;
    ui32 Vecteur_RhoTP_indice_511_size;

    f32xyz *Vecteur_Position;
    f32xyz *Vecteur_Rho_Energy;
    f32xyzw *Vecteur_RhoET;
    f32xyzw *Vecteur_RhoTP;
    f32xyz *Vecteur_Position_511;
    f32xyzw *Vecteur_RTP_511;

    f32xy *Vecteur_Rho_indice;
    f32xy *Vecteur_RhoET_indice;
    f32xy *Vecteur_RhoET2_indice;
    f32xy *Vecteur_RhoTP_indice;
    f32xy *Vecteur_RhoTP2_indice;
    f32xy *Vecteur_RhoTP_indice_511;
    };
#endif


// class GGEMSSource;

/*!
  \class TruebeamSource
  \brief Class cone-beam source for composed by different methods to 
  characterize the source. In this source, the user can define a focal, non
  only a poi32 source.
*/
class TruebeamSource : public GGEMSSource
{
    public:
        /*!
        \brief TruebeamSource constructor
        */
        TruebeamSource();

        /*!
        \brief TruebeamSource destructor
        */
        ~TruebeamSource();

        
        void set_energy(i32 E);
        void set_flattening_filter(bool b);
        void get_primaries_generator(Particles particles) ;
        void initialize(GlobalSimulationParameters params) ;


        
    private:
    
    TrueBeamSourceData data_h;
    TrueBeamSourceData data_d;
    GlobalSimulationParameters m_params; /*!< Simulation parameters */
    i32 mSourceEnergy;
};

#endif

