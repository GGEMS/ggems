#include "truebeam_source.cuh"

// GGEMS Copyright (C) 2015

/*!
/*!
 * \file truebeam_source.cu
 * \brief Novalis Truebeam source model
 * \author Yannick Lemaréchal <yannick.lemarechal@gmail.com>
 * \version 0.1
 * \date Friday January 26, 2015
*/


TruebeamSource::TruebeamSource()
{
    ;
}

TruebeamSource::~TruebeamSource()
{
    ;
}


void copy_TrueBeam_source_to_device(TrueBeamSourceData& h_TrueBeam,TrueBeamSourceData& d_TrueBeam)
    {
    d_TrueBeam.max_ind_rho =            h_TrueBeam.max_ind_rho;
    d_TrueBeam.max_ind_rhoet =          h_TrueBeam.max_ind_rhoet;
    d_TrueBeam.max_ind_rhotp =          h_TrueBeam.max_ind_rhotp;
    d_TrueBeam.max_ind_rhotp_511 =      h_TrueBeam.max_ind_rhotp_511;
    d_TrueBeam.max_ind_position =       h_TrueBeam.max_ind_position;
    d_TrueBeam.max_ind_position_511 =   h_TrueBeam.max_ind_position_511;
//     d_TrueBeam.mFlatteningFilter =      h_TrueBeam.mFlatteningFilter;
//     d_TrueBeam.mSourceEnergy =          h_TrueBeam.mSourceEnergy;

//    std::string mPathToData,mModel;  // useless in gpu
    d_TrueBeam.pas_pos =        h_TrueBeam.pas_pos;
    d_TrueBeam.pas_ener =       h_TrueBeam.pas_ener;
    d_TrueBeam.pas_theta =      h_TrueBeam.pas_theta;
    d_TrueBeam.pas_phi =        h_TrueBeam.pas_phi;
    d_TrueBeam.pas_pos_511 =    h_TrueBeam.pas_pos_511;
    d_TrueBeam.pas_theta_511 =  h_TrueBeam.pas_theta_511;
    d_TrueBeam.pas_phi_511 =    h_TrueBeam.pas_phi_511;

    d_TrueBeam.Tx_photon =  h_TrueBeam.Tx_photon;
    d_TrueBeam.Tx_511 =     h_TrueBeam.Tx_511;

    // For vector size
    d_TrueBeam.Vecteur_Position_size =          h_TrueBeam.Vecteur_Position_size;
    d_TrueBeam.Vecteur_Rho_Energy_size =        h_TrueBeam.Vecteur_Rho_Energy_size;
    d_TrueBeam.Vecteur_RhoET_size =             h_TrueBeam.Vecteur_RhoET_size;
    d_TrueBeam.Vecteur_RhoTP_size =             h_TrueBeam.Vecteur_RhoTP_size;
    d_TrueBeam.Vecteur_Position_511_size =      h_TrueBeam.Vecteur_Position_511_size;
    d_TrueBeam.Vecteur_RTP_511_size =           h_TrueBeam.Vecteur_RTP_511_size;
    d_TrueBeam.Vecteur_Rho_indice_size =        h_TrueBeam.Vecteur_Rho_indice_size;
    d_TrueBeam.Vecteur_RhoET_indice_size =      h_TrueBeam.Vecteur_RhoET_indice_size;
    d_TrueBeam.Vecteur_RhoET2_indice_size =     h_TrueBeam.Vecteur_RhoET2_indice_size;
    d_TrueBeam.Vecteur_RhoTP_indice_size =      h_TrueBeam.Vecteur_RhoTP_indice_size;
    d_TrueBeam.Vecteur_RhoTP2_indice_size =     h_TrueBeam.Vecteur_RhoTP2_indice_size;
    d_TrueBeam.Vecteur_RhoTP_indice_511_size =  h_TrueBeam.Vecteur_RhoTP_indice_511_size;

    HANDLE_ERROR(cudaMemcpy(d_TrueBeam.Vecteur_Position,            h_TrueBeam.Vecteur_Position,            h_TrueBeam.Vecteur_Position_size *          sizeof(f32xyz),  cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_TrueBeam.Vecteur_Rho_Energy,          h_TrueBeam.Vecteur_Rho_Energy,          h_TrueBeam.Vecteur_Rho_Energy_size *        sizeof(f32xyz),  cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_TrueBeam.Vecteur_RhoET,               h_TrueBeam.Vecteur_RhoET,               h_TrueBeam.Vecteur_RhoET_size *             sizeof(f32xyzw), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_TrueBeam.Vecteur_RhoTP,               h_TrueBeam.Vecteur_RhoTP,               h_TrueBeam.Vecteur_RhoTP_size *             sizeof(f32xyzw), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_TrueBeam.Vecteur_Position_511,        h_TrueBeam.Vecteur_Position_511,        h_TrueBeam.Vecteur_Position_511_size *      sizeof(f32xyz),  cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_TrueBeam.Vecteur_RTP_511,             h_TrueBeam.Vecteur_RTP_511,             h_TrueBeam.Vecteur_RTP_511_size *           sizeof(f32xyzw), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMemcpy(d_TrueBeam.Vecteur_Rho_indice,          h_TrueBeam.Vecteur_Rho_indice,          h_TrueBeam.Vecteur_Rho_indice_size *        sizeof(f32xy),   cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_TrueBeam.Vecteur_RhoET_indice,        h_TrueBeam.Vecteur_RhoET_indice,        h_TrueBeam.Vecteur_RhoET_indice_size *      sizeof(f32xy),   cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_TrueBeam.Vecteur_RhoET2_indice,       h_TrueBeam.Vecteur_RhoET2_indice,       h_TrueBeam.Vecteur_RhoET2_indice_size *     sizeof(f32xy),   cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_TrueBeam.Vecteur_RhoTP_indice,        h_TrueBeam.Vecteur_RhoTP_indice,        h_TrueBeam.Vecteur_RhoTP_indice_size *      sizeof(f32xy),   cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_TrueBeam.Vecteur_RhoTP2_indice,       h_TrueBeam.Vecteur_RhoTP2_indice,       h_TrueBeam.Vecteur_RhoTP2_indice_size *     sizeof(f32xy),   cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_TrueBeam.Vecteur_RhoTP_indice_511,    h_TrueBeam.Vecteur_RhoTP_indice_511,    h_TrueBeam.Vecteur_RhoTP_indice_511_size *  sizeof(f32xy),   cudaMemcpyHostToDevice));

    }


void init_TrueBeam_source_device(TrueBeamSourceData& h_TrueBeam,TrueBeamSourceData& d_TrueBeam)
    {

    HANDLE_ERROR(cudaMalloc((void**) &d_TrueBeam.Vecteur_Position,          h_TrueBeam.Vecteur_Position_size *          sizeof(f32xyz)));
    HANDLE_ERROR(cudaMalloc((void**) &d_TrueBeam.Vecteur_Rho_Energy,        h_TrueBeam.Vecteur_Rho_Energy_size *        sizeof(f32xyz)));
    HANDLE_ERROR(cudaMalloc((void**) &d_TrueBeam.Vecteur_RhoET,             h_TrueBeam.Vecteur_RhoET_size *             sizeof(f32xyzw)));
    HANDLE_ERROR(cudaMalloc((void**) &d_TrueBeam.Vecteur_RhoTP,             h_TrueBeam.Vecteur_RhoTP_size *             sizeof(f32xyzw)));
    HANDLE_ERROR(cudaMalloc((void**) &d_TrueBeam.Vecteur_Position_511,      h_TrueBeam.Vecteur_Position_511_size *      sizeof(f32xyz)));
    HANDLE_ERROR(cudaMalloc((void**) &d_TrueBeam.Vecteur_RTP_511,           h_TrueBeam.Vecteur_RTP_511_size *           sizeof(f32xyzw)));

    HANDLE_ERROR(cudaMalloc((void**) &d_TrueBeam.Vecteur_Rho_indice,        h_TrueBeam.Vecteur_Rho_indice_size *        sizeof(f32xy)));
    HANDLE_ERROR(cudaMalloc((void**) &d_TrueBeam.Vecteur_RhoET_indice,      h_TrueBeam.Vecteur_RhoET_indice_size *      sizeof(f32xy)));
    HANDLE_ERROR(cudaMalloc((void**) &d_TrueBeam.Vecteur_RhoET2_indice,     h_TrueBeam.Vecteur_RhoET2_indice_size *     sizeof(f32xy)));
    HANDLE_ERROR(cudaMalloc((void**) &d_TrueBeam.Vecteur_RhoTP_indice,      h_TrueBeam.Vecteur_RhoTP_indice_size *      sizeof(f32xy)));
    HANDLE_ERROR(cudaMalloc((void**) &d_TrueBeam.Vecteur_RhoTP2_indice,     h_TrueBeam.Vecteur_RhoTP2_indice_size *     sizeof(f32xy)));
    HANDLE_ERROR(cudaMalloc((void**) &d_TrueBeam.Vecteur_RhoTP_indice_511,  h_TrueBeam.Vecteur_RhoTP_indice_511_size *  sizeof(f32xy)));
    }
    
    
    
void TruebeamSource::initialize(GlobalSimulationParameters params)
{
    m_params = params;
    data_h.Vecteur_Position_size = 1000001;
    data_h.Vecteur_Rho_Energy_size= 1000001;
    data_h.Vecteur_RhoET_size= 15625001;
    data_h.Vecteur_RhoTP_size= 15625001;
    data_h.Vecteur_Position_511_size= 1000001;
    data_h.Vecteur_RTP_511_size= 1000001;
 
    data_h.Vecteur_Rho_indice_size= 1001;
    data_h.Vecteur_RhoET_indice_size= 1001;
    data_h.Vecteur_RhoET2_indice_size=1000001;
    data_h.Vecteur_RhoTP_indice_size = 1001;
    data_h.Vecteur_RhoTP2_indice_size =1000001;
    data_h.Vecteur_RhoTP_indice_511_size = 101;

    data_h.Vecteur_Position =        new f32xyz [data_h.Vecteur_Position_size];
    data_h.Vecteur_Rho_Energy=       new f32xyz [data_h.Vecteur_Rho_Energy_size];
    data_h.Vecteur_RhoET=            new f32xyzw[data_h.Vecteur_RhoET_size];
    data_h.Vecteur_RhoTP=            new f32xyzw[data_h.Vecteur_RhoTP_size];
    data_h.Vecteur_Position_511=     new f32xyz [data_h.Vecteur_Position_511_size];
    data_h.Vecteur_RTP_511=          new f32xyzw[data_h.Vecteur_RTP_511_size];
                                                    
    data_h.Vecteur_Rho_indice=       new f32xy[data_h.Vecteur_Rho_indice_size];
    data_h.Vecteur_RhoET_indice=     new f32xy[data_h.Vecteur_RhoET_indice_size];
    data_h.Vecteur_RhoET2_indice=    new f32xy[data_h.Vecteur_RhoET2_indice_size];
    data_h.Vecteur_RhoTP_indice=     new f32xy[data_h.Vecteur_RhoTP_indice_size];
    data_h.Vecteur_RhoTP2_indice=    new f32xy[data_h.Vecteur_RhoTP2_indice_size];
    data_h.Vecteur_RhoTP_indice_511= new f32xy[data_h.Vecteur_RhoTP_indice_511_size];
    
    
    
    if( m_params.data_h.device_target == GPU_DEVICE)
    {
        init_TrueBeam_source_device(data_h,data_d);
        copy_TrueBeam_source_to_device(data_h,data_d);
    }

}

void TruebeamSource::set_energy(i32 E)
{
    if( ( E != 6 ) && ( E != 10 ) )
    {
        print_error("Energy Truebeam available are 6 or 10 MeV");
        exit_simulation();
    }
    mSourceEnergy = E;

}

void TruebeamSource::set_flattening_filter(bool b)
{
    mFlatteningFilter = b;

}