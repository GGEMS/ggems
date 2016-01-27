#include "truebeam_source.cuh"
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <iostream>

using namespace std;

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
    m_path_to_data = "";
    m_source_energy = 0;
    m_flattening_filter = false;
}

TruebeamSource::~TruebeamSource()
{
    ;
}


void TruebeamSource::get_source_parameters()
    {

    std::string name_file,data_file,str_energy,str_fff;
    int i,j,k;
    f32 tmp1,tmp2,tmp3,tmp4,tmp12,tmp22,tmp32,tmp42,indice_tampon,indice_tampon2;
    float ftmp1,ftmp2,ftmp3,ftmp4;

    if(m_source_energy==6&&m_flattening_filter==true)
        {
        data_h.Tx_photon=2513248823.;
        data_h.Tx_511=10057967.;
        }
    if(m_source_energy==6&&m_flattening_filter==false)
        {
        data_h.Tx_photon=2577964273.;
        data_h.Tx_511=7550085.;
        }
    if(m_source_energy==10&&m_flattening_filter==true)
        {
        data_h.Tx_photon=2434000351.;
        data_h.Tx_511=16203030.;
        }
    if(m_source_energy==10&&m_flattening_filter==false)
        {
        data_h.Tx_photon=2407580032.;
        data_h.Tx_511=13665070.;
        }



    std::stringstream ss;
    ss<<m_source_energy;
    str_energy=ss.str();
    if(m_flattening_filter==true)
        str_fff="X";
    else
        str_fff="FFF";
    data_file=m_path_to_data+"TrueBeam_"+str_energy+str_fff;

//     print_information("TrueBeam energy : " + ss.str() + str_fff);
//     print_information("Read truebeam files : " + data_file + "*");

//   print_information("TrueBeam filter : " + ss.str() + "MeV");

    name_file=data_file+"_Rho_Energy.bin";
    ifstream Rho_energy_file(name_file.c_str(),ios::in|ios::binary);

    name_file=data_file+"_Rho_Ener_Theta.bin";
    ifstream RET_file(name_file.c_str(),ios::in|ios::binary);

    name_file=data_file+"_Rho_Theta_Phi.bin";
    ifstream RTP_file(name_file.c_str(),ios::in|ios::binary);

    name_file=data_file+"_Rho_Theta_Phi_511.bin";
    ifstream Rho_theta_phi_file_511(name_file.c_str(),ios::in|ios::binary);

    name_file=data_file+"_Position.bin";
    ifstream Position_file(name_file.c_str(),ios::in|ios::binary);

    name_file=data_file+"_Position_511.bin";
    ifstream Position_511_file(name_file.c_str(),ios::in|ios::binary);

    if(!Rho_energy_file)
        {
        printf("Pas de fichier de rho versus energie\n");
        exit(EXIT_FAILURE);

        }
    else
        {
        i=0;
        j=0;
        tmp12=0.;
        tmp22=10.;
        tmp32=0.;
        data_h.pas_ener=0.;

//     while(true)
        for(;;)
            {
            Rho_energy_file.read((char *)&ftmp1,sizeof(float));
            tmp1=(f32)ftmp1;
            Rho_energy_file.read((char *)&ftmp2,sizeof(float));
            tmp2=(f32)ftmp2;
            Rho_energy_file.read((char *)&ftmp3,sizeof(float));
            tmp3=(f32)ftmp3;
            // Rho_energy_file >> tmp1 >> tmp2 >> tmp3;
            if(Rho_energy_file.eof()) break;
            if(tmp1==tmp12&&tmp2==tmp22&&tmp3==tmp32) break;
            if((data_h.pas_ener==0.||data_h.pas_ener>(tmp3-tmp32))&&tmp32<tmp3&&i!=0) data_h.pas_ener=tmp3-tmp32;
            data_h.Vecteur_Rho_Energy[i].x=tmp1;
            data_h.Vecteur_Rho_Energy[i].y=tmp2;
            data_h.Vecteur_Rho_Energy[i].z=tmp3;
            tmp12=tmp1;
            tmp22=tmp2;
            tmp32=tmp3;
            if(i==0)
                {
                data_h.Vecteur_Rho_indice[j].x=0.;
                data_h.Vecteur_Rho_indice[j].y=0.;
                j++;
                indice_tampon=tmp1;
                }
            if(indice_tampon!=tmp1)
                {
                data_h.Vecteur_Rho_indice[j].x=indice_tampon;
                indice_tampon=tmp1;
                data_h.Vecteur_Rho_indice[j].y=(f32)i;
                j++;
                }
            i++;
            }
        if(indice_tampon!=data_h.Vecteur_Rho_indice[0].x)
            {
            data_h.Vecteur_Rho_indice[j].x=indice_tampon;
            data_h.Vecteur_Rho_indice[j].y=(f32)i;
            }
        }
    data_h.max_ind_rho=j;
    Rho_energy_file.close();

    if(!RET_file)
        {
        printf("Pas de fichier de rho versus energy and theta\n");
        exit(EXIT_FAILURE);

        }
    else
        {
        i=0;
        j=0;
        k=0;
        tmp12=0.;
        tmp22=0.;
        tmp42=0.;
        tmp32=10.;
        data_h.pas_theta=0.;
//     while(true)
        for(;;)
            {
            RET_file.read((char *)&ftmp1,sizeof(float));
            tmp1=(f32)ftmp1;
            RET_file.read((char *)&ftmp2,sizeof(float));
            tmp2=(f32)ftmp2;
            RET_file.read((char *)&ftmp3,sizeof(float));
            tmp3=(f32)ftmp3;
            RET_file.read((char *)&ftmp4,sizeof(float));
            tmp4=(f32)ftmp4;
            // RET_file >> tmp1 >> tmp2 >> tmp3 >> tmp4;
            if(RET_file.eof()) break;
            if(tmp1==tmp12&&tmp2==tmp22&&tmp3==tmp32&&tmp4==tmp42) break;
            if((data_h.pas_theta==0.||data_h.pas_theta>(tmp4-tmp42))&&tmp42<tmp4&&i!=0) data_h.pas_theta=tmp4-tmp42;
            data_h.Vecteur_RhoET[i].x=tmp1;
            data_h.Vecteur_RhoET[i].y=tmp2;
            data_h.Vecteur_RhoET[i].z=tmp3;
            data_h.Vecteur_RhoET[i].w=tmp4;
            tmp12=tmp1;
            tmp22=tmp2;
            tmp32=tmp3;
            tmp42=tmp4;
            if(i==0)
                {
                data_h.Vecteur_RhoET_indice[j].x=0.;
                data_h.Vecteur_RhoET_indice[j].y=0.;
                data_h.Vecteur_RhoET2_indice[k].x=0.;
                data_h.Vecteur_RhoET2_indice[k].y=0.;
                j++;
                k++;
                indice_tampon=tmp1;
                indice_tampon2=tmp2;
                }
            if(indice_tampon!=tmp1)
                {
                data_h.Vecteur_RhoET_indice[j].x=tmp1;
                data_h.Vecteur_RhoET_indice[j].y=(f32)k;
                if(indice_tampon2!=tmp2)
                    {
                    data_h.Vecteur_RhoET2_indice[k].x=indice_tampon2;
                    data_h.Vecteur_RhoET2_indice[k].y=(f32)i;
                    indice_tampon2=tmp2;
                    k++;
                    }
                data_h.Vecteur_RhoET2_indice[k].x=0.;
                data_h.Vecteur_RhoET2_indice[k].y=(f32)i;
                indice_tampon=tmp1;
                indice_tampon2=tmp2;
                k++;
                j++;
                }
            else if(indice_tampon2!=tmp2)
                {
                data_h.Vecteur_RhoET2_indice[k].x=indice_tampon2;
                data_h.Vecteur_RhoET2_indice[k].y=(f32)i;
                indice_tampon2=tmp2;
                k++;
                }
            i++;
            }
        if(indice_tampon!=data_h.Vecteur_RhoET_indice[0].x)
            {
            data_h.Vecteur_RhoET_indice[j].x=indice_tampon;
            data_h.Vecteur_RhoET_indice[j].y=(f32)i;
            }
        }
    data_h.max_ind_rhoet=j;
    RET_file.close();

    if(!RTP_file)
        {
        printf("Pas de fichier de rho versus theta and phi\n");
        exit(EXIT_FAILURE);

        }
    else
        {
        i=0;
        j=0;
        k=0;
        tmp12=0.;
        tmp22=0.;
        tmp42=0.;
        tmp32=10.;
        data_h.pas_phi=0.;
//     while(true)
        for(;;)
            {
            RTP_file.read((char *)&ftmp1,sizeof(float));
            tmp1=(f32)ftmp1;
            RTP_file.read((char *)&ftmp2,sizeof(float));
            tmp2=(f32)ftmp2;
            RTP_file.read((char *)&ftmp3,sizeof(float));
            tmp3=(f32)ftmp3;
            RTP_file.read((char *)&ftmp4,sizeof(float));
            tmp4=(f32)ftmp4;
            // RTP_file >> tmp1 >> tmp2 >> tmp3 >> tmp4;
            if(RTP_file.eof()) break;
            if(tmp1==tmp12&&tmp2==tmp22&&tmp3==tmp32&&tmp4==tmp42) break;
            if((data_h.pas_phi==0.||data_h.pas_phi>(tmp4-tmp42))&&tmp42<tmp4&&i!=0) data_h.pas_phi=tmp4-tmp42;
            data_h.Vecteur_RhoTP[i].x=tmp1;
            data_h.Vecteur_RhoTP[i].y=tmp2;
            data_h.Vecteur_RhoTP[i].z=tmp3;
            data_h.Vecteur_RhoTP[i].w=tmp4;
            tmp12=tmp1;
            tmp22=tmp2;
            tmp32=tmp3;
            tmp42=tmp4;
            if(i==0)
                {
                data_h.Vecteur_RhoTP_indice[j].x=0.;
                data_h.Vecteur_RhoTP_indice[j].y=0.;
                data_h.Vecteur_RhoTP2_indice[k].x=0.;
                data_h.Vecteur_RhoTP2_indice[k].y=0.;
                j++;
                k++;
                indice_tampon=tmp1;
                indice_tampon2=tmp2;
                }
            if(indice_tampon!=tmp1)
                {
                data_h.Vecteur_RhoTP_indice[j].x=tmp1;
                data_h.Vecteur_RhoTP_indice[j].y=(f32)k;
                if(indice_tampon2!=tmp2)
                    {
                    data_h.Vecteur_RhoTP2_indice[k].x=indice_tampon2;
                    data_h.Vecteur_RhoTP2_indice[k].y=(f32)i;
                    indice_tampon2=tmp2;
                    k++;
                    }
                data_h.Vecteur_RhoTP2_indice[k].x=0.;
                data_h.Vecteur_RhoTP2_indice[k].y=(f32)i;
                indice_tampon=tmp1;
                indice_tampon2=tmp2;
                k++;
                j++;
                }
            else if(indice_tampon2!=tmp2)
                {
                data_h.Vecteur_RhoTP2_indice[k].x=indice_tampon2;
                data_h.Vecteur_RhoTP2_indice[k].y=(f32)i;
                indice_tampon2=tmp2;
                k++;
                }
            i++;
            }
        if(indice_tampon!=data_h.Vecteur_RhoTP_indice[0].x)
            {
            data_h.Vecteur_RhoTP_indice[j].x=indice_tampon;
            data_h.Vecteur_RhoTP_indice[j].y=(f32)i;
            }
        }
    data_h.max_ind_rhotp=j;
    RTP_file.close();

    if(!Rho_theta_phi_file_511)
        {
        printf("Pas de fichier de rho versus theta versus phi 511\n");
        exit(EXIT_FAILURE);

        }
    else
        {
        i=0;
        j=0;
        tmp12=0.;
        tmp32=0.;
        tmp42=0.;
        tmp22=10.;
        data_h.pas_theta_511=0.;
        data_h.pas_phi_511=0.;
//     while(true)
        for(;;)
            {
            Rho_theta_phi_file_511.read((char *)&ftmp1,sizeof(float));
            tmp1=(f32)ftmp1;
            Rho_theta_phi_file_511.read((char *)&ftmp2,sizeof(float));
            tmp2=(f32)ftmp2;
            Rho_theta_phi_file_511.read((char *)&ftmp3,sizeof(float));
            tmp3=(f32)ftmp3;
            Rho_theta_phi_file_511.read((char *)&ftmp4,sizeof(float));
            tmp4=(f32)ftmp4;
            // Rho_theta_phi_file_511 >> tmp1 >> tmp2 >> tmp3 >> tmp4;
            if(Rho_theta_phi_file_511.eof()) break;
            if(tmp1==tmp12&&tmp2==tmp22&&tmp3==tmp32&&tmp4==tmp42) break;
            if((data_h.pas_phi_511==0.||data_h.pas_phi_511>(tmp4-tmp42))&&tmp42<tmp4&&i!=0) data_h.pas_phi_511=tmp4-tmp42;
            if((data_h.pas_theta_511==0.||data_h.pas_theta_511>(tmp3-tmp32))&&tmp32<tmp3&&i!=0) data_h.pas_theta_511=tmp3-tmp32;
            // printf("%d %lf %lf %lf %lf\n",Rho_theta_phi_file_511.eof(),tmp1,tmp2,tmp3,tmp4);
            data_h.Vecteur_RTP_511[i].x=tmp1;
            data_h.Vecteur_RTP_511[i].y=tmp2;
            data_h.Vecteur_RTP_511[i].z=tmp3;
            data_h.Vecteur_RTP_511[i].w=tmp4;
            tmp12=tmp1;
            tmp22=tmp2;
            tmp32=tmp3;
            tmp42=tmp4;
            if(i==0)
                {
                data_h.Vecteur_RhoTP_indice_511[j].x=0.;
                data_h.Vecteur_RhoTP_indice_511[j].y=0.;
                j++;
                indice_tampon=tmp1;
                }
            if(indice_tampon!=tmp1)
                {
                data_h.Vecteur_RhoTP_indice_511[j].x=indice_tampon;
                indice_tampon=tmp1;
                data_h.Vecteur_RhoTP_indice_511[j].y=(f32)i;
                j++;
                }
            i++;
            }
        if(indice_tampon!=data_h.Vecteur_RhoTP_indice_511[0].x)
            {
            data_h.Vecteur_RhoTP_indice_511[j].x=indice_tampon;
            data_h.Vecteur_RhoTP_indice_511[j].y=(f32)i;
            }
        }
    data_h.max_ind_rhotp_511=j;
    Rho_theta_phi_file_511.close();

    if(!Position_file)
        {
        printf("Pas de fichier de positions\n");
        exit(EXIT_FAILURE);

        }
    else
        {
        i=0;
        tmp12=10.;
        tmp22=0.;
        tmp32=0.;
        data_h.pas_pos=0.;
//     while(true)
        for(;;)
            {
            Position_file.read((char *)&ftmp1,sizeof(float));
            tmp1=(f32)ftmp1;
            Position_file.read((char *)&ftmp2,sizeof(float));
            tmp2=(f32)ftmp2;
            Position_file.read((char *)&ftmp3,sizeof(float));
            tmp3=(f32)ftmp3;
            // Position_file >> tmp1 >> tmp2 >> tmp3;
            if(Position_file.eof()) break;
            if((data_h.pas_pos==0.||data_h.pas_pos>(tmp2-tmp22))&&tmp22<tmp2&&i!=0) data_h.pas_pos=tmp2-tmp22;
            if(tmp1==tmp12&&tmp2==tmp22&&tmp3==tmp32) break;
            data_h.Vecteur_Position[i].x=tmp1;
            data_h.Vecteur_Position[i].y=tmp2;
            data_h.Vecteur_Position[i].z=tmp3;
            tmp12=tmp1;
            tmp22=tmp2;
            tmp32=tmp3;
            i++;
            }
        }
    data_h.max_ind_position=i-1;
    Position_file.close();

    if(!Position_511_file)
        {
        printf("Pas de fichier de positions 511\n");
        exit(EXIT_FAILURE);

        }
    else
        {
        i=0;
        tmp12=10.;
        tmp22=0.;
        tmp32=0.;
        data_h.pas_pos_511=0.;
//     while(true)
        for(;;)
            {
            Position_511_file.read((char *)&ftmp1,sizeof(float));
            tmp1=(f32)ftmp1;
            Position_511_file.read((char *)&ftmp2,sizeof(float));
            tmp2=(f32)ftmp2;
            Position_511_file.read((char *)&ftmp3,sizeof(float));
            tmp3=(f32)ftmp3;
            // Position_511_file >> tmp1 >> tmp2 >> tmp3;
            if(Position_511_file.eof()) break;
            if((data_h.pas_pos_511==0.||data_h.pas_pos_511>(tmp2-tmp22))&&tmp22<tmp2&&i!=0) data_h.pas_pos_511=tmp2-tmp22;
            if(tmp1==tmp12&&tmp2==tmp22&&tmp3==tmp32) break;
            data_h.Vecteur_Position_511[i].x=tmp1;
            data_h.Vecteur_Position_511[i].y=tmp2;
            data_h.Vecteur_Position_511[i].z=tmp3;
            tmp12=tmp1;
            tmp22=tmp2;
            tmp32=tmp3;
            i++;
            }
        }
    data_h.max_ind_position_511=i-1;
    Position_511_file.close();

    GGcout<<"Pas pos "<<data_h.pas_pos<<"; Pas ener "<<data_h.pas_ener<<"; Pas theta "<<data_h.pas_theta<<"; Pas phi "<<data_h.pas_phi<<GGendl;
    GGcout<<"Pas pos 511 "<<data_h.pas_pos_511<<"; Pas theta 511 "<<data_h.pas_theta_511<<"; Pas phi 511 "<<data_h.pas_phi_511<<GGendl;


    }


void copy_TrueBeam_source_to_device(TrueBeamSourceData& h_TrueBeam,TrueBeamSourceData& d_TrueBeam)
    {
    d_TrueBeam.max_ind_rho =            h_TrueBeam.max_ind_rho;
    d_TrueBeam.max_ind_rhoet =          h_TrueBeam.max_ind_rhoet;
    d_TrueBeam.max_ind_rhotp =          h_TrueBeam.max_ind_rhotp;
    d_TrueBeam.max_ind_rhotp_511 =      h_TrueBeam.max_ind_rhotp_511;
    d_TrueBeam.max_ind_position =       h_TrueBeam.max_ind_position;
    d_TrueBeam.max_ind_position_511 =   h_TrueBeam.max_ind_position_511;
//     d_TrueBeam.m_flattening_filter =      h_TrueBeam.m_flattening_filter;
//     d_TrueBeam.m_source_energy =          h_TrueBeam.m_source_energy;

//    std::string m_path_to_data,mModel;  // useless in gpu
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
 
    if(m_source_energy == 0) 
    {
        throw std::runtime_error( "Missing energy parameters for truebeam source!!!" );
    }
 
    if(m_path_to_data == "") 
    {
        throw std::runtime_error( "Missing data path for truebeam source!!!" );
    }
 
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
    m_source_energy = E;
}

void TruebeamSource::set_flattening_filter(bool b)
{
    m_flattening_filter = b;
}

void TruebeamSource::set_path_to_data(std::string datapath)
{
    m_path_to_data = datapath;
}
