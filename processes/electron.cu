// GGEMS Copyright (C) 2015

#ifndef ELECTRON_CU
#define ELECTRON_CU
#include "electron.cuh"


void ElectronCrossSection::initialize ( GlobalSimulationParameters params,MaterialsTable materials )
{
    nb_bins = params.data_h.cs_table_nbins;
    nb_mat = materials.nb_materials;
//     params = parameters;
    parameters = params;
    myMaterials = materials;
    MaxKinEnergy = parameters.data_h.cs_table_max_E;
    MinKinEnergy = parameters.data_h.cs_table_min_E;
    cutEnergyElectron = parameters.data_h.electron_cut;
    cutEnergyGamma = parameters.data_h.photon_cut;
}


void ElectronCrossSection::generateTable()
{
    data_h.nb_bins = nb_bins;
    data_h.nb_mat = nb_mat;
    // Memory allocation for tables
    data_h.E = new f32[nb_mat *nb_bins];  //Mandatories tables
    data_h.eRange = new f32[nb_mat *nb_bins];

    data_h.eIonisationCS = new f32[nb_mat * nb_bins];
    data_h.eIonisationdedx= new f32[nb_mat * nb_bins];

    data_h.eBremCS = new f32[nb_mat * nb_bins];
    data_h.eBremdedx= new f32[nb_mat * nb_bins];

    data_h.eMSC = new f32[nb_mat * nb_bins];

    data_h.pAnni_CS_tab = new f32[nb_mat * nb_bins];


    for ( int i=0; i<nb_mat*nb_bins; i++ ) // Create Energy table between Emin and Emax
    {
        data_h.E[i]=0.;
    }


    Energy_table();


    for ( int id_mat = 0; id_mat< myMaterials.nb_materials; ++id_mat )
    {

//         int i = 0;
        for ( int i=0; i<nb_bins; i++ ) // Initialize tables
        {

            //                 if(m_physics_list[ELECTRON_IONISATION] == 1)
            //                 {
            data_h.eIonisationCS[i + nb_bins * id_mat]=0.;
            data_h.eIonisationdedx[i + nb_bins * id_mat]=0.;
            //                 }
            //                 if(m_physics_list[ELECTRON_BREMSSTRAHLUNG] == 1)
            //                 {
            data_h.eBremCS[i + nb_bins * id_mat]=0.;
            data_h.eBremdedx[i + nb_bins * id_mat]=0.;
            //                 }
            //                 if(m_physics_list[ELECTRON_MSC] == 1)
            //                 {
            data_h.eMSC[i + nb_bins * id_mat]=0.;
            //                 }

            data_h.eRange[i + nb_bins * id_mat]=0.;

            data_h.pAnni_CS_tab[i + nb_bins * id_mat]=0.;

        }


        // Create tables if physic is activated
        if ( parameters.data_h.physics_list[ELECTRON_IONISATION] == true )
        {
            eIoni_DEDX_table ( id_mat );
            eIoni_CrossSection_table ( id_mat );
        }

        if ( parameters.data_h.physics_list[ELECTRON_BREMSSTRAHLUNG] == true )
        {
            eBrem_DEDX_table ( id_mat );
            eBrem_CrossSection_table ( id_mat );
        }

        if ( parameters.data_h.physics_list[ELECTRON_MSC] == true )
        {
            eMSC_CrossSection_table ( id_mat );
        }


        Range_table ( id_mat );


    }

}

void ElectronCrossSection::Range_table ( int id_mat )
{

    int i,j,n;
    f32  energy,de,hsum,esum,eDXDE=0.,hDXDE=0.;

    i=0;
    n=100;


    eDXDE=data_h.eIonisationdedx[i+id_mat*nb_bins]+data_h.eBremdedx[i+id_mat*nb_bins];
    if ( eDXDE>0. )
        eDXDE=2.*data_h.E[i+id_mat*nb_bins]/eDXDE;
    data_h.eRange[i+id_mat*nb_bins]=eDXDE;

    for ( i=1; i<nb_bins; i++ )
    {
        de= ( data_h.E[i+id_mat*nb_bins]-data_h.E[i-1+id_mat*nb_bins] ) /n;
        energy=data_h.E[i+id_mat*nb_bins]+de*.5;

        esum=0.;
        for ( j=0; j<n; j++ )
        {
            energy-=de;
            eDXDE=GetDedx ( energy,id_mat ); //+GetDedx(energy,2);
            if ( eDXDE>0. )
                esum+=de/eDXDE;
        }
        data_h.eRange[i+id_mat*nb_bins]=data_h.eRange[i-1+id_mat*nb_bins]+esum;
    }



}

f32 ElectronCrossSection::GetDedx ( f32 Energy,int material ) // get dedx eioni and ebrem sum for energy Energy in material material
{
    int index = 0;
    index = binary_search ( Energy,data_h.E, ( material+1 ) *nb_bins, ( material ) *nb_bins );

    f32 DeDxeIoni = linear_interpolation ( data_h.E[index]  ,data_h.eIonisationdedx[index],
                                           data_h.E[index+1],data_h.eIonisationdedx[index+1],
                                           Energy
                                         );

    f32 DeDxeBrem = linear_interpolation ( data_h.E[index]  ,data_h.eBremdedx[index],
                                           data_h.E[index+1],data_h.eBremdedx[index+1],
                                           Energy
                                         );

    return DeDxeIoni + DeDxeBrem;

}

//Print a table file per material
void ElectronCrossSection::printElectronTables ( std::string dirname )
{

    for ( int i = 0; i< nb_mat; ++i )
    {
        std::string tmp = dirname + to_string ( i ) + ".txt";
        ImageReader::recordTables ( tmp.c_str(),i*nb_bins, ( i+1 ) *nb_bins,
                                    data_h.E,
                                    data_h.eIonisationdedx,
                                    data_h.eIonisationCS,
                                    data_h.eBremdedx,
                                    data_h.eBremCS,
                                    data_h.eMSC,
                                    data_h.eRange );
    }

}


void ElectronCrossSection::Energy_table() // Create energy table between emin and emax
{
//     int id_mat = myMaterials.nb_materials;
    int i;
    f32  constant,slope,x,energy;

    constant=parameters.data_h.cs_table_min_E;
    slope=log ( parameters.data_h.cs_table_max_E/parameters.data_h.cs_table_min_E );
    for ( int id_mat = 0; id_mat< myMaterials.nb_materials; ++id_mat )
        for ( i=0; i<nb_bins; i++ )
        {
            x= ( f32 ) i;
            x/= ( nb_bins-1 );
            data_h.E[i+id_mat*nb_bins]=constant*exp ( slope*x ) *MeV;
        }
}


void ElectronCrossSection:: eIoni_DEDX_table ( int id_mat )
{
    int i; // Index to increment energy

    for ( i=0; i<nb_bins; i++ )
    {

        f32 Ekine=data_h.E[i];
        data_h.eIonisationdedx[i + nb_bins * id_mat]=eIoniDEDX ( Ekine,id_mat );

    }

}

f32 ElectronCrossSection::eIoniDEDX ( f32 Ekine,int id_mat )
{

    f32  Dedx=0.;
    f32  th=.25*sqrt ( myMaterials.nb_electrons_per_vol[id_mat]/myMaterials.nb_atoms_per_vol[id_mat] ) *keV;
    f32  lowLimit=.2*keV;
    f32  tmax,tkin;
    f32  eexc,eexc2,d,x,y;
    f32  tau,gamma,gamma2,beta2,bg2;

    tkin=Ekine;
    if ( Ekine<th )
        tkin=th;
    tmax=tkin*.5;
    tau=tkin/electron_mass_c2;
    gamma=tau+1.;
    gamma2=gamma*gamma;
    beta2=1.-1./gamma2;
    bg2=beta2*gamma2;
    eexc=myMaterials.electron_mean_excitation_energy[id_mat]/electron_mass_c2;
    eexc2=eexc*eexc;
    d=std::min ( cutEnergyElectron,tmax );
    d/=electron_mass_c2;

    Dedx=log ( 2.* ( tau+2. ) /eexc2 )-1.-beta2+log ( ( tau-d ) *d ) +tau/ ( tau-d )
         + ( .5*d*d+ ( 2.*tau+1. ) *log ( 1.-d/tau ) ) /gamma2;

    x=log ( bg2 ) / ( 2.*log ( 10. ) );
    Dedx-=DensCorrection ( x,id_mat );
    Dedx*=twopi_mc2_rcl2*myMaterials.nb_electrons_per_vol[id_mat]/beta2;

    if ( Dedx<0. )
        Dedx=0.;
    if ( Ekine<th )
    {
        if ( Ekine>=lowLimit )
            Dedx*=sqrt ( tkin/Ekine );
        else
            Dedx*=sqrt ( tkin*Ekine ) /lowLimit;
    }

    return    Dedx;
}


f32 ElectronCrossSection::DensCorrection ( f32 x, int id_mat )
{
    f32  y=0.;

    if ( x<myMaterials.fX0[id_mat] )
    {
        if ( myMaterials.fD0[id_mat]>0. )
            y=myMaterials.fD0[id_mat]*pow ( 10.,2.* ( x-myMaterials.fX0[id_mat] ) );
    }
    else if ( x>=myMaterials.fX1[id_mat] )
        y=2.*log ( 10. ) *x-myMaterials.fC[id_mat];
    else
        y=2.*log ( 10. ) *x-myMaterials.fC[id_mat]+myMaterials.fA[id_mat]
          *pow ( myMaterials.fX1[id_mat]-x,myMaterials.fM[id_mat] );
    return  y;
}

void ElectronCrossSection::eIoni_CrossSection_table ( int id_mat )
{
    int i;

    for ( i=0; i<nb_bins; i++ )
    {
        f32 Ekine=data_h.E[i];
        data_h.eIonisationCS[i + nb_bins * id_mat]=eIoniCrossSection ( id_mat, Ekine );
    }
}

f32  ElectronCrossSection::eIoniCrossSection ( int id_mat, f32 Ekine )
{
    int i;
    f32  CrossTotale=0.;
    int index = myMaterials.index[id_mat]; // Get index of 1st element of the mixture

    for ( i=0; i<myMaterials.nb_elements[id_mat]; ++i ) // Get the sum of each element cross section
    {
        f32 tempo = myMaterials.atom_num_dens[index+i] * eIoniCrossSectionPerAtom ( index+i, Ekine );
//         tempo *=myMaterials.nb_atoms_per_vol[id_mat]; // Tempo value to avoid overflow
        CrossTotale+=tempo;//myMaterials.atom_num_dens[index+i]*myMaterials.nb_atoms_per_vol[id_mat]*eIoniCrossSectionPerAtom(index+i, Ekine);
    }

    return  CrossTotale;
}

f32 ElectronCrossSection:: eIoniCrossSectionPerAtom ( int index, f32 Ekine )
{
    f32  Cross=0.;
    f32  tmax=std::min ( 1.*GeV,Ekine*.5 );
    f32  xmin,xmax,gamma,gamma2,beta2,g;
//     GGcout << cutEnergyElectron << "  " << tmax << GGendl;
    if ( cutEnergyElectron<tmax )
    {
        xmin=cutEnergyElectron/Ekine;
        xmax=tmax/Ekine;
        gamma=Ekine/electron_mass_c2+1.;
        gamma2=gamma*gamma;
        beta2=1.-1./gamma2;
        g= ( 2.*gamma-1. ) /gamma2;
        Cross= ( ( xmax-xmin ) * ( 1.-g+1./ ( xmin*xmax ) +1./ ( ( 1.-xmin ) * ( 1.-xmax ) ) )
                 -g*std::log ( xmax* ( 1.-xmin ) / ( xmin* ( 1.-xmax ) ) ) ) /beta2;

        Cross*=twopi_mc2_rcl2/Ekine;
    }
    Cross*=myMaterials.mixture[index];

    return  Cross;
}



void ElectronCrossSection::eBrem_DEDX_table ( int id_mat )
{
    int i;

    for ( i=0; i<nb_bins; i++ )
    {
        f32 Ekine=data_h.E[i];
        data_h.eBremdedx[i + nb_bins * id_mat]=eBremDEDX ( Ekine,id_mat ) *mm2; //G4 internal unit
    }
}

f32 ElectronCrossSection::eBremDEDX ( f32 Ekine,int id_mat ) //id_mat = index material
{
    int i,n,nn,nmax;
    f32  Dedx;
    f32  totalEnergy,Z,natom,kp2,kmin,kmax,floss;
    f32  vmin,vmax,u,fac,c,v,dv;
    f32  thigh=100.*GeV;
    f32  cut=std::min ( cutEnergyGamma,Ekine );
    f32  rate,loss;
    f32  factorHigh=36./ ( 1450.*GeV );
    f32  coef1=-.5;
    f32  coef2=2./9.;
    f32  lowKinEnergy=0.*eV;
    f32  highKinEnergy=1.*GeV;
    f32  probsup=1.;
    f32  MigdalConstant=elec_radius*hbarc*hbarc*4.*pi/ ( electron_mass_c2*electron_mass_c2 );

    totalEnergy=Ekine+electron_mass_c2;
    Dedx=0.;

    if ( Ekine<lowKinEnergy )
        return  0.;

    for ( i=0; i<myMaterials.nb_elements[id_mat]; ++i ) // Check in each elt
    {
        int indexelt= i + myMaterials.index[id_mat];
        Z=myMaterials.mixture[indexelt];
        natom=myMaterials.atom_num_dens[indexelt]/myMaterials.nb_atoms_per_vol[id_mat];
        if ( Ekine<=thigh )
            loss=eBremLoss ( Z,Ekine,cut );
        loss*=natom;
        kp2=MigdalConstant*totalEnergy*totalEnergy*myMaterials.nb_electrons_per_vol[id_mat];

        kmin=1.*eV;
        kmax=cut;
        if ( kmax>kmin )
        {
            floss=0.;
            nmax=100;
            vmin=log ( kmin );
            vmax=log ( kmax );
            nn= ( int ) ( nmax* ( vmax-vmin ) / ( log ( highKinEnergy )-vmin ) ) ;
            if ( nn>0 )
            {
                dv= ( vmax-vmin ) /nn;
                v=vmin-dv;
                for ( n=0; n<=nn; n++ )
                {
                    v+=dv;
                    u=exp ( v );
                    //fac=u*SupressionFunction(material,Ekine,u);   //LPM flag off
                    fac=u*1.;
                    fac*=probsup* ( u*u/ ( u*u+kp2 ) ) +1.-probsup;
                    if ( ( n==0 ) || ( n==nn ) )
                        c=.5;
                    else
                        c=1.;
                    fac*=c;
                    floss+=fac ;
                }
                floss*=dv/ ( kmax-kmin );
            }
            else
                floss=1.;
            if ( floss>1. )
                floss=1.;
            loss*=floss;
        }
        Dedx+=loss;
    }
    if ( Dedx<0. )
        Dedx=0.;
    Dedx*=myMaterials.nb_atoms_per_vol[id_mat];
    return  Dedx;
}

f32 ElectronCrossSection::eBremLoss ( f32 Z,f32 T,f32 Cut )
{

    int   i,j;
    int   NZ=8,Nloss=11,iz=0;
    f32    Loss;
    f32    dz,xx,yy,fl,E;
    f32    aaa=.414,bbb=.345,ccc=.460,delz=1.e6;
    f32  beta=1.0,ksi=2.0,clossh=.254,closslow=1./3.,alosslow=1.;
    f32    Tlim=10.*MeV,xlim=1.2;

    for ( i=0; i<NZ; i++ )
    {
        dz=fabs ( Z-ZZ[i] );
        if ( dz<delz )
        {
            iz=i;
            delz=dz;
        }
    }
    xx=log10 ( T );
    fl=1.;
    if ( xx<=xlim )
    {
        xx/=xlim;
        yy=1.;
        fl=0.;
        for ( j=0; j<Nloss; j++ )
        {
            fl+=yy+coefloss[iz][j];
            yy*=xx;
        }
        if ( fl<.00001 )
            fl=.00001;
        else if ( fl>1. )
            fl=1.;
    }

    E=T+electron_mass_c2;
    Loss=Z* ( Z+ksi ) *E*E/ ( T+E ) *exp ( beta*log ( Cut/T ) ) * ( 2.-clossh*exp ( log ( Z ) /4. ) );
    if ( T<=Tlim )
        Loss/=exp ( closslow*log ( Tlim/T ) );
    if ( T<=Cut )
        Loss*=exp ( alosslow*log ( T/Cut ) );
    Loss*= ( aaa+bbb*T/Tlim ) / ( 1.+ccc*T/Tlim );
    Loss*=fl;
    Loss/=N_avogadro;

    return  Loss;
}


void ElectronCrossSection::eBrem_CrossSection_table ( int id_mat )
{

    for ( int i=0; i<nb_bins; i++ )
    {
        f32 Ekine=data_h.E[i];
        data_h.eBremCS[i + nb_bins * id_mat]=eBremCrossSection ( Ekine,id_mat ) *mm2; //G4 internal unit;
    }
}

f32 ElectronCrossSection::eBremCrossSection ( f32 Ekine,int id_mat )
{
    f32 CrossTotale=0.;
    CrossTotale=eBremCrossSectionPerVolume ( Ekine, id_mat );
    return  CrossTotale;
}

f32 ElectronCrossSection::eBremCrossSectionPerVolume ( f32 Ekine, int id_mat )
{
    int i,n,nn,nmax=100;
    f32  Cross=0.;
    f32  kmax,kmin,vmin,vmax,totalEnergy,kp2;
    f32  u,fac,c,v,dv,y;
    f32  tmax=std::min ( MaxKinEnergy,Ekine );
    f32  cut=std::max ( cutEnergyGamma, ( f32 ).1* ( f32 ) keV );
    f32  fsig=0.;
    f32  highKinEnergy=1.*GeV;
    f32  probsup=1.;
    f32  MigdalConstant=elec_radius*hbarc*hbarc*4.*pi/ ( electron_mass_c2*electron_mass_c2 );

    if ( cut>=tmax )
        return Cross;

    for ( i=0; i<myMaterials.nb_elements[id_mat]; i++ )
    {
        int indexelt= i + myMaterials.index[id_mat];

        Cross+=myMaterials.atom_num_dens[indexelt]
               *eBremCrossSectionPerAtom ( myMaterials.mixture[indexelt],cut, Ekine );
        if ( tmax<Ekine )
            Cross-=myMaterials.atom_num_dens[indexelt]
                   *eBremCrossSectionPerAtom ( myMaterials.mixture[indexelt],tmax,Ekine );
    }

    kmax=tmax;
    kmin=cut;
    totalEnergy=Ekine+electron_mass_c2;
    kp2=MigdalConstant*totalEnergy*totalEnergy*myMaterials.nb_electrons_per_vol[id_mat];
    vmin=log ( kmin );
    vmax=log ( kmax ) ;
    nn= ( int ) ( nmax* ( vmax-vmin ) / ( log ( highKinEnergy )-vmin ) );
    if ( nn>0 )
    {
        dv= ( vmax-vmin ) /nn;
        v=vmin-dv;
        for ( n=0; n<=nn; n++ )
        {
            v+=dv;
            u=exp ( v );
            //fac=SupressionFunction(material,Ekine,u);     //LPM flag is off
            fac=1.;
            y=u/kmax;
            fac*= ( 4.-4.*y+3.*y*y ) /3.;
            fac*=probsup* ( u*u/ ( u*u+kp2 ) ) +1.-probsup;
            if ( ( n==0 ) || ( n==nn ) )
                c=.5;
            else
                c=1.;
            fac*=c;
            fsig+=fac;
        }
        y=kmin/kmax;
        fsig*=dv/ ( -4.*log ( y ) /3.-4.* ( 1.-y ) /3.+0.5* ( 1.-y*y ) );
    }
    else
        fsig=1.;
    if ( fsig>1. )
        fsig=1.;
    Cross*=fsig;

    return Cross;
}

f32 ElectronCrossSection::eBremCrossSectionPerAtom ( f32 Z,f32 cut, f32 Ekine )
{
    int i,j,iz=0,NZ=8,Nsig=11;
    f32    Cross=0.;
    f32    ksi=2.,alfa=1.;
    f32    csigh=.127,csiglow=.25,asiglow=.02*MeV;
    f32    Tlim=10.*MeV;
    f32    xlim=1.2,delz=1.E6,absdelz;
    f32    xx,fs;

    if ( Ekine<1.*keV||Ekine<cut )
        return  Cross;

    for ( i=0; i<NZ; i++ )
    {
        absdelz=fabs ( Z-ZZ[i] );
        if ( absdelz<delz )
        {
            iz=i;
            delz=absdelz;
        }
    }

    xx=log10 ( Ekine );
    fs=1.;
    if ( xx<=xlim )
    {
        fs=coefsig[iz][Nsig-1];
        for ( j=Nsig-2; j>=0; j-- )
            fs=fs*xx+coefsig[iz][j];
        if ( fs<0. )
            fs=0.;
    }
    Cross=Z* ( Z+ksi ) * ( 1.-csigh*exp ( log ( Z ) /4. ) ) *pow ( log ( Ekine/cut ),alfa );

    if ( Ekine<=Tlim )
        Cross*=exp ( csiglow*log ( Tlim/Ekine ) ) * ( 1.+asiglow/ ( sqrt ( Z ) *Ekine ) );
    Cross*=fs/N_avogadro;
    if ( Cross<0. )
        Cross=0.;

    return  Cross;
}

void ElectronCrossSection::eMSC_CrossSection_table ( int id_mat )
{
    int i;
    for ( i=0; i<nb_bins; i++ )
    {
        f32 Ekine=data_h.E[i];
        data_h.eMSC[i + nb_bins * id_mat]=eMscCrossSection ( Ekine,id_mat );
    }
}

f32 ElectronCrossSection::eMscCrossSection ( f32 Ekine, int id_mat )
{
    int i;
    f32 CrossTotale=0.;

    for ( i=0; i<myMaterials.nb_elements[id_mat]; i++ )
    {
        int indexelt= i + myMaterials.index[id_mat];
        CrossTotale+=myMaterials.atom_num_dens[indexelt]
                     *eMscCrossSectionPerAtom ( Ekine, myMaterials.mixture[indexelt] );
    }

    return  CrossTotale;
}


f32 ElectronCrossSection::eMscCrossSectionPerAtom ( f32 Ekine,  unsigned short int AtomNumber )
{
    f32 AtomicNumber = ( f32 ) AtomNumber;

    int iZ=14,iT=21;
    f32  Cross=0.;
    f32  eKin,eTot,T,E;
    f32  beta2,bg2,b2big,b2small,ratb2,Z23,tau,w;
    f32  Z1,Z2,ratZ;
    f32  c,c1,c2,cc1,cc2,corr;
    f32  Tlim=10.*MeV;
    f32  sigmafactor=2.*pi*elec_radius*elec_radius;
    f32  epsfactor=2.*electron_mass_c2*electron_mass_c2*Bohr_radius*Bohr_radius/ ( hbarc*hbarc );
    f32  eps,epsmin=1.e-4,epsmax=1.e10;
    f32  beta2lim=Tlim* ( Tlim+2.*electron_mass_c2 ) / ( ( Tlim+electron_mass_c2 ) * ( Tlim+electron_mass_c2 ) );
    f32  bg2lim=Tlim* ( Tlim+2.*electron_mass_c2 ) / ( electron_mass_c2*electron_mass_c2 );

    Z23=2.*log ( AtomicNumber ) /3.;
    Z23=exp ( Z23 );



    tau=Ekine/electron_mass_c2;
    c=electron_mass_c2*tau* ( tau+2. ) / ( electron_mass_c2* ( tau+1. ) ); // a simplifier
    w=c-2.;
    tau=.5* ( w+sqrt ( w*w+4.*c ) );
    eKin=electron_mass_c2*tau;


    eTot=eKin+electron_mass_c2;
    beta2=eKin* ( eTot+electron_mass_c2 ) / ( eTot*eTot );
    bg2=eKin* ( eTot+electron_mass_c2 ) / ( electron_mass_c2*electron_mass_c2 );
    eps=epsfactor*bg2/Z23;
    if ( eps<epsmin )
        Cross=2.*eps*eps;
    else if ( eps<epsmax )
        Cross=log ( 1.+2.*eps )-2.*eps/ ( 1.+2.*eps );
    else
        Cross=log ( 2.*eps )-1.+1./eps;
    Cross*=AtomicNumber*AtomicNumber/ ( beta2*bg2 );

    while ( ( iZ>=0 ) && ( Zdat[iZ]>=AtomicNumber ) )
        iZ-=1;
    if ( iZ==14 )
        iZ=13;
    if ( iZ==-1 )
        iZ=0;
    Z1=Zdat[iZ];
    Z2=Zdat[iZ+1];
    ratZ= ( AtomicNumber-Z1 ) * ( AtomicNumber+Z1 ) / ( ( Z2-Z1 ) * ( Z2+Z1 ) );

    if ( eKin<=Tlim )
    {
        while ( ( iT>=0 ) && ( Tdat[iT]>=eKin ) )
            iT-=1;
        if ( iT==21 )
            iT=20;
        if ( iT==-1 )
            iT=0;
        T=Tdat[iT];
        E=T+electron_mass_c2;
        b2small=T* ( E+electron_mass_c2 ) / ( E*E );
        T=Tdat[iT+1];
        E=T+electron_mass_c2;
        b2big=T* ( E+electron_mass_c2 ) / ( E*E );
        ratb2= ( beta2-b2small ) / ( b2big-b2small );

        c1=celectron[iZ][iT];
        c2=celectron[iZ+1][iT];
        cc1=c1+ratZ* ( c2-c1 );
        c1=celectron[iZ][iT+1];
        c2=celectron[iZ+1][iT+1];
        cc2=c1+ratZ* ( c2-c1 );
        corr=cc1+ratb2* ( cc2-cc1 );
        Cross*=sigmafactor/corr;


    }
    else
    {
        c1=bg2lim*sig0[iZ]* ( 1.+hecorr[iZ]* ( beta2-beta2lim ) ) /bg2;
        c2=bg2lim*sig0[iZ+1]* ( 1.+hecorr[iZ+1]* ( beta2-beta2lim ) ) /bg2;
        if ( ( AtomicNumber>=Z1 ) && ( AtomicNumber<=Z2 ) )
            Cross=c1+ratZ* ( c2-c1 );
        else if ( AtomicNumber<Z1 )
            Cross=AtomicNumber*AtomicNumber*c1/ ( Z1*Z1 );
        else if ( AtomicNumber>Z2 )
            Cross=AtomicNumber*AtomicNumber*c2/ ( Z2*Z2 );
    }
    return  Cross;
}


void ElectronCrossSection::m_copy_cs_table_cpu2gpu()
{

        ui32 n = data_h.nb_bins;
        ui32 k = data_h.nb_mat;
// 
//         // Allocate GPU mem
        HANDLE_ERROR( cudaMalloc((void**) &data_d.E              , k*n*sizeof(f32)) );
        HANDLE_ERROR( cudaMalloc((void**) &data_d.eIonisationdedx, k*n*sizeof(f32)) );
        HANDLE_ERROR( cudaMalloc((void**) &data_d.eIonisationCS  , k*n*sizeof(f32)) );
        HANDLE_ERROR( cudaMalloc((void**) &data_d.eBremdedx      , k*n*sizeof(f32)) );
        HANDLE_ERROR( cudaMalloc((void**) &data_d.eBremCS        , k*n*sizeof(f32)) );
        HANDLE_ERROR( cudaMalloc((void**) &data_d.eMSC           , k*n*sizeof(f32)) );
        HANDLE_ERROR( cudaMalloc((void**) &data_d.eRange         , k*n*sizeof(f32)) );
//         HANDLE_ERROR( cudaMalloc((void**) &data_d.pIonisationdedx, k*n*sizeof(f32)) );
//         HANDLE_ERROR( cudaMalloc((void**) &data_d.pIonisationCS  , k*n*sizeof(f32)) );
//         HANDLE_ERROR( cudaMalloc((void**) &data_d.pBremdedx      , k*n*sizeof(f32)) );
//         HANDLE_ERROR( cudaMalloc((void**) &data_d.pBremCS        , k*n*sizeof(f32)) );
//         HANDLE_ERROR( cudaMalloc((void**) &data_d.pMSC           , k*n*sizeof(f32)) );
//         HANDLE_ERROR( cudaMalloc((void**) &data_d.pRange         , k*n*sizeof(f32)) );
//         HANDLE_ERROR( cudaMalloc((void**) &data_d.pAnni_CS_tab   , k*n*sizeof(f32)) );

// 
//         // Copy data to GPU
        data_d.nb_bins = n;
        data_d.nb_mat = k;
        data_d.E_min = data_h.E_min;
        data_d.E_max = data_h.E_max;
        data_d.cutEnergyElectron = data_h.cutEnergyElectron;
        data_d.cutEnergyGamma    = data_h.cutEnergyGamma;
        
        
        HANDLE_ERROR( cudaMemcpy(data_d.E              , data_h.E               , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(data_d.eIonisationdedx, data_h.eIonisationdedx , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(data_d.eIonisationCS  , data_h.eIonisationCS   , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(data_d.eBremdedx      , data_h.eBremdedx       , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(data_d.eBremCS        , data_h.eBremCS         , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(data_d.eMSC           , data_h.eMSC            , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(data_d.eRange         , data_h.eRange          , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
//         HANDLE_ERROR( cudaMemcpy(data_d.pIonisationdedx, data_h.pIonisationdedx , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
//         HANDLE_ERROR( cudaMemcpy(data_d.pIonisationCS  , data_h.pIonisationCS   , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
//         HANDLE_ERROR( cudaMemcpy(data_d.pBremdedx      , data_h.pBremdedx       , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
//         HANDLE_ERROR( cudaMemcpy(data_d.pBremCS        , data_h.pBremCS         , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
//         HANDLE_ERROR( cudaMemcpy(data_d.pMSC           , data_h.pMSC            , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
//         HANDLE_ERROR( cudaMemcpy(data_d.pRange         , data_h.pRange          , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
//         HANDLE_ERROR( cudaMemcpy(data_d.pAnni_CS_tab   , data_h.pAnni_CS_tab    , sizeof(f32)*n*k, cudaMemcpyHostToDevice) );
}


#endif
