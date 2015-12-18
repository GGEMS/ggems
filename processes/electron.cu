// GGEMS Copyright (C) 2015

#ifndef ELECTRON_CU
#define ELECTRON_CU
#include "electron.cuh"


__host__ __device__ void e_read_CS_table(
//                             ParticlesData particles,
    int id,
    int mat, //material
    f32 energy, //energy of particle
    ElectronsCrossSectionTable &d_table,
    unsigned char &next_discrete_process, //next discrete process id
    int &table_index,
    f32 & next_interaction_distance,
    f32 & dedxeIoni,
    f32 & dedxeBrem,
    f32 & erange,
    f32 & lambda,
    f32 randomnumbereBrem,
    f32 randomnumbereIoni,
    GlobalSimulationParametersData parameters)
    {

//     if(id==DEBUGID)printf("energy %e d_table.E %e mat %d d_table.nb_bins %u \n",energy,d_table.E,mat,d_table.nb_bins);
//     table_index = binary_search(energy,d_table.E,d_table.nb_bins) + mat*d_table.nb_bins;
    table_index = binary_search(energy,d_table.E,(mat+1)*d_table.nb_bins,mat*d_table.nb_bins);

    if(parameters.physics_list[ELECTRON_IONISATION] == ENABLED)
        {

        f32 distanceeIoni = randomnumbereIoni / linear_interpolation(d_table.E[table_index-1], d_table.eIonisationCS[table_index-1], d_table.E[table_index], d_table.eIonisationCS[table_index], energy );

        if(distanceeIoni<next_interaction_distance)
            {
            next_interaction_distance = distanceeIoni;
            next_discrete_process = ELECTRON_IONISATION;
            }

        dedxeIoni = linear_interpolation(d_table.E[table_index-1],d_table.eIonisationdedx[table_index-1], d_table.E[table_index], d_table.eIonisationdedx[table_index], energy );

        }
// 
    if(parameters.physics_list[ELECTRON_BREMSSTRAHLUNG] == ENABLED)
        {

        f32 distanceeBrem = randomnumbereBrem /  linear_interpolation(d_table.E[table_index-1], d_table.eBremCS[table_index-1], d_table.E[table_index], d_table.eBremCS[table_index], energy ) ;

        if(distanceeBrem<next_interaction_distance)
            {
            next_interaction_distance = distanceeBrem;
            next_discrete_process = ELECTRON_BREMSSTRAHLUNG;
            }

        dedxeBrem =  linear_interpolation(d_table.E[table_index-1],d_table.eBremdedx[table_index-1], d_table.E[table_index], d_table.eBremdedx[table_index], energy );
        }




    erange = linear_interpolation(d_table.E[table_index-1],d_table.eRange[table_index-1], d_table.E[table_index], d_table.eRange[table_index], energy );
    
    
//         printf("d_table.E[table_index-1] %e ,d_table.eRange[table_index-1] %e , d_table.E[table_index] %e, d_table.eRange[table_index] %e table_index %d\n",d_table.E[table_index-1],d_table.eRange[table_index-1], d_table.E[table_index], d_table.eRange[table_index],table_index);
    
    
    if(parameters.physics_list[ELECTRON_MSC] == ENABLED)
        {
        lambda = linear_interpolation(d_table.E[table_index-1],d_table.eMSC[table_index-1], d_table.E[table_index], d_table.eMSC[table_index], energy );
        lambda=1./lambda;
        }

    }

__host__ __device__ f32 StepFunction(f32 Range)
    {
    f32 alpha=0.2;
    f32 rho=1.*mm;
    f32  StepF;
    if(Range<rho)
        return  Range;
    StepF=alpha*Range+rho*(1.-alpha)*(2.-rho/Range);
    if(StepF<rho)
        StepF=rho;

    return  StepF;
    }
    
__host__ __device__ f32 LossApproximation(f32 StepLength, f32 Ekine, f32 erange, ElectronsCrossSectionTable d_table, int mat, int id)
    {
    f32  range,perteApp = 0;
    range=erange;
    range-=StepLength;
    if (range >1.*nm)
        perteApp=GetEnergy(range, d_table, mat);
    else
        perteApp = 0.;

    perteApp=Ekine-perteApp;

    return  perteApp;
    }
    
    
#define rate .55
#define fw 4.
#define nmaxCont 16
#define minLoss 10.*eV
__host__ __device__ f32 eFluctuation(f32 meanLoss,f32 cutEnergy, MaterialsTable materials, ParticlesData &particles, int id, int id_mat)
    {
    /*if(id==DEBUGID) PRINT_PARTICLE_STATE("");*/
    int nb,k;
    f32  LossFluct=0.,lossc=0.;//,minLoss=10.*eV;
//     f32  rate=.55,fw=4.,nmaxCont=16.;
    f32  tau,gamma,gamma2,beta2;
    f32  F1,F2,E0,E1,E2,E1Log,E2Log,I,ILog;
    f32  e1,e2,esmall,w,w1,w2,C,alfa,alfa1,namean;
    f32  a1=0.,a2=0.,a3=0.,sa1;
    f32  emean=0.,sig2e=0.,sige=0.,p1=0.,p2=0.,p3=0.;
    f32  tmax=min(cutEnergy,.5*particles.E[id]);

    if(meanLoss<minLoss)
        return  meanLoss;
    /*if(id==DEBUGID) PRINT_PARTICLE_STATE("");*/
    tau=particles.E[id]/electron_mass_c2;
    gamma=tau+1.;
    gamma2=gamma*gamma;
    beta2=tau*(tau+2.)/gamma2;
    F1=materials.fF1[id_mat];
    F2=materials.fF2[id_mat];
    E0=materials.fEnergy0[id_mat];
    E1=materials.fEnergy1[id_mat];
    E2=materials.fEnergy2[id_mat];
    E1Log=materials.fLogEnergy1[id_mat];
    E2Log=materials.fLogEnergy2[id_mat];
    I=materials.electron_mean_excitation_energy[id_mat];
    ILog=logf(I);//materials.fLogMeanExcitationEnergy[id_mat];
    esmall=.5*sqrtf(E0*I);

//     if(id==DEBUGID) printf("%f \n",ILog);
//     return meanLoss;
    if(tmax<=E0)
        {
        return  meanLoss;
        }

    if(tmax>I)
        {
        w2=logf(2.*electron_mass_c2*beta2*gamma2)-beta2;
        if(w2>ILog)
            {
            C=meanLoss*(1.-rate)/(w2-ILog);
            a1=C*F1*(w2-E1Log)/E1;
            if(w2>E2Log)
                a2=C*F2*(w2-E2Log)/E2;
            if(a1<nmaxCont)
                {
                sa1=sqrtf(a1);
                if(JKISS32(particles, id)<expf(-sa1))
                    {
                    e1=esmall;
                    a1=meanLoss*(1.-rate)/e1;
                    a2=0.;
                    e2=E2;
                    }
                else
                    {
                    a1=sa1 ;
                    e1=sa1*E1;
                    e2=E2;
                    }
                }
            else
                {
                a1/=fw;
                e1=fw*E1;
                e2=E2;
                }
            }
        }

    w1=tmax/E0;
    if(tmax>E0)
        a3=rate*meanLoss*(tmax-E0)/(E0*tmax*log(w1));
    if(a1>nmaxCont)
        {
        emean+=a1*e1;
        sig2e+=a1*e1*e1;
        }
    else if(a1>0.)
        {
        p1=(f32)G4Poisson(a1, particles, id);
        LossFluct+=p1*e1;
        if(p1>0.)
            LossFluct+=(1.-2.*JKISS32(particles, id))*e1;
        }
    if(a2>nmaxCont)
        {
        emean+=a2*e2;
        sig2e+=a2*e2*e2;
        }
    else if(a2>0.)
        {
        p2=(f32)G4Poisson(a2, particles, id);
        LossFluct+=p2*e2;
        if(p2>0.)
            LossFluct+=(1.-2.*JKISS32(particles, id))*e2;
        }
    if(a3>0.)
        {
        p3=a3;
        alfa=1.;
        if(a3>nmaxCont)
            {
            alfa=w1*(nmaxCont+a3)/(w1*nmaxCont+a3);
            alfa1=alfa*log(alfa)/(alfa-1.);
            namean=a3*w1*(alfa-1.)/((w1-1.)*alfa);
            emean+=namean*E0*alfa1;
            sig2e+=E0*E0*namean*(alfa-alfa1*alfa1);
            p3=a3-namean;
            }
        w2=alfa*E0;
        w=(tmax-w2)/tmax;
        nb=G4Poisson(p3, particles, id);
        if(nb>0)
            for(k=0; k<nb; k++)
                lossc+=w2/(1.-w*JKISS32(particles, id));
        }
    if(emean>0.)
        {
        sige=sqrtf(sig2e);
//         if(isnan(emean)) emean = 0;
//         if(isnan(sige)) sige = 0;
        LossFluct+=max(0.,Gaussian(emean,sige,particles, id));
//         f32 toto = Gaussian(emean,sige,particles, id);
//         if(isnan(toto)||isinf(toto)) toto = 0;
//         if(toto>0) LossFluct += toto;

//         LossFluct+=fmaxf(0.,emean);


        }
    LossFluct+=lossc;

    return  LossFluct;
    }
#undef rate
#undef fw
#undef nmaxCont
#undef minLoss

    
__host__ __device__ f32 eLoss(f32 LossLength, f32 &Ekine, f32 dedxeIoni, f32 dedxeBrem, f32 erange,ElectronsCrossSectionTable d_table, int mat, MaterialsTable materials, ParticlesData &particles,GlobalSimulationParametersData parameters, int id)
    {
    f32  perteTot=0.;//,perteBrem=0.,perteIoni=0.;
    perteTot=LossLength*(dedxeIoni + dedxeBrem);

    if(perteTot>Ekine*0.01) // 0.01 is xi
        perteTot=LossApproximation(LossLength, Ekine, erange, d_table, mat, id);

/// \warning ADD for eFluctuation
    if(dedxeIoni>0.)
        perteTot=eFluctuation(perteTot,parameters.electron_cut,materials,particles,id,mat);

    if((Ekine-perteTot)<=(1.*eV))
        {
        perteTot=Ekine;

        }

    Ekine-=perteTot;

    return  perteTot;
    }
    
    
#define tausmall 1.E-16
__host__ __device__ f32 gGeomLengthLimit(f32 gPath,f32 cStep,f32 currentLambda,f32 currentRange,f32 par1,f32 par3)
    {
    f32  tPath;
//     f32  tausmall=1.E-16; //tausmall=1.E-16;

    par3=1.+par3;
    tPath=gPath;
    if(gPath>currentLambda*tausmall)
        {
        if(par1<0.)
            tPath=-currentLambda*log(1.-gPath/currentLambda);
        else
            {
            if(par1*par3*gPath<1.)
                tPath=(1.-exp(log(1.-par1*par3*gPath)/par3))/par1;
            else
                tPath=currentRange;
            }
        }
    if(tPath<gPath)
        tPath=gPath;
    return  tPath;
    }
#undef tausmall


__device__ f32 eSimpleScattering(f32 xmeanth,f32 x2meanth, int id, ParticlesData &particles)
    {
    f32    a=(2.*xmeanth+9.*x2meanth-3.)/(2.*xmeanth-3.*x2meanth+1.);
    f32    prob=(a+2.)*xmeanth/a;
    f32  cth=1.;

    if(JKISS32(particles, id)<prob)
        cth=-1.+2.*expf(logf(JKISS32(particles, id))/(a+1.));
    else
        cth=-1.+2.*JKISS32(particles, id);
    return    cth;
    }



__device__ f32 eCosineTheta(f32 trueStep,f32 currentRange,f32 currentLambda,f32 currentEnergy,f32 *currentTau,f32 par1,f32 par2, MaterialsTable materials, int id_mat, int id, ParticlesData &particles)
    {
    f32 particleEnergy = particles.E[id];
    f32  costh,sinth;
    f32  tau;
    const f32 taubig=8.,tausmall=1.E-16,taulim=1.E-6;
    f32  c,c1,x0,b,bx,b1,ebx,eb1;
    f32  prob=0.,qprob=1.;
    f32  a=1.,ea=0.,eaa=1.;
    f32  xmeanth,xmean1=1.,xmean2=0.,x2meanth;
    f32  dtrl=0.05;//5./100.;
    f32  xsi=3.;
    f32    theta0,theta0max=pi/6.,y,corr,betacp,c_highland=13.6*MeV;
    f32    f1x0,f2x0;


    costh=1.;
    tau=trueStep/currentLambda;

    if(trueStep>=currentRange*dtrl)
        {
        if((par1*trueStep)<1.)
            {
            tau=-par2*logf(1.-par1*trueStep);
            }
        else if((1.-particleEnergy/currentEnergy)>taulim)
            tau=taubig;
        }

    *currentTau=tau;
    if(tau>=taubig)
        {
        f32 temp;
//         do{
        temp=JKISS32(particles, id);
//         }while((1.-temp)<2.e-7); // to avoid 1 due to f32 approximation
        costh=-1.+2.*temp;

        }
    else if(tau>=tausmall)
        {
        x0=1.;
        b=2.;
        b1=3.;
        bx=1.;
        eb1=3.;
        ebx=1.;
        prob=1.;
        qprob=1.;
        xmeanth=expf(-tau);
        x2meanth=(1.+2.*expf(-2.5*tau))/3.;
        if(1.-particleEnergy/currentEnergy>.5)
            {
            costh=eSimpleScattering(xmeanth,x2meanth, id, particles);

            return  costh;
            }

        betacp=sqrtf(currentEnergy*(currentEnergy+2.*electron_mass_c2)
                     *particleEnergy*(particleEnergy+2.*electron_mass_c2)
                     /((currentEnergy+electron_mass_c2)*(particleEnergy+electron_mass_c2)));


        y=trueStep/materials.rad_length[id_mat];
        theta0=c_highland*sqrtf(y)/betacp;
        y=logf(y);

        f32 Zeff = materials.nb_electrons_per_vol[id_mat]/materials.nb_atoms_per_vol[id_mat];

        corr=(1.-8.778E-2/Zeff)*(.87+.03*logf(Zeff))
             +(4.078E-2+1.7315E-4*Zeff)*(.87+.03*logf(Zeff))*y;
        theta0*=corr ;

        if(theta0*theta0<tausmall)
            return  costh;
        if(theta0>theta0max)
            {
            costh=eSimpleScattering(xmeanth,x2meanth, id, particles);
            return  costh;
            }

        sinth=sinf(.5*theta0);
        a=.25/(sinth*sinth);
        ea=expf(-xsi);
        eaa=1.-ea ;
        xmean1=1.-(1.-(1.+xsi)*ea)/(a*eaa);
        x0=1.-xsi/a;
        if(xmean1<=.999*xmeanth)
            {
            costh=eSimpleScattering(xmeanth,x2meanth, id, particles);
            return  costh;
            }

        c=2.943-.197*logf(Zeff+1.)
          +(.0987-.0143*logf(Zeff+1.))*y;

        if(fabsf(c-3.)<.001)
            c=3.001;
        if(fabsf(c-2.)<.001)
            c=2.001;
        if(fabsf(c-1.)<.001)
            c=1.001;
        c1=c-1.;

        b=1.+(c-xsi)/a;
        b1=b+1.;
        bx=c/a;
        eb1=expf(c1*logf(b1));
        ebx=expf(c1*logf(bx));
        xmean2=(x0*eb1+ebx-(eb1*bx-b1*ebx)/(c-2.))/(eb1-ebx);
        f1x0=a*ea/eaa;
        f2x0=c1*eb1/(bx*(eb1-ebx));
        prob=f2x0/(f1x0+f2x0);
        qprob=xmeanth/(prob*xmean1+(1.-prob)*xmean2);
        if(JKISS32(particles, id)<qprob)
            {
            if(JKISS32(particles, id)<prob)
                costh=1.+logf(ea+JKISS32(particles, id)*eaa)/a;
            else
                costh=b-b1*bx/expf(logf(ebx+(eb1-ebx)*JKISS32(particles, id))/c1);
            }
        else
            costh=-1.+2.*JKISS32(particles, id);
        }
    return  costh;
    }



__host__ __device__ void gLatCorrection(f32xyz currentDir,f32 tPath,f32 zPath,f32 currentTau,f32 phi,f32 sinth, ParticlesData &particles, int id, f32 safety)
    {
    f32  latcorr,etau,rmean,rmax,Phi,psi,lambdaeff;
    const f32  kappa=2.5,tlimitminfix=1.E-6*mm,taulim=1.E-6,tausmall=1.E-16,taubig=8.,geomMin=1.E-6*mm;
//     struct  Vector  latDir;
    f32xyz  latDir;
    lambdaeff=tPath/currentTau;

    if(safety>tlimitminfix)   // Safety is distance to near voxel
        {
        rmean=0.;
        if((currentTau>=tausmall))
            {
            if(currentTau<taulim)
                rmean=kappa*currentTau*currentTau*currentTau*(1.-(kappa+1.)*currentTau*.25)/6.; //powf(currentTau,3.)
            else
                {
                etau=0.;
                if(currentTau<taubig)
                    etau=expf(-currentTau);
                rmean=-kappa*currentTau;
                rmean=-expf(rmean)/(kappa*(kappa-1.));
                rmean+=currentTau-(kappa+1.)/kappa+kappa*etau/(kappa-1.);
                }
            if(rmean>0.)
                rmean=2.*lambdaeff*sqrtf(rmean/3.);
            else
                rmean=0.;
            }
        rmax=(tPath-zPath)*(tPath+zPath);
        if(rmax<0.)
            rmax=0.;
        else
            rmax=sqrtf(rmax);
        if(rmean>=rmax)
            rmean=rmax;

        if(rmean<=geomMin)
            return;

        if(rmean>0.)
            {
            if((currentTau>=tausmall) )
                {
                if(currentTau<taulim)
                    {
                    latcorr=lambdaeff*kappa*currentTau*currentTau*(1.-(kappa+1.)*currentTau/3.)/3.;
                    }
                else
                    {
                    etau=0.;
                    if(currentTau<taubig)
                        etau=expf(-currentTau);
                    latcorr=-kappa*currentTau;
                    latcorr=expf(latcorr)/(kappa-1.);
                    latcorr+=1.-kappa*etau/(kappa-1.);
                    latcorr*=2.*lambdaeff/3.;

                    }
                }
            if(latcorr>rmean)
                latcorr=rmean;
            else if(latcorr<0.)
                latcorr=0.;
            Phi=0.;
            if(fabsf(rmean*sinth)<=latcorr)
                {
                Phi=2.*pi*JKISS32(particles, id);
                }
            else
                {
                psi=acosf(latcorr/(rmean*sinth));
                if(JKISS32(particles, id)<.5)
                    Phi=phi+psi;
                else
                    Phi=phi-psi;
                }
            latDir.x=cos(Phi);
            latDir.y=sin(Phi);
            latDir.z=0.;
            latDir=rotateUz(latDir,currentDir);
            if(rmean>safety)
                rmean=safety*.99;

            particles.px[id]+=latDir.x*rmean;
            particles.py[id]+=latDir.y*rmean;
            particles.pz[id]+=latDir.z*rmean;


            }
        }
    }



    
__host__ __device__ void eMscScattering(f32 tPath,f32 zPath,f32 currentRange,f32 currentLambda,f32 currentEnergy,f32 par1,f32 par2, ParticlesData &particles, int id, MaterialsTable materials, int mat, VoxVolumeData phantom,ui16xyzw index_phantom)
{


    f32  costh,sinth,phi,currentTau;
    const f32  tlimitminfix=1.E-10*mm,tausmall=1.E-16; //,taulim=1.E-6
    f32xyz  Dir, currentDir;

    if((particles.E[id]<0.)||(tPath<=tlimitminfix)||(tPath/tausmall<currentLambda))
        {
        return;
        }


    costh=eCosineTheta(tPath,currentRange,currentLambda,currentEnergy,&currentTau,par1,par2, materials, mat, id,particles);

    if(fabs(costh)>1.)
        return;
    if(costh<(1.-1000.*tPath/currentLambda)&&(particles.E[id])>(20.*MeV))
        {
        do
            {

            costh=1.+2.*logf(JKISS32(particles, id))*tPath/currentLambda;
            }
        while((costh<-1.));
        }


    sinth=sqrtf((1.-costh)*(1.+costh));
    phi=2.*pi*JKISS32(particles, id);

    Dir = make_f32xyz(sinth*cosf(phi), sinth*sinf(phi), costh);

    particles.px[id]+=particles.dx[id]*zPath;
    particles.py[id]+=particles.dy[id]*zPath;
    particles.pz[id]+=particles.dz[id]*zPath;


    currentDir = make_f32xyz(particles.dx[id],particles.dy[id],particles.dz[id]);

    Dir=rotateUz(Dir,currentDir);


    particles.dx[id] = Dir.x;
    particles.dy[id] = Dir.y;
    particles.dz[id] = Dir.z;


    // Read position
    f32xyz position; // mm
    position.x = particles.px[id];
    position.y = particles.py[id];
    position.z = particles.pz[id];

    // Read direction
    f32xyz direction;
    direction.x = particles.dx[id];
    direction.y = particles.dy[id];
    direction.z = particles.dz[id];

    //Get Phantom index
//         int4 index_phantom;
//         const f32xyz ivoxsize = vec3_inverse(phantom.voxel_size);
//         index_phantom.x = 0;//int(position.x * ivoxsize.x);
//         index_phantom.y = 0;//int(position.y * ivoxsize.y);
//         index_phantom.z = 0;//int(position.z * ivoxsize.z);
//         index_phantom.w = index_phantom.z*phantom.nb_voxel_slice
//                           + index_phantom.y*phantom.size_in_vox.x
//                           + index_phantom.x; // linear index

    f32 safety = GetSafety(position, direction,index_phantom,make_f32xyz(phantom.spacing_x,phantom.spacing_y,phantom.spacing_z)) ;

//     Comment next line to disable lateral correction
    gLatCorrection(currentDir,tPath,zPath,currentTau,phi,sinth,particles,id,safety);

    }
    
    
// From Eric's code
__host__ __device__ f32 GlobalMscScattering(f32 GeomPath,f32 cutstep,f32 CurrentRange,f32 CurrentEnergy, f32 CurrentLambda, f32 dedxeIoni, f32 dedxeBrem, ElectronsCrossSectionTable d_table, int mat, ParticlesData &particles, int id,f32 par1,f32 par2, MaterialsTable materials,DoseData &dosi, ui16xyzw index_phantom, VoxVolumeData phantom,GlobalSimulationParametersData parameters )
    {

    f32  edep,TruePath,zPath;//,tausmall=1.E-16;
//     // MSC disabled
    if(parameters.physics_list[ELECTRON_MSC] != ENABLED)
        {
        if(GeomPath<cutstep)
            {
            edep = eLoss(GeomPath, particles.E[id], dedxeIoni, dedxeBrem, CurrentRange, d_table, mat, materials, particles,parameters, id );

            /// TODO WARNING  ACTIVER LA FONCTION DE DOSIMETRIE
//             dose_record(dosi, edep, particles.px[id],particles.py[id],particles.pz[id]);

            }
        particles.px[id] += particles.dx[id] * GeomPath;
        particles.py[id] += particles.dy[id] * GeomPath;
        particles.pz[id] += particles.dz[id] * GeomPath;

        return  GeomPath;
        }

    if(GeomPath==cutstep)
        {
        zPath=gTransformToGeom(GeomPath,CurrentRange,CurrentLambda,CurrentEnergy,&par1,&par2,d_table, mat);
        /*if(id==DEBUGID) PRINT_PARTICLE_STATE("");*/
        }
    else
        {
        zPath=GeomPath;
        TruePath=gGeomLengthLimit(GeomPath,cutstep,CurrentLambda,CurrentRange,par1,par2);
        GeomPath=TruePath;

        edep = eLoss(TruePath, particles.E[id], dedxeIoni, dedxeBrem, CurrentRange, d_table, mat, materials, particles, parameters, id );
        
            /// TODO WARNING  ACTIVER LA FONCTION DE DOSIMETRIE
//         dose_record(dosi, edep, particles.px[id],particles.py[id],particles.pz[id], id);

        }


    if(particles.E[id] != 0.0)   // if not laststep
        {
        eMscScattering(GeomPath,zPath,CurrentRange,CurrentLambda,CurrentEnergy,par1,par2, particles,  id, materials, mat, phantom, index_phantom);
        /*if(id==DEBUGID) PRINT_PARTICLE_STATE("");*/

        }
    else
        {
        particles.endsimu[id]=PARTICLE_DEAD;
        particles.px[id]+=particles.dx[id]*zPath;
        particles.py[id]+=particles.dy[id]*zPath;
        particles.pz[id]+=particles.dz[id]*zPath;
        }
    /*if(id==DEBUGID) PRINT_PARTICLE_STATE("");*/
    return  TruePath;
    }    

    
__host__ __device__ void eSampleSecondarieElectron(f32 CutEnergy, ParticlesData &particles, int id, f32 *cache_secondaries, DoseData &dosi,GlobalSimulationParametersData parameters )
    {
    f32  totalEnergy,deltaEnergy,totMom,deltaMom;
    f32  xmin,xmax,gamma,gamma2;//,beta2;
    f32  x,z,q,grej,g,y;
    f32  cost,sint,phi;
    f32  tmax=fminf(1.*GeV,.5*particles.E[id]);
    f32  tmin=CutEnergy;

    f32xyz  ElecDir;

    if(tmin>=tmax)
        return;

    totalEnergy=particles.E[id]+electron_mass_c2;
    totMom=sqrtf(particles.E[id]*(totalEnergy+ electron_mass_c2));
    xmin=tmin/particles.E[id];
    xmax=tmax/particles.E[id];
    gamma=totalEnergy/electron_mass_c2;
    gamma2=gamma*gamma;
//     beta2=1.-1./gamma2;
    g=(2.*gamma-1.)/gamma2;
    y=1.-xmax;
    grej=1.-g*xmax+xmax*xmax*(1.-g+(1.-g*y)/(y*y));

    do
        {
        q=JKISS32(particles,id);
        x=xmin*xmax/(xmin*(1.-q)+xmax*q);
        y=1.-x;
        z=1.-g*x+x*x*(1.-g+(1.-g*y)/(y*y));
        }
    while((grej*JKISS32(particles,id)>z));

    deltaEnergy=x*particles.E[id];
    deltaMom=sqrtf(deltaEnergy*(deltaEnergy+2.*electron_mass_c2));
    cost=deltaEnergy*(totalEnergy+electron_mass_c2)/(deltaMom*totMom);
    sint=1.-cost*cost;
    if(sint>0.)
        sint=sqrtf(sint);
    phi=2.*pi*JKISS32(particles,id);

    ElecDir.x=sint*cosf(phi);
    ElecDir.y=sint*sinf(phi);
    ElecDir.z=cost;
    f32xyz currentDir;
    currentDir = make_f32xyz(particles.dx[id],particles.dy[id],particles.dz[id]);

    ElecDir=rotateUz(ElecDir,currentDir);

//     deltaEnergy = __int2f32_rn(__f322int_rn(deltaEnergy));
    particles.E[id]-=deltaEnergy;
//     if(id==7949384) printf(" delta %f ",deltaEnergy);
    if(particles.E[id]>0.0)
        currentDir=CorrUnit(currentDir,ElecDir,totMom,deltaMom);

    particles.dx[id]=currentDir.x;
    particles.dy[id]=currentDir.y;
    particles.dz[id]=currentDir.z;

    if((int)(particles.level[id])<parameters.nb_of_secondaries)
        {

/// \warning \TODO COMMENT FOR NO SECONDARY,
//         GenerateNewElectronParticle(deltaEnergy,ElecDir);

        }
    else
        {
        /// WARNING TODO ACTIVER DOSIMETRY ICI
//         dose_record(dosi, deltaEnergy, particles.px[id],particles.py[id],particles.pz[id],id);
        }

    }


__host__ __device__ f32xyz CorrUnit(f32xyz u, f32xyz v,f32 uMom, f32 vMom)
    {
    f32  r;
    f32xyz  Final;

    Final.x=u.x*uMom-v.x*vMom;
    Final.y=u.y*uMom-v.y*vMom;
    Final.z=u.z*uMom-v.z*vMom;
    r=Final.x*Final.x+Final.y*Final.y+Final.z*Final.z;
    if(r>0.)
        {
        r=sqrt(Final.x*Final.x+Final.y*Final.y+Final.z*Final.z);
        Final.x=Final.x/r;
        Final.y=Final.y/r;
        Final.z=Final.z/r;
        }

    return  Final;
    }
    
    
#define tausmall 1.E-20
#define taulim 1.E-6
#define tlimitminfix 1.E-6*mm
#define dtrl 5./100
__host__ __device__ f32 gTransformToGeom(f32 TPath,f32 currentRange,f32 currentLambda,f32 currentEnergy,f32 *par1,f32 *par2, ElectronsCrossSectionTable electron_CS_table, int mat)
    {
    f32  ZPath,zmean;
//     f32  tausmall=1.E-20,taulim=1.E-6,tlimitminfix=1.E-6*mm;
//     f32  dtrl=5./100.;
    f32  tau,t1,lambda1;
    f32  par3;

    *par1=-1.;
    *par2=par3=0.;
    ZPath=TPath;
    if(TPath<tlimitminfix)
        return  ZPath;
    if(TPath>currentRange)
        TPath=currentRange;
    tau=TPath/currentLambda;
    if((tau<=tausmall)/*||insideskin*/)
        {
        ZPath=TPath;
        if(ZPath>currentLambda)
            ZPath=currentLambda;
        return  ZPath;
        }
    zmean=TPath;
    if(TPath<currentRange*dtrl)
        {
        if(tau<taulim)
            zmean=TPath*(1.-0.5*tau);
        else
            zmean=currentLambda*(1.-expf(-tau));
        }
    else if(currentEnergy<electron_mass_c2)
        {
        *par1=1./currentRange;
        *par2=1./(*par1*currentLambda);
        par3=1.+*par2;
        if(TPath<currentRange)
            zmean=(1.-expf(par3*logf(1.-TPath/currentRange)))/(*par1*par3);
        else
            zmean=1./(*par1*par3);
        }
    else
        {
        t1=GetEnergy(currentRange-TPath, electron_CS_table, mat);
        lambda1=1./GetLambda(t1,1, electron_CS_table, mat);
        *par1=(currentLambda-lambda1)/(currentLambda*TPath);
        *par2=1./(*par1*currentLambda);
        par3=1.+*par2;

        zmean=(1.-expf(par3*logf(lambda1/currentLambda)))/(*par1*par3);
        }
    ZPath=zmean;

//     return (fminf(ZPath,currentLambda));
    if(ZPath>currentLambda)
        ZPath=currentLambda;
    return  ZPath;
    }
#undef tausmall
#undef taulim
#undef tlimitminfix
#undef dtrl

__host__ __device__ f32 GetEnergy(f32 Range, ElectronsCrossSectionTable d_table, int mat)
    {

    int index = binary_search( Range, d_table.eRange, d_table.nb_bins*mat+d_table.nb_bins, d_table.nb_bins*mat );

    f32 newRange = linear_interpolation(d_table.eRange[index-1], d_table.E[index-1], d_table.eRange[index], d_table.E[index], Range );

    return newRange;
    }

__host__ __device__ f32 GetLambda(f32 Range, unsigned short int flag, ElectronsCrossSectionTable d_table, int mat)
    {
    int index = binary_search( Range, d_table.E, d_table.nb_bins*mat+d_table.nb_bins, d_table.nb_bins*mat );

    if (flag == 1) return linear_interpolation(d_table.E[index-1],d_table.eMSC[index-1], d_table.E[index], d_table.eMSC[index], Range );

    else if (flag == 2) return linear_interpolation(d_table.E[index-1],d_table.eIonisationCS[index-1], d_table.E[index], d_table.eIonisationCS[index], Range );

    else /*if (flag == 3)*/ return linear_interpolation(d_table.E[index-1],d_table.eBremCS[index-1], d_table.E[index], d_table.eBremCS[index], Range );

    }
    
void ElectronCrossSection::initialize(GlobalSimulationParameters params,MaterialsTable materials)
{
    nb_bins = params.data_h.cs_table_nbins;
    nb_mat = materials.nb_materials;
//     params = parameters;
    parameters = params;
    myMaterials = materials;
    MaxKinEnergy = parameters.data_h.cs_table_max_E;
    MinKinEnergy = parameters.data_h.cs_table_min_E;
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


    for(int i=0; i<nb_mat*nb_bins; i++) // Create Energy table between Emin and Emax
    {
        data_h.E[i]=0.;
    }   
        
        
        Energy_table();

        
    for(int id_mat = 0; id_mat< myMaterials.nb_materials; ++id_mat )
        {
        
//         int i = 0;
        for(int i=0; i<nb_bins; i++) // Initialize tables
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
            if(parameters.data_h.physics_list[ELECTRON_IONISATION] == true)
                {
                eIoni_DEDX_table( id_mat);
                eIoni_CrossSection_table(id_mat);
                }

            if(parameters.data_h.physics_list[ELECTRON_BREMSSTRAHLUNG] == true)
                {
                eBrem_DEDX_table(id_mat);
                eBrem_CrossSection_table(id_mat);
                }

            if(parameters.data_h.physics_list[ELECTRON_MSC] == true)
                {
                eMSC_CrossSection_table(id_mat);
                }
                
                
            Range_table(id_mat);
                
                
        }
        
}

void ElectronCrossSection::Range_table(int id_mat)
{

    int i,j,n;
    f32  energy,de,hsum,esum,eDXDE=0.,hDXDE=0.;

    i=0;
    n=100;


    eDXDE=data_h.eIonisationdedx[i+id_mat*nb_bins]+data_h.eBremdedx[i+id_mat*nb_bins];
    if(eDXDE>0.)
        eDXDE=2.*data_h.E[i+id_mat*nb_bins]/eDXDE;
    data_h.eRange[i+id_mat*nb_bins]=eDXDE;

    for(i=1; i<nb_bins; i++)
        {
        de=(data_h.E[i+id_mat*nb_bins]-data_h.E[i-1+id_mat*nb_bins])/n;
        energy=data_h.E[i+id_mat*nb_bins]+de*.5;

        esum=0.;
        for(j=0; j<n; j++)
            {
            energy-=de;
            eDXDE=GetDedx(energy,id_mat);//+GetDedx(energy,2);
            if(eDXDE>0.)
                esum+=de/eDXDE;
            }
        data_h.eRange[i+id_mat*nb_bins]=data_h.eRange[i-1+id_mat*nb_bins]+esum;
        }



}

f32 ElectronCrossSection::GetDedx(f32 Energy,int material) // get dedx eioni and ebrem sum for energy Energy in material material
    {
    int index = 0;
    index = binary_search(Energy,data_h.E, (material+1)*nb_bins,(material)*nb_bins);

    f32 DeDxeIoni = linear_interpolation(data_h.E[index]  ,data_h.eIonisationdedx[index],
                          data_h.E[index+1],data_h.eIonisationdedx[index+1],
                          Energy
                                              );

    f32 DeDxeBrem = linear_interpolation(data_h.E[index]  ,data_h.eBremdedx[index],
                          data_h.E[index+1],data_h.eBremdedx[index+1],
                          Energy
                                              );

    return DeDxeIoni + DeDxeBrem;

    }

//Print a table file per material
void ElectronCrossSection::printElectronTables(std::string dirname)
{

    for(int i = 0; i< nb_mat; ++i)
    {
        std::string tmp = dirname + ImageReader::to_string(i) + ".txt";
        ImageReader::recordTables(tmp.c_str(),i*nb_bins,(i+1)*nb_bins, 
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
    slope=log(parameters.data_h.cs_table_max_E/parameters.data_h.cs_table_min_E);
    for(int id_mat = 0; id_mat< myMaterials.nb_materials; ++id_mat )
        for(i=0; i<nb_bins; i++)
            {
            x=(f32)i;
            x/=(nb_bins-1);
            data_h.E[i+id_mat*nb_bins]=constant*exp(slope*x)*MeV;
            }
    }
    
    
void ElectronCrossSection:: eIoni_DEDX_table( int id_mat)
    {
    int i; // Index to increment energy

    for(i=0; i<nb_bins; i++)
        {

        f32 Ekine=data_h.E[i];
        data_h.eIonisationdedx[i + nb_bins * id_mat]=eIoniDEDX(Ekine,id_mat);

        }

    }
    
f32 ElectronCrossSection::eIoniDEDX( f32 Ekine,int id_mat)
    {

    f32  Dedx=0.;
    f32  th=.25*sqrt(myMaterials.nb_electrons_per_vol[id_mat]/myMaterials.nb_atoms_per_vol[id_mat])*keV;
    f32  lowLimit=.2*keV;
    f32  tmax,tkin;
    f32  eexc,eexc2,d,x,y;
    f32  tau,gamma,gamma2,beta2,bg2;

    tkin=Ekine;
    if(Ekine<th)
        tkin=th;
    tmax=tkin*.5;
    tau=tkin/electron_mass_c2;
    gamma=tau+1.;
    gamma2=gamma*gamma;
    beta2=1.-1./gamma2;
    bg2=beta2*gamma2;
    eexc=myMaterials.electron_mean_excitation_energy[id_mat]/electron_mass_c2;
    eexc2=eexc*eexc;
    d=std::min(MaxKinEnergy,tmax);
    d/=electron_mass_c2;

    Dedx=log(2.*(tau+2.)/eexc2)-1.-beta2+log((tau-d)*d)+tau/(tau-d)
         +(.5*d*d+(2.*tau+1.)*log(1.-d/tau))/gamma2;

    x=log(bg2)/(2.*log(10.));
    Dedx-=DensCorrection(x,id_mat);
    Dedx*=twopi_mc2_rcl2*myMaterials.nb_electrons_per_vol[id_mat]/beta2;

    if(Dedx<0.)
        Dedx=0.;
    if(Ekine<th)
        {
        if (Ekine>=lowLimit)
            Dedx*=sqrt(tkin/Ekine);
        else
            Dedx*=sqrt(tkin*Ekine)/lowLimit;
        }

    return    Dedx;
    }
    
    
f32 ElectronCrossSection::DensCorrection(f32 x, int id_mat)
    {
    f32  y=0.;

    if(x<myMaterials.fX0[id_mat])
        {
        if(myMaterials.fD0[id_mat]>0.)
            y=myMaterials.fD0[id_mat]*pow(10.,2.*(x-myMaterials.fX0[id_mat]));
        }
    else if(x>=myMaterials.fX1[id_mat])
        y=2.*log(10.)*x-myMaterials.fC[id_mat];
    else
        y=2.*log(10.)*x-myMaterials.fC[id_mat]+myMaterials.fA[id_mat]
          *pow(myMaterials.fX1[id_mat]-x,myMaterials.fM[id_mat]);
    return  y;
    }
    
void ElectronCrossSection::eIoni_CrossSection_table(int id_mat)
    {
    int i;

    for(i=0; i<nb_bins; i++)
        {
        f32 Ekine=data_h.E[i];
        data_h.eIonisationCS[i + nb_bins * id_mat]=eIoniCrossSection(id_mat, Ekine);
        }
    }
    
f32  ElectronCrossSection::eIoniCrossSection(int id_mat, f32 Ekine)
    {
    int i;
    f32  CrossTotale=0.;
    int index = myMaterials.index[id_mat]; // Get index of 1st element of the mixture

    for(i=0; i<myMaterials.nb_elements[id_mat]; ++i) // Get the sum of each element cross section
        {
        f32 tempo = myMaterials.atom_num_dens[index+i] * eIoniCrossSectionPerAtom(index+i, Ekine);
//         tempo *=myMaterials.nb_atoms_per_vol[id_mat]; // Tempo value to avoid overflow
        CrossTotale+=tempo;//myMaterials.atom_num_dens[index+i]*myMaterials.nb_atoms_per_vol[id_mat]*eIoniCrossSectionPerAtom(index+i, Ekine);
        }

    return  CrossTotale;
    }

f32 ElectronCrossSection:: eIoniCrossSectionPerAtom(int index, f32 Ekine)
    {
    f32  Cross=0.;
    f32  tmax=std::min(1.*GeV,Ekine*.5);
    f32  xmin,xmax,gamma,gamma2,beta2,g;

    if(MaxKinEnergy<tmax)
        {
        xmin=MaxKinEnergy/Ekine;
        xmax=tmax/Ekine;
        gamma=Ekine/electron_mass_c2+1.;
        gamma2=gamma*gamma;
        beta2=1.-1./gamma2;
        g=(2.*gamma-1.)/gamma2;
        Cross=((xmax-xmin)*(1.-g+1./(xmin*xmax)+1./((1.-xmin)*(1.-xmax)))
               -g*std::log(xmax*(1.-xmin)/(xmin*(1.-xmax))))/beta2;

        Cross*=twopi_mc2_rcl2/Ekine;
        }
    Cross*=myMaterials.mixture[index];

    return  Cross;
    }
    
    
    
void ElectronCrossSection::eBrem_DEDX_table(int id_mat)
    {
    int i;

    for(i=0; i<nb_bins; i++)
        {
        f32 Ekine=data_h.E[i];
        data_h.eBremdedx[i + nb_bins * id_mat]=eBremDEDX(Ekine,id_mat)*mm2;    //G4 internal unit
        }
    }
    
f32 ElectronCrossSection::eBremDEDX(f32 Ekine,int id_mat) //id_mat = index material
    {
    int i,n,nn,nmax;
    f32  Dedx;
    f32  totalEnergy,Z,natom,kp2,kmin,kmax,floss;
    f32  vmin,vmax,u,fac,c,v,dv;
    f32  thigh=100.*GeV;
    f32  cut=std::min(cutEnergyGamma,Ekine);
    f32  rate,loss;
    f32  factorHigh=36./(1450.*GeV);
    f32  coef1=-.5;
    f32  coef2=2./9.;
    f32  lowKinEnergy=0.*eV;
    f32  highKinEnergy=1.*GeV;
    f32  probsup=1.;
    f32  MigdalConstant=elec_radius*hbarc*hbarc*4.*pi/(electron_mass_c2*electron_mass_c2);

    totalEnergy=Ekine+electron_mass_c2;
    Dedx=0.;

    if(Ekine<lowKinEnergy)
        return  0.;

    for(i=0; i<myMaterials.nb_elements[id_mat]; ++i) // Check in each elt
        {
        int indexelt= i + myMaterials.index[id_mat];
        Z=myMaterials.mixture[indexelt];
        natom=myMaterials.atom_num_dens[indexelt]/myMaterials.nb_atoms_per_vol[id_mat];
        if(Ekine<=thigh)
            loss=eBremLoss(Z,Ekine,cut);
        loss*=natom;
        kp2=MigdalConstant*totalEnergy*totalEnergy*myMaterials.nb_electrons_per_vol[id_mat];

        kmin=1.*eV;
        kmax=cut;
        if(kmax>kmin)
            {
            floss=0.;
            nmax=100;
            vmin=log(kmin);
            vmax=log(kmax);
            nn=(int)(nmax*(vmax-vmin)/(log(highKinEnergy)-vmin)) ;
            if(nn>0)
                {
                dv=(vmax-vmin)/nn;
                v=vmin-dv;
                for(n=0; n<=nn; n++)
                    {
                    v+=dv;
                    u=exp(v);
                    //fac=u*SupressionFunction(material,Ekine,u);   //LPM flag off
                    fac=u*1.;
                    fac*=probsup*(u*u/(u*u+kp2))+1.-probsup;
                    if((n==0)||(n==nn))
                        c=.5;
                    else
                        c=1.;
                    fac*=c;
                    floss+=fac ;
                    }
                floss*=dv/(kmax-kmin);
                }
            else
                floss=1.;
            if(floss>1.)
                floss=1.;
            loss*=floss;
            }
        Dedx+=loss;
        }
    if(Dedx<0.)
        Dedx=0.;
    Dedx*=myMaterials.nb_atoms_per_vol[id_mat];
    return  Dedx;
    }
    
f32 ElectronCrossSection::eBremLoss(f32 Z,f32 T,f32 Cut)
    {

    int   i,j;
    int   NZ=8,Nloss=11,iz=0;
    f32    Loss;
    f32    dz,xx,yy,fl,E;
    f32    aaa=.414,bbb=.345,ccc=.460,delz=1.e6;
    f32  beta=1.0,ksi=2.0,clossh=.254,closslow=1./3.,alosslow=1.;
    f32    Tlim=10.*MeV,xlim=1.2;

    for(i=0; i<NZ; i++)
        {
        dz=fabs(Z-ZZ[i]);
        if(dz<delz)
            {
            iz=i;
            delz=dz;
            }
        }
    xx=log10(T);
    fl=1.;
    if(xx<=xlim)
        {
        xx/=xlim;
        yy=1.;
        fl=0.;
        for(j=0; j<Nloss; j++)
            {
            fl+=yy+coefloss[iz][j];
            yy*=xx;
            }
        if(fl<.00001)
            fl=.00001;
        else if(fl>1.)
            fl=1.;
        }

    E=T+electron_mass_c2;
    Loss=Z*(Z+ksi)*E*E/(T+E)*exp(beta*log(Cut/T))*(2.-clossh*exp(log(Z)/4.));
    if(T<=Tlim)
        Loss/=exp(closslow*log(Tlim/T));
    if(T<=Cut)
        Loss*=exp(alosslow*log(T/Cut));
    Loss*=(aaa+bbb*T/Tlim)/(1.+ccc*T/Tlim);
    Loss*=fl;
    Loss/=N_avogadro;

    return  Loss;
    }
    
    
void ElectronCrossSection::eBrem_CrossSection_table(int id_mat)
    {

    for(int i=0; i<nb_bins; i++)
        {
        f32 Ekine=data_h.E[i];
        data_h.eBremCS[i + nb_bins * id_mat]=eBremCrossSection(Ekine,id_mat)*mm2;  //G4 internal unit;
        }
    }
    
f32 ElectronCrossSection::eBremCrossSection(f32 Ekine,int id_mat)
    {
    f32 CrossTotale=0.;
    CrossTotale=eBremCrossSectionPerVolume(Ekine, id_mat);
    return  CrossTotale;
    }
    
f32 ElectronCrossSection::eBremCrossSectionPerVolume(f32 Ekine, int id_mat)
    {
    int i,n,nn,nmax=100;
    f32  Cross=0.;
    f32  kmax,kmin,vmin,vmax,totalEnergy,kp2;
    f32  u,fac,c,v,dv,y;
    f32  tmax=std::min(MaxKinEnergy,Ekine);
    f32  cut=std::max(cutEnergyGamma,(f32).1*(f32)keV);
    f32  fsig=0.;
    f32  highKinEnergy=1.*GeV;
    f32  probsup=1.;
    f32  MigdalConstant=elec_radius*hbarc*hbarc*4.*pi/(electron_mass_c2*electron_mass_c2);

    if(cut>=tmax)
        return Cross;

    for(i=0; i<myMaterials.nb_elements[id_mat]; i++)
        {
        int indexelt= i + myMaterials.index[id_mat];

        Cross+=myMaterials.atom_num_dens[indexelt]
               *eBremCrossSectionPerAtom(myMaterials.mixture[indexelt],cut, Ekine);
        if(tmax<Ekine)
            Cross-=myMaterials.atom_num_dens[indexelt]
                   *eBremCrossSectionPerAtom(myMaterials.mixture[indexelt],tmax,Ekine);
        }

    kmax=tmax;
    kmin=cut;
    totalEnergy=Ekine+electron_mass_c2;
    kp2=MigdalConstant*totalEnergy*totalEnergy*myMaterials.nb_electrons_per_vol[id_mat];
    vmin=log(kmin);
    vmax=log(kmax) ;
    nn=(int)(nmax*(vmax-vmin)/(log(highKinEnergy)-vmin));
    if(nn>0)
        {
        dv=(vmax-vmin)/nn;
        v=vmin-dv;
        for(n=0; n<=nn; n++)
            {
            v+=dv;
            u=exp(v);
            //fac=SupressionFunction(material,Ekine,u);     //LPM flag is off
            fac=1.;
            y=u/kmax;
            fac*=(4.-4.*y+3.*y*y)/3.;
            fac*=probsup*(u*u/(u*u+kp2))+1.-probsup;
            if((n==0)||(n==nn))
                c=.5;
            else
                c=1.;
            fac*=c;
            fsig+=fac;
            }
        y=kmin/kmax;
        fsig*=dv/(-4.*log(y)/3.-4.*(1.-y)/3.+0.5*(1.-y*y));
        }
    else
        fsig=1.;
    if(fsig>1.)
        fsig=1.;
    Cross*=fsig;

    return Cross;
    }

f32 ElectronCrossSection::eBremCrossSectionPerAtom(f32 Z,f32 cut, f32 Ekine)
    {
    int i,j,iz=0,NZ=8,Nsig=11;
    f32    Cross=0.;
    f32    ksi=2.,alfa=1.;
    f32    csigh=.127,csiglow=.25,asiglow=.02*MeV;
    f32    Tlim=10.*MeV;
    f32    xlim=1.2,delz=1.E6,absdelz;
    f32    xx,fs;

    if (Ekine<1.*keV||Ekine<cut)
        return  Cross;

    for(i=0; i<NZ; i++)
        {
        absdelz=fabs(Z-ZZ[i]);
        if(absdelz<delz)
            {
            iz=i;
            delz=absdelz;
            }
        }

    xx=log10(Ekine);
    fs=1.;
    if(xx<=xlim)
        {
        fs=coefsig[iz][Nsig-1];
        for(j=Nsig-2; j>=0; j--)
            fs=fs*xx+coefsig[iz][j];
        if(fs<0.)
            fs=0.;
        }
    Cross=Z*(Z+ksi)*(1.-csigh*exp(log(Z)/4.))*pow(log(Ekine/cut),alfa);

    if(Ekine<=Tlim)
        Cross*=exp(csiglow*log(Tlim/Ekine))*(1.+asiglow/(sqrt(Z)*Ekine));
    Cross*=fs/N_avogadro;
    if(Cross<0.)
        Cross=0.;

    return  Cross;
    }
    
void ElectronCrossSection::eMSC_CrossSection_table(int id_mat)
    {
    int i;
    for(i=0; i<nb_bins; i++)
        {
        f32 Ekine=data_h.E[i];
        data_h.eMSC[i + nb_bins * id_mat]=eMscCrossSection(Ekine,id_mat);
        }
    }

f32 ElectronCrossSection::eMscCrossSection(f32 Ekine, int id_mat)
    {
    int i;
    f32 CrossTotale=0.;

    for(i=0; i<myMaterials.nb_elements[id_mat]; i++)
        {
        int indexelt= i + myMaterials.index[id_mat];
        CrossTotale+=myMaterials.atom_num_dens[indexelt]
                     *eMscCrossSectionPerAtom(Ekine, myMaterials.mixture[indexelt]);
        }

    return  CrossTotale;
    }
    
    
f32 ElectronCrossSection::eMscCrossSectionPerAtom(f32 Ekine,  unsigned short int AtomNumber)
    {
    f32 AtomicNumber = (f32)AtomNumber;

    int iZ=14,iT=21;
    f32  Cross=0.;
    f32  eKin,eTot,T,E;
    f32  beta2,bg2,b2big,b2small,ratb2,Z23,tau,w;
    f32  Z1,Z2,ratZ;
    f32  c,c1,c2,cc1,cc2,corr;
    f32  Tlim=10.*MeV;
    f32  sigmafactor=2.*pi*elec_radius*elec_radius;
    f32  epsfactor=2.*electron_mass_c2*electron_mass_c2*Bohr_radius*Bohr_radius/(hbarc*hbarc);
    f32  eps,epsmin=1.e-4,epsmax=1.e10;
    f32  beta2lim=Tlim*(Tlim+2.*electron_mass_c2)/((Tlim+electron_mass_c2)*(Tlim+electron_mass_c2));
    f32  bg2lim=Tlim*(Tlim+2.*electron_mass_c2)/(electron_mass_c2*electron_mass_c2);

    Z23=2.*log(AtomicNumber)/3.;
    Z23=exp(Z23);



    tau=Ekine/electron_mass_c2;
    c=electron_mass_c2*tau*(tau+2.)/(electron_mass_c2*(tau+1.)); // a simplifier
    w=c-2.;
    tau=.5*(w+sqrt(w*w+4.*c));
    eKin=electron_mass_c2*tau;


    eTot=eKin+electron_mass_c2;
    beta2=eKin*(eTot+electron_mass_c2)/(eTot*eTot);
    bg2=eKin*(eTot+electron_mass_c2)/(electron_mass_c2*electron_mass_c2);
    eps=epsfactor*bg2/Z23;
    if(eps<epsmin)
        Cross=2.*eps*eps;
    else if(eps<epsmax)
        Cross=log(1.+2.*eps)-2.*eps/(1.+2.*eps);
    else
        Cross=log(2.*eps)-1.+1./eps;
    Cross*=AtomicNumber*AtomicNumber/(beta2*bg2);

    while((iZ>=0)&&(Zdat[iZ]>=AtomicNumber))
        iZ-=1;
    if(iZ==14)
        iZ=13;
    if(iZ==-1)
        iZ=0;
    Z1=Zdat[iZ];
    Z2=Zdat[iZ+1];
    ratZ=(AtomicNumber-Z1)*(AtomicNumber+Z1)/((Z2-Z1)*(Z2+Z1));

    if(eKin<=Tlim)
        {
        while((iT>=0)&&(Tdat[iT]>=eKin))
            iT-=1;
        if(iT==21)
            iT=20;
        if(iT==-1)
            iT=0;
        T=Tdat[iT];
        E=T+electron_mass_c2;
        b2small=T*(E+electron_mass_c2)/(E*E);
        T=Tdat[iT+1];
        E=T+electron_mass_c2;
        b2big=T*(E+electron_mass_c2)/(E*E);
        ratb2=(beta2-b2small)/(b2big-b2small);

        c1=celectron[iZ][iT];
        c2=celectron[iZ+1][iT];
        cc1=c1+ratZ*(c2-c1);
        c1=celectron[iZ][iT+1];
        c2=celectron[iZ+1][iT+1];
        cc2=c1+ratZ*(c2-c1);
        corr=cc1+ratb2*(cc2-cc1);
        Cross*=sigmafactor/corr;


        }
    else
        {
        c1=bg2lim*sig0[iZ]*(1.+hecorr[iZ]*(beta2-beta2lim))/bg2;
        c2=bg2lim*sig0[iZ+1]*(1.+hecorr[iZ+1]*(beta2-beta2lim))/bg2;
        if((AtomicNumber>=Z1)&&(AtomicNumber<=Z2))
            Cross=c1+ratZ*(c2-c1);
        else if(AtomicNumber<Z1)
            Cross=AtomicNumber*AtomicNumber*c1/(Z1*Z1);
        else if(AtomicNumber>Z2)
            Cross=AtomicNumber*AtomicNumber*c2/(Z2*Z2);
        }
    return  Cross;
    }
#endif
