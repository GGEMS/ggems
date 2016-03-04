// GGEMS Copyright (C) 2015

/*!
 * \file range_cut.cu
 * \brief Range to energy cut converter
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date 4 february 2016
 *
 *
 *
 */

#ifndef RANGE_CUT_CU
#define RANGE_CUT_CU

#include "range_cut.cuh"

//// Class LogEnergyTable ///////////////////////////

/// Constructor ///
LogEnergyTable::LogEnergyTable( f32 theEmin, f32 theEmax, size_t theNbin )
{

    if ( theEmin <= 0.0) theEmin = 1. *eV;  // Log protection can't be =0 - JB

    dBin     =  log( theEmax / theEmin ) / theNbin;    /// G4Log
    baseBin  =  log( theEmin ) / dBin;

    numberOfNodes = theNbin + 1;
    dataVector.reserve( numberOfNodes );
    binVector.reserve( numberOfNodes );

    binVector.push_back( theEmin );
    dataVector.push_back( 0.0 );

    for (size_t i=1; i < numberOfNodes-1; i++)
    {
        binVector.push_back( exp( ( baseBin+i ) * dBin ) );  /// G4Exp
        dataVector.push_back( 0.0 );
    }
    binVector.push_back( theEmax );
    dataVector.push_back( 0.0 );

    edgeMin = binVector[ 0 ];
    edgeMax = binVector[ numberOfNodes-1 ];
}

void LogEnergyTable::put_value( size_t index, f32 value )
{
    dataVector[ index ] = value;
}

f32 LogEnergyTable::get_energy( size_t index )
{
    return binVector[ index ];
}

f32 LogEnergyTable::get_value( size_t index )
{
    return dataVector[ index ];
}

f32 LogEnergyTable::get_low_edge_energy( size_t index )
{
  return binVector[ index ];
}

size_t LogEnergyTable::find_bin_location( f32 theEnergy )
{
    size_t bin = size_t( std::log( theEnergy ) / dBin - baseBin );
    if ( bin + 2 > numberOfNodes )
    {
        bin = numberOfNodes - 2;
    }
    else if ( bin > 0 && theEnergy < binVector[ bin ] )
    {
        --bin;
    }
    else if ( bin + 2 < numberOfNodes && theEnergy > binVector[ bin+1 ] )
    {
        ++bin;
    }

    return bin;
}

size_t LogEnergyTable::find_bin( f32 e, size_t idx )
{
  size_t id = idx;
  if ( e < binVector[ 1 ] )
  {
    id = 0;
  }
  else if( e >= binVector[ numberOfNodes-2 ] )
  {
    id = numberOfNodes - 2;
  } else if ( idx >= numberOfNodes || e < binVector[ idx ]
              || e > binVector[ idx+1 ] )
  {
    id = find_bin_location( e );
  }
  return id;
}

f32 LogEnergyTable::interpolation( size_t idx, f32 e )
{
  return dataVector[ idx ] +
         ( dataVector[ idx + 1 ] - dataVector[ idx ] ) * ( e - binVector[ idx ] )
         /( binVector[ idx + 1 ] - binVector[ idx ] );
}

f32 LogEnergyTable::value( f32 energy )
{
    size_t lastIdx = 0;
    f32 y;
    if ( energy <= edgeMin )
    {
        lastIdx = 0;
        y = dataVector[ 0 ];
    }
    else if ( energy >= edgeMax )
    {
        lastIdx = numberOfNodes - 1;
        y = dataVector[ lastIdx ];
    }
    else
    {
        lastIdx = find_bin( energy, lastIdx );
        y = interpolation( lastIdx, energy );
    }

    return y;
}


//// Class Range Cut ////////////////////////////////

/// Constructor ///

RangeCut::RangeCut()
{
    // Global
    TotBin = 300;
    LowestEnergy = 250 *eV;
    HighestEnergy = 100.0e6 *MeV;
    MaxEnergyCut = 10.0 *GeV;

    // For gamma
    gZ = -1;
    s200keV = 0.; s1keV = 0.;
    tmin = 0.;    tlow = 0.;
    smin = 0.;    slow = 0.;
    cmin = 0.;    clow = 0.; chigh = 0.;

    // For e-
    Mass = electron_mass_c2;
    eZ = -1.;
    taul = 0.0;
    ionpot = 0.0;
    ionpotlog = -1.0e-10;
    bremfactor = 0.1;
}

/// private ///

// from G4RToEConvForGamma
f32 RangeCut::compute_gamma_cross_sections(f32 AtomicNumber, f32 KineticEnergy)
{
    //  Compute the "absorption" cross section of the photon "absorption"
    //  cross section means here the sum of the cross sections of the
    //  pair production, Compton scattering and photoelectric processes
    const  f32 t1keV = 1.*keV;
    const  f32 t200keV = 200.*keV;
    const  f32 t100MeV = 100.*MeV;

    //  compute Z dependent quantities in the case of a new AtomicNumber
    if ( std::abs( AtomicNumber-gZ ) > 0.1)
    {
        gZ = AtomicNumber;
        f32 Zsquare = gZ*gZ;
        f32 Zlog = std::log(gZ);
        f32 Zlogsquare = Zlog*Zlog;

        s200keV = ( 0.2651 - 0.1501*Zlog + 0.02283*Zlogsquare ) * Zsquare;
        tmin = ( 0.552 + 218.5/gZ + 557.17/Zsquare ) *MeV;
        smin = ( 0.01239 + 0.005585*Zlog - 0.000923*Zlogsquare ) * std::exp( 1.5*Zlog );
        cmin = std::log( s200keV/smin ) / ( std::log( tmin/t200keV ) * std::log( tmin/t200keV ) );
        tlow = 0.2*std::exp( -7.355/std::sqrt( gZ ) ) *MeV;
        slow = s200keV*std::exp( 0.042*gZ*std::log( t200keV/tlow )*std::log( t200keV/tlow ) );
        s1keV = 300.*Zsquare;
        clow = std::log( s1keV/slow ) / std::log( tlow/t1keV );

        chigh = ( 7.55e-5-0.0542e-5*gZ ) *Zsquare*gZ/std::log( t100MeV/tmin );
    }

    //  calculate the cross section (using an approximate empirical formula)
    f32 xs;
    if ( KineticEnergy < tlow )
    {
        if ( KineticEnergy < t1keV ) xs = slow*std::exp( clow*std::log( tlow/t1keV ) );
        else                         xs = slow*std::exp( clow*std::log( tlow/KineticEnergy ) );
    }
    else if ( KineticEnergy < t200keV )
    {
        xs = s200keV
                * std::exp( 0.042*gZ*std::log( t200keV/KineticEnergy )*std::log( t200keV/KineticEnergy ) );
    }
    else if( KineticEnergy < tmin )
    {
        xs = smin
                * std::exp( cmin*std::log( tmin/KineticEnergy )*std::log( tmin/KineticEnergy ) );
    }
    else
    {
        xs = smin + chigh*std::log( KineticEnergy/tmin );
    }

    return xs * barn;
}

// from G4RToEConvForElectron
f32 RangeCut::compute_electron_loss( f32 AtomicNumber, f32 KineticEnergy)
{
    const  f32 cbr1 = 0.02, cbr2 = -5.7e-5, cbr3 = 1., cbr4 = 0.072;
    const  f32 Tlow = 10.*keV, Thigh = 1.*GeV;

    //  calculate dE/dx for electrons
    if ( std::fabs( AtomicNumber-eZ )>0.1 )
    {
        eZ = AtomicNumber;
        taul = Tlow/Mass;
        ionpot = 1.6e-5*MeV * std::exp( 0.9*std::log( eZ ) ) / Mass;
        ionpotlog = std::log( ionpot );
    }

    f32 tau = KineticEnergy / Mass;
    f32 dEdx;

    if ( tau<taul )
    {
        f32 t1 = taul + 1.;
        f32 t2 = taul + 2.;
        f32 tsq = taul*taul;
        f32 beta2 = taul*t2 / ( t1*t1 );
        f32 f = 1. - beta2 + std::log( tsq/2. )
                + ( 0.5 + 0.25*tsq + ( 1.+2.*taul ) * std::log( 0.5 ) ) / ( t1*t1 );
        dEdx = ( std::log( 2.*taul + 4. ) - 2.*ionpotlog + f ) / beta2;
        dEdx = twopi_mc2_rcl2*eZ*dEdx;
        f32 clow = dEdx*std::sqrt( taul );
        dEdx = clow / std::sqrt( KineticEnergy / Mass );
    }
    else
    {
        f32 t1 = tau+1.;
        f32 t2 = tau+2.;
        f32 tsq = tau*tau;
        f32 beta2 = tau*t2 / ( t1*t1 );
        f32 f = 1. - beta2 + std::log( tsq/2. )
                + ( 0.5 + 0.25*tsq + ( 1. + 2.*tau )*std::log( 0.5 ) ) / ( t1*t1 );
        dEdx = ( std::log( 2.*tau + 4. ) - 2.*ionpotlog + f ) / beta2;
        dEdx = twopi_mc2_rcl2*eZ*dEdx;

        // loss from bremsstrahlung follows
        f32 cbrem = ( cbr1 + cbr2*eZ )
                *( cbr3 + cbr4*std::log( KineticEnergy/Thigh ) );
        cbrem = eZ * ( eZ+1. )*cbrem*tau/beta2;
        cbrem *= bremfactor;
        dEdx += twopi_mc2_rcl2*cbrem;
    }

    return dEdx;
}

// Build CS table for gamma
void RangeCut::build_gamma_cross_sections_table(const ui16* mixture, ui16 NumberOfElements, ui32 abs_index)
{
    //  Build CS tables for elements
    gamma_CS_table.reserve( NumberOfElements );

    // fill the CS table
    for ( size_t j=0; j<size_t( NumberOfElements ); j++)
    {
        f32 Value;
        LogEnergyTable* aVector = 0;
        aVector = new LogEnergyTable ( LowestEnergy, MaxEnergyCut, TotBin );

        for ( size_t i = 0; i <= size_t( TotBin ); i++)
        {
            //                                       Z
            Value = compute_gamma_cross_sections( mixture[ abs_index+j ], aVector->get_energy( i ) );
            aVector->put_value( i, Value );

        }
        gamma_CS_table.push_back( aVector );

    } // for

}

// Build eLoss table for e-
void RangeCut::build_electron_loss_table(const ui16* mixture, ui16 NumberOfElements, ui32 abs_index)
{
    //  Build CS tables for elements
    e_loss_table.reserve( NumberOfElements );

    // fill the CS table
    for ( size_t j=0; j<size_t( NumberOfElements ); j++)
    {
        f32 Value;
        LogEnergyTable* aVector = 0;
        aVector = new LogEnergyTable ( LowestEnergy, MaxEnergyCut, TotBin );

        for ( size_t i = 0; i <= size_t( TotBin ); i++)
        {
            //                                       Z
            Value = compute_electron_loss( mixture[ abs_index+j ], aVector->get_energy( i ) );
            aVector->put_value( i, Value );
        }
        e_loss_table.push_back( aVector );

    } // for
}

//  create range table for a material (gamma)
void RangeCut::gamma_build_range_table( const ui16* mixture, ui16 nb_elts, const f32 *atom_num_dens, ui32 abs_index,
                                        LogEnergyTable *range_table )
{              
    for ( size_t ib=0; ib<=size_t( TotBin ); ib++)
    {
        f32 sigma = 0.;
        for (size_t i=0; i<size_t( nb_elts ); i++)
        {
            sigma += atom_num_dens[ abs_index+i ] * ( gamma_CS_table[ i ] )->get_value( ib );
        }

        range_table->put_value( ib, 5.0f / sigma );
    }

}

//  create range table for a material (e-)
void RangeCut::electron_build_range_table( const ui16* mixture, ui16 nb_elts, const f32 *atom_num_dens, ui32 abs_index,
                                           LogEnergyTable* range_table)
{
    // calculate parameters of the low energy part first
    size_t i;
    std::vector< f32 > lossV;

    for ( size_t ib=0; ib<=size_t( TotBin ); ib++)
    {
        f32 loss = 0.;
        for ( i=0; i<size_t( nb_elts ); i++)
        {                        
            loss += atom_num_dens[ abs_index+i ] * ( e_loss_table[ i ] )->get_value( ib );
        }
        lossV.push_back( loss );
    }

    // Integrate with Simpson formula with logarithmic binning
    f32 dltau = 1.0;
    if ( LowestEnergy > 0. )
    {
        f32 ltt = std::log( MaxEnergyCut / LowestEnergy );
        dltau = ltt / TotBin;
    }

    f32 s0 = 0.;
    f32 Value;
    for ( i=0; i<=size_t( TotBin ); i++ ) {
        f32 t = range_table->get_low_edge_energy( i );
        f32 q = t / lossV[ i ];
        if ( i==0 ) s0 += 0.5*q;
        else s0 += q;

        if ( i==0 )
        {
            Value = ( s0 + 0.5*q ) * dltau;
        }
        else
        {
            Value = ( s0 - 0.5*q ) * dltau;
        }
        range_table->put_value( i, Value );
    }
}

// Convert cut to energy
f32 RangeCut::convert_cut_to_energy( LogEnergyTable *rangeVector, f32 theCutInLength )
{
    const f32 epsilon = 0.01;

    //  find max. range and the corresponding energy (rmax,Tmax)
    f32 rmax = -1.e10 *mm;

    f32 T1 = LowestEnergy;
    //f32 r1 = ( *rangeVector )[ 0 ] ;
    f32 r1 = rangeVector->get_value( 0 );

    f32 T2 = MaxEnergyCut;

    // check theCutInLength < r1
    if ( theCutInLength <= r1 ) return T1;

    // scan range vector to find nearest bin
    // ( suppose that r(Ti) > r(Tj) if Ti >Tj )
    for ( size_t ibin = 0; ibin <= size_t( TotBin ); ibin++ )
    {
        f32 T = rangeVector->get_low_edge_energy( ibin );
        //f32 r = ( *rangeVector )[ ibin ];
        f32 r = rangeVector->get_value( ibin );
        if ( r > rmax ) rmax = r;
        if ( r < theCutInLength )
        {
            T1 = T;
            r1 = r;
        }
        else if ( r > theCutInLength )
        {
            T2 = T;
            break;
        }

        //printf("T %e r %e\n", T, r);

    }

    //printf("T1 %e T2 %e r1 %e\n", T1, T2, r1);

    // check cut in length is smaller than range max
    if ( theCutInLength >= rmax )
    {
        return  MaxEnergyCut;
    }

    // convert range to energy
    f32 T3 = std::sqrt( T1*T2 );
    f32 r3 = rangeVector->value( T3 );    // HERE

    //printf("T3 %e r3 %e\n", T3, r3);

    while ( std::fabs(1.-r3/theCutInLength)>epsilon ) {
        if ( theCutInLength <= r3 ) {
            T2 = T3;
        } else {
            T1 = T3;
        }
        T3 = std::sqrt( T1*T2 );
        r3 = rangeVector->value( T3 );
    }

    //printf("\nT3 %e\n", T3);

    return T3;
}

/// Public ///

void RangeCut::set_energy_range(f32 lowE, f32 highE)
{
    LowestEnergy = lowE;
    HighestEnergy = highE;
}

f32 RangeCut::convert_gamma( f32 rangeCut, const ui16* mixture, ui16 nb_elts, const f32 *atom_num_dens, ui32 abs_index )
{

    // init vars
    f32 theKineticEnergyCuts = 0.;

    // Build the energy CS table
    build_gamma_cross_sections_table(mixture, nb_elts, abs_index);

    // Clear and init range table
    gamma_range_table = new LogEnergyTable( LowestEnergy, MaxEnergyCut, TotBin );
    gamma_build_range_table( mixture, nb_elts, atom_num_dens, abs_index, gamma_range_table );

    // Convert Range Cut ro Kinetic Energy Cut
    theKineticEnergyCuts = convert_cut_to_energy( gamma_range_table, rangeCut );

    if ( theKineticEnergyCuts < LowestEnergy )
    {
      theKineticEnergyCuts = LowestEnergy ;
    }
    else if ( theKineticEnergyCuts > MaxEnergyCut )
    {
      theKineticEnergyCuts = MaxEnergyCut;
    }

    return theKineticEnergyCuts;
}

f32 RangeCut::convert_electron( f32 rangeCut, const ui16* mixture, ui16 nb_elts, const f32 *atom_num_dens, f32 density, ui32 abs_index )
{

    // init vars
    f32 theKineticEnergyCuts = 0.;

    // Build the energy CS table   
    build_electron_loss_table(mixture, nb_elts, abs_index);

    // Clear and init range table
    electron_range_table = new LogEnergyTable( LowestEnergy, MaxEnergyCut, TotBin );
    electron_build_range_table( mixture, nb_elts, atom_num_dens, abs_index, electron_range_table );

    // Convert Range Cut ro Kinetic Energy Cut
    theKineticEnergyCuts = convert_cut_to_energy( electron_range_table, rangeCut );

    static const f32 tune = 0.025*mm*g/cm3, lowen = 30.*keV ;

    if ( theKineticEnergyCuts < lowen )  {
      //  corr. should be switched on smoothly
      theKineticEnergyCuts /= ( 1. + ( 1. - theKineticEnergyCuts/lowen ) *
                              tune / ( rangeCut*density ) );
    }

    if ( theKineticEnergyCuts < LowestEnergy )
    {
      theKineticEnergyCuts = LowestEnergy ;
    }
    else if ( theKineticEnergyCuts > MaxEnergyCut )
    {
      theKineticEnergyCuts = MaxEnergyCut;
    }

    return theKineticEnergyCuts;
}


#endif

















