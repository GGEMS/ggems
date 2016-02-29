// GGEMS Copyright (C) 2015

/*!
 * \file physical_constants.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date February, 11 2016
 *
 * Extract from CLHEP v2.2.0.4
 *
 */

#ifndef PHYSICAL_CONSTANTS_CUH
#define PHYSICAL_CONSTANTS_CUH

#include "global.cuh"

/// TODO clean this part - JB

// Pi
#define gpu_pi               3.141592653589793116
#define gpu_twopi            2.0*gpu_pi

// For electrons phys
#define elec_radius          (2.8179409421853486E-15*m)      // Metre
#define N_avogadro           (6.0221367E+23/mole)

//
//
//
static const  f32 Avogadro = 6.02214179e+23/mole;

//
// c   = 299.792458 mm/ns
// c^2 = 898.7404 (mm/ns)^2
//
static const  f32 c_light   = 2.99792458e+8 * m/s;
static const  f32 c_squared = c_light * c_light;

//
// h     = 4.13566e-12 MeV*ns
// hbar  = 6.58212e-13 MeV*ns
// hbarc = 197.32705e-12 MeV*mm
//
static const  f32 h_Planck      = 6.62606896e-34 * joule*s;
static const  f32 hbar_Planck   = h_Planck/twopi;
static const  f32 hbarc         = hbar_Planck * c_light;
static const  f32 hbarc_squared = hbarc * hbarc;

//
//
//
static const  f32 electron_charge = - eplus; // see SystemOfUnits.h
static const  f32 e_squared = eplus * eplus;

//
// amu_c2 - atomic equivalent mass unit
//        - AKA, unified atomic mass unit (u)
// amu    - atomic mass unit
//
static const  f32 electron_mass_c2 = 0.510998910 * MeV;
static const  f32   proton_mass_c2 = 938.272013 * MeV;
static const  f32  neutron_mass_c2 = 939.56536 * MeV;
static const  f32           amu_c2 = 931.494028 * MeV;
static const  f32              amu = amu_c2/c_squared;

//
// permeability of free space mu0    = 2.01334e-16 Mev*(ns*eplus)^2/mm
// permittivity of free space epsil0 = 5.52636e+10 eplus^2/(MeV*mm)
//
static const  f32 mu0      = 4*pi*1.e-7 * henry/m;
static const  f32 epsilon0 = 1./(c_squared*mu0);

//
// electromagnetic coupling = 1.43996e-12 MeV*mm/(eplus^2)
//
static const  f32 elm_coupling           = e_squared/(4*pi*epsilon0);
static const  f32 fine_structure_const   = elm_coupling/hbarc;
static const  f32 classic_electr_radius  = elm_coupling/electron_mass_c2;
static const  f32 electron_Compton_length = hbarc/electron_mass_c2;
static const  f32 Bohr_radius = electron_Compton_length/fine_structure_const;

static const  f32 alpha_rcl2 = fine_structure_const
        *classic_electr_radius
        *classic_electr_radius;

static const  f32 twopi_mc2_rcl2 = twopi*electron_mass_c2
        *classic_electr_radius
        *classic_electr_radius;
//
//
//
static const  f32 k_Boltzmann = 8.617343e-11 * MeV/kelvin;

//
//
//
static const  f32 STP_Temperature = 273.15*kelvin;
static const  f32 STP_Pressure    = 1.*atmosphere;
static const  f32 kGasThreshold   = 10.*mg/cm3;

//
//
//
static const  f32 universe_mean_density = 1.e-25*g/cm3;



#endif
