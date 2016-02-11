// GGEMS Copyright (C) 2015

/*!
 * \file system_of_units.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date February, 11 2016
 *
 * Extract from CLHEP v2.2.0.4
 *
 */

#ifndef SYSTEM_OF_UNITS_CUH
#define SYSTEM_OF_UNITS_CUH

#include "global.cuh"

//
//
//
static const  f32     pi  = 3.14159265358979323846;
static const  f32  twopi  = 2*pi;
static const  f32 halfpi  = pi/2;
static const  f32     pi2 = pi*pi;

//
// Length [L]
//
static const  f32 millimeter  = 1.;
static const  f32 millimeter2 = millimeter*millimeter;
static const  f32 millimeter3 = millimeter*millimeter*millimeter;

static const  f32 centimeter  = 10.*millimeter;
static const  f32 centimeter2 = centimeter*centimeter;
static const  f32 centimeter3 = centimeter*centimeter*centimeter;

static const  f32 meter  = 1000.*millimeter;
static const  f32 meter2 = meter*meter;
static const  f32 meter3 = meter*meter*meter;

static const  f32 kilometer = 1000.*meter;
static const  f32 kilometer2 = kilometer*kilometer;
static const  f32 kilometer3 = kilometer*kilometer*kilometer;

static const  f32 parsec = 3.0856775807e+16*meter;

static const  f32 micrometer = 1.e-6 *meter;
static const  f32  nanometer = 1.e-9 *meter;
static const  f32  angstrom  = 1.e-10*meter;
static const  f32  fermi     = 1.e-15*meter;

static const  f32      barn = 1.e-28*meter2;
static const  f32 millibarn = 1.e-3 *barn;
static const  f32 microbarn = 1.e-6 *barn;
static const  f32  nanobarn = 1.e-9 *barn;
static const  f32  picobarn = 1.e-12*barn;

// symbols
static const  f32 nm  = nanometer;
static const  f32 um  = micrometer;

static const  f32 mm  = millimeter;
static const  f32 mm2 = millimeter2;
static const  f32 mm3 = millimeter3;

static const  f32 cm  = centimeter;
static const  f32 cm2 = centimeter2;
static const  f32 cm3 = centimeter3;

static const  f32 m  = meter;
static const  f32 m2 = meter2;
static const  f32 m3 = meter3;

static const  f32 km  = kilometer;
static const  f32 km2 = kilometer2;
static const  f32 km3 = kilometer3;

static const  f32 pc = parsec;

//
// Angle
//
static const  f32 radian      = 1.;
static const  f32 milliradian = 1.e-3*radian;
static const  f32 degree = (pi/180.0)*radian;

static const  f32   steradian = 1.;

// symbols
static const  f32 rad  = radian;
static const  f32 mrad = milliradian;
static const  f32 sr   = steradian;
static const  f32 deg  = degree;

//
// Time [T]
//
static const  f32 nanosecond  = 1.;
static const  f32 second      = 1.e+9 *nanosecond;
static const  f32 millisecond = 1.e-3 *second;
static const  f32 microsecond = 1.e-6 *second;
static const  f32  picosecond = 1.e-12*second;

static const  f32 hertz = 1./second;
static const  f32 kilohertz = 1.e+3*hertz;
static const  f32 megahertz = 1.e+6*hertz;

// symbols
static const  f32 ns = nanosecond;
static const  f32  s = second;
static const  f32 ms = millisecond;

//
// Electric charge [Q]
//
static const  f32 eplus = 1. ;// positron charge
static const  f32 e_SI  = 1.602176487e-19;// positron charge in coulomb
static const  f32 coulomb = eplus/e_SI;// coulomb = 6.24150 e+18 * eplus

//
// Energy [E]
//
static const  f32 megaelectronvolt = 1. ;
static const  f32     electronvolt = 1.e-6*megaelectronvolt;
static const  f32 kiloelectronvolt = 1.e-3*megaelectronvolt;
static const  f32 gigaelectronvolt = 1.e+3*megaelectronvolt;
static const  f32 teraelectronvolt = 1.e+6*megaelectronvolt;
static const  f32 petaelectronvolt = 1.e+9*megaelectronvolt;

static const  f32 joule = electronvolt/e_SI;// joule = 6.24150 e+12 * MeV

// symbols
static const  f32 MeV = megaelectronvolt;
static const  f32  eV = electronvolt;
static const  f32 keV = kiloelectronvolt;
static const  f32 GeV = gigaelectronvolt;
static const  f32 TeV = teraelectronvolt;
static const  f32 PeV = petaelectronvolt;

//
// Mass [E][T^2][L^-2]
//
static const  f32  kilogram = joule*second*second/(meter*meter);
static const  f32      gram = 1.e-3*kilogram;
static const  f32 milligram = 1.e-3*gram;

// symbols
static const  f32  kg = kilogram;
static const  f32   g = gram;
static const  f32  mg = milligram;

//
// Power [E][T^-1]
//
static const  f32 watt = joule/second;// watt = 6.24150 e+3 * MeV/ns

//
// Force [E][L^-1]
//
static const  f32 newton = joule/meter;// newton = 6.24150 e+9 * MeV/mm

//
// Pressure [E][L^-3]
//
#define pascal hep_pascal                          // a trick to avoid warnings
static const  f32 hep_pascal = newton/m2;   // pascal = 6.24150 e+3 * MeV/mm3
static const  f32 bar        = 100000*pascal; // bar    = 6.24150 e+8 * MeV/mm3
static const  f32 atmosphere = 101325*pascal; // atm    = 6.32420 e+8 * MeV/mm3

//
// Electric current [Q][T^-1]
//
static const  f32      ampere = coulomb/second; // ampere = 6.24150 e+9 * eplus/ns
static const  f32 milliampere = 1.e-3*ampere;
static const  f32 microampere = 1.e-6*ampere;
static const  f32  nanoampere = 1.e-9*ampere;

//
// Electric potential [E][Q^-1]
//
static const  f32 megavolt = megaelectronvolt/eplus;
static const  f32 kilovolt = 1.e-3*megavolt;
static const  f32     volt = 1.e-6*megavolt;

//
// Electric resistance [E][T][Q^-2]
//
static const  f32 ohm = volt/ampere;// ohm = 1.60217e-16*(MeV/eplus)/(eplus/ns)

//
// Electric capacitance [Q^2][E^-1]
//
static const  f32 farad = coulomb/volt;// farad = 6.24150e+24 * eplus/Megavolt
static const  f32 millifarad = 1.e-3*farad;
static const  f32 microfarad = 1.e-6*farad;
static const  f32  nanofarad = 1.e-9*farad;
static const  f32  picofarad = 1.e-12*farad;

//
// Magnetic Flux [T][E][Q^-1]
//
static const  f32 weber = volt*second;// weber = 1000*megavolt*ns

//
// Magnetic Field [T][E][Q^-1][L^-2]
//
static const  f32 tesla     = volt*second/meter2;// tesla =0.001*megavolt*ns/mm2

static const  f32 gauss     = 1.e-4*tesla;
static const  f32 kilogauss = 1.e-1*tesla;

//
// Inductance [T^2][E][Q^-2]
//
static const  f32 henry = weber/ampere;// henry = 1.60217e-7*MeV*(ns/eplus)**2

//
// Temperature
//
static const  f32 kelvin = 1.;

//
// Amount of substance
//
static const  f32 mole = 1.;

//
// Activity [T^-1]
//
static const  f32 becquerel = 1./second ;
static const  f32 curie = 3.7e+10 * becquerel;
static const  f32 kilobecquerel = 1.e+3*becquerel;
static const  f32 megabecquerel = 1.e+6*becquerel;
static const  f32 gigabecquerel = 1.e+9*becquerel;
static const  f32 millicurie = 1.e-3*curie;
static const  f32 microcurie = 1.e-6*curie;
static const  f32 Bq = becquerel;
static const  f32 kBq = kilobecquerel;
static const  f32 MBq = megabecquerel;
static const  f32 GBq = gigabecquerel;
static const  f32 Ci = curie;
static const  f32 mCi = millicurie;
static const  f32 uCi = microcurie;

//
// Absorbed dose [L^2][T^-2]
//
static const  f32      gray = joule/kilogram ;
static const  f32  kilogray = 1.e+3*gray;
static const  f32 milligray = 1.e-3*gray;
static const  f32 microgray = 1.e-6*gray;

//
// Luminous intensity [I]
//
static const  f32 candela = 1.;

//
// Luminous flux [I]
//
static const  f32 lumen = candela*steradian;

//
// Illuminance [I][L^-2]
//
static const  f32 lux = lumen/meter2;

//
// Miscellaneous
//
static const  f32 perCent     = 0.01 ;
static const  f32 perThousand = 0.001;
static const  f32 perMillion  = 0.000001;





#endif
