#ifndef GUARD_GGEMS_TOOLS_SYSTEMOFUNITS_HH
#define GUARD_GGEMS_TOOLS_SYSTEMOFUNITS_HH

/*!
  \file system_of_units.hh

  \brief Namespace storing all the usefull physical units

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Tuesday October 1, 2019
*/

#include "GGEMS/opencl/types.hh"

#ifdef PASCAL
#undef PASCAL
#endif

#ifdef __cplusplus
#define __constant inline static constexpr
#endif

/*!
  \namespace Units
  \brief namespace storing all the usefull physical units
*/
#ifdef __cplusplus
namespace Units
{
#endif
  // Pi definitions
  __constant f64cl_t PI       = 3.141592653589793238463;
  __constant f64cl_t PI_TWICE = 2.0 * 3.141592653589793238463;
  __constant f64cl_t PI_HALF  = 3.141592653589793238463 / 2.0;
  __constant f64cl_t PI_PI    = 3.141592653589793238463 * 3.141592653589793238463;

  // Lengths
  __constant f64cl_t MILLIMETER  = 1.0;
  __constant f64cl_t MILLIMETER2 = 1.0*1.0;
  __constant f64cl_t MILLIMETER3 = 1.0*1.0*1.0;

  __constant f64cl_t CENTIMETER  = 10.*1.0;
  __constant f64cl_t CENTIMETER2 = 10.*1.0*10.*1.0;
  __constant f64cl_t CENTIMETER3 = 10.*1.0*10.*1.0*10.*1.0;

  __constant f64cl_t METER  = 1000.*1.0;
  __constant f64cl_t METER2 = 1000.*1.0*1000.*1.0;
  __constant f64cl_t METER3 = 1000.*1.0*1000.*1.0*1000.*1.0;

  __constant f64cl_t KILOMETER  = 1000.*1000.*1.0;
  __constant f64cl_t KILOMETER2 = 1000.*1000.*1.0*1000.*1000.*1.0;
  __constant f64cl_t KILOMETER3 =
    1000.*1000.*1.0*1000.*1000.*1.0*1000.*1000.*1.0;

  __constant f64cl_t PARSEC = 96939420213600000*1000.*1.0/3.141592653589793238463;

  __constant f64cl_t MICROMETER = 1.e-6 *1000.*1.0;
  __constant f64cl_t NANOMETER  = 1.e-9 *1000.*1.0;
  __constant f64cl_t ANGSTROM   = 1.e-10*1000.*1.0;
  __constant f64cl_t FERMI      = 1.e-15*1000.*1.0;

  __constant f64cl_t BARN      = 1.e-28*1000.*1.0;
  __constant f64cl_t MILLIBARN = 1.e-3 *1.e-28*1000.*1.0;
  __constant f64cl_t MICROBARN = 1.e-6 *1.e-28*1000.*1.0;
  __constant f64cl_t NANOBARN  = 1.e-9 *1.e-28*1000.*1.0;
  __constant f64cl_t PICOBARN  = 1.e-12*1.e-28*1000.*1.0;

  // Symbol definitions
  __constant f64cl_t nm  = 1.e-9 *1000.*1.0;
  __constant f64cl_t um  = 1.e-6 *1000.*1.0;

  __constant f64cl_t mm = 1.0;
  __constant f64cl_t mm2 = 1.0*1.0;
  __constant f64cl_t mm3 = 1.0*1.0*1.0;

  __constant f64cl_t cm  = 10.*1.0;
  __constant f64cl_t cm2 = 10.*1.0*10.*1.0;
  __constant f64cl_t cm3 = 10.*1.0*10.*1.0*10.*1.0;

  __constant f64cl_t m  = 1000.*1.0;
  __constant f64cl_t m2 = 1000.*1.0*1000.*1.0;
  __constant f64cl_t m3 = 1000.*1.0*1000.*1.0*1000.*1.0;

  __constant f64cl_t Km  = 1000.*1000.*1.0;
  __constant f64cl_t Km2 = 1000.*1000.*1.0*1000.*1000.*1.0;
  __constant f64cl_t Km3 = 1000.*1000.*1.0*1000.*1000.*1.0*1000.*1000.*1.0;

  __constant f64cl_t pc = 96939420213600000*1000.*1.0/3.141592653589793238463;

  // Angles
  __constant f64cl_t RADIAN      = 1.0;
  __constant f64cl_t MILLIRADIAN = 1.e-3*1.0;
  __constant f64cl_t DEGREE      = (3.141592653589793238463/180.0)*1.0;

  __constant f64cl_t STERADIAN = 1.0;

  // Symbols definitions
  __constant f64cl_t rad  = 1.0;
  __constant f64cl_t mrad = 1.e-3*1.0;
  __constant f64cl_t sr   = 1.0;
  __constant f64cl_t deg  = (3.141592653589793238463/180.0)*1.0;

  // Time
  __constant f64cl_t NANOSECOND  = 1.;
  __constant f64cl_t SECOND      = 1.e+9 *1.;
  __constant f64cl_t MILLISECOND = 1.e-3 *1.e+9 *1.;
  __constant f64cl_t MICROSECOND = 1.e-6 *1.e+9 *1.;
  __constant f64cl_t PICOSECOND  = 1.e-12*1.e+9 *1.;

  __constant f64cl_t HERTZ     = 1./(1.e+9 *1.);
  __constant f64cl_t KILOHERTZ = 1.e+3*(1./(1.e+9 *1.));
  __constant f64cl_t MEGAHERTZ = 1.e+6*(1./(1.e+9 *1.));

  // Symbols definitions
  __constant f64cl_t ns = 1.;
  __constant f64cl_t s  = 1.e+9 *1.;
  __constant f64cl_t ms = 1.e-3 *1.e+9 *1.;

  // Electric charge [Q]
  __constant f64cl_t EPLUS   = 1. ;// positron charge
  __constant f64cl_t ESI     = 1.602176487e-19;// positron charge in coulomb
  __constant f64cl_t COULOMB = 1./1.602176487e-19;// coulomb = 6.24150 e+18 * eplus

  // Energy [E]
  __constant f64cl_t MEGAELECTRONVOLT = 1. ;
  __constant f64cl_t ELECTRONVOLT     = 1.e-6*1.;
  __constant f64cl_t KILOELECTRONVOLT = 1.e-3*1.;
  __constant f64cl_t GIGAELECTRONVOLT = 1.e+3*1.;
  __constant f64cl_t TERAELECTRONVOLT = 1.e+6*1.;
  __constant f64cl_t PETAELECTRONVOLT = 1.e+9*1.;

  __constant f64cl_t JOULE = 1.e-6*1./1.602176487e-19;// joule = 6.24150 e+12 * MeV

  // Symbols definitions
  __constant f64cl_t MeV = 1.;
  __constant f64cl_t eV  = 1.e-6*1.;
  __constant f64cl_t KeV = 1.e-3*1.;
  __constant f64cl_t GeV = 1.e+3*1.;
  __constant f64cl_t TeV = 1.e+6*1.;
  __constant f64cl_t PeV = 1.e+9*1.;

  // Mass [E][T^2][L^-2]
  __constant f64cl_t KILOGRAM  =
    1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0);
  __constant f64cl_t GRAM      =
    1.e-3*(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0));
  __constant f64cl_t MILLIGRAM =
    1.e-3*1.e-3*(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1.
    /(1000.*1.0*1000.*1.0));

  // Symbols definitions
  __constant f64cl_t  Kg =
    1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0);
  __constant f64cl_t  g  =
    1.e-3*(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0));
  __constant f64cl_t  Mg =
    1.e-3*1.e-3*(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1.
    /(1000.*1.0*1000.*1.0));

  // Power [E][T^-1]
  __constant f64cl_t WATT = 1.e-6*1./1.602176487e-19/(1.e+9 *1.);// watt = 6.24150 e+3 * MeV/ns

  // Force [E][L^-1]
  __constant f64cl_t NEWTON = 1.e-6*1./1.602176487e-19/(1000.*1.0);// newton = 6.24150 e+9 * MeV/mm

  // Pressure [E][L^-3]
  __constant f64cl_t PASCAL     =
    (1.e-6*1./1.602176487e-19/(1000.*1.0))/(1000.*1.0*1000.*1.0);   // pascal = 6.24150 e+3 * MeV/mm3
  __constant f64cl_t BAR        =
    100000*((1.e-6*1./1.602176487e-19/(1000.*1.0))/(1000.*1.0*1000.*1.0)); // bar    = 6.24150 e+8 * MeV/mm3
  __constant f64cl_t ATMOSPHERE =
    101325*((1.e-6*1./1.602176487e-19/(1000.*1.0))/(1000.*1.0*1000.*1.0)); // atm    = 6.32420 e+8 * MeV/mm3

  // Electric current [Q][T^-1]
  __constant f64cl_t AMPERE      = (1./1.602176487e-19)/(1.e+9 *1.); // ampere = 6.24150 e+9 * eplus/ns
  __constant f64cl_t MILLIAMPERE = 1.e-3*((1./1.602176487e-19)/(1.e+9 *1.));
  __constant f64cl_t MICROAMPERE = 1.e-6*((1./1.602176487e-19)/(1.e+9 *1.));
  __constant f64cl_t NANOAMPERE  = 1.e-9*((1./1.602176487e-19)/(1.e+9 *1.));

  // Electric potential [E][Q^-1]
  __constant f64cl_t MEGAVOLT = (1.)/(1.);
  __constant f64cl_t KILOVOLT = 1.e-3*((1.)/(1.));
  __constant f64cl_t VOLT     = 1.e-6*((1.)/(1.));

  // Electric resistance [E][T][Q^-2]
  __constant f64cl_t OHM =
    (1.e-6*((1.)/(1.)))/((1./1.602176487e-19)/(1.e+9 *1.));// ohm = 1.60217e-16*(MeV/eplus)/(eplus/ns)

  // Electric capacitance [Q^2][E^-1]
  __constant f64cl_t FARAD = (1./1.602176487e-19)/(1.e-6*((1.)/(1.)));// farad = 6.24150e+24 * eplus/Megavolt
  __constant f64cl_t MILLIFARAD =
    1.e-3*((1./1.602176487e-19)/(1.e-6*((1.)/(1.))));
  __constant f64cl_t MICROFARAD =
    1.e-6*((1./1.602176487e-19)/(1.e-6*((1.)/(1.))));
  __constant f64cl_t NANOFARAD =
    1.e-9*((1./1.602176487e-19)/(1.e-6*((1.)/(1.))));
  __constant f64cl_t PICOFARAD =
    1.e-12*((1./1.602176487e-19)/(1.e-6*((1.)/(1.))));

  // Magnetic Flux [T][E][Q^-1]
  __constant f64cl_t WEBER = (1.e-6*((1.)/(1.)))*(1.e+9 *1.);// weber = 1000*megavolt*ns

  // Magnetic Field [T][E][Q^-1][L^-2]
  __constant f64cl_t TESLA     =
    (1.e-6*((1.)/(1.)))*(1.e+9 *1.)/(1000.*1.0*1000.*1.0);// tesla =0.001*megavolt*ns/mm2

  __constant f64cl_t GAUSS     =
    1.e-4*((1.e-6*((1.)/(1.)))*(1.e+9 *1.)/(1000.*1.0*1000.*1.0));
  __constant f64cl_t KILOGAUSS =
    1.e-1*((1.e-6*((1.)/(1.)))*(1.e+9 *1.)/(1000.*1.0*1000.*1.0));

  // Inductance [T^2][E][Q^-2]
  __constant f64cl_t HENRY =
    ((1.e-6*((1.)/(1.)))*(1.e+9 *1.))/((1./1.602176487e-19)/(1.e+9 *1.));// henry = 1.60217e-7*MeV*(ns/eplus)**2

  // Temperature
  __constant f64cl_t KELVIN = 1.;

  // Amount of substance
  __constant f64cl_t MOLE = 1.;

  // Activity [T^-1]
  __constant f64cl_t BECQUEREL     = 1./(1.e+9 *1.) ;
  __constant f64cl_t CURIE         = 3.7e+10 * (1./(1.e+9 *1.));
  __constant f64cl_t KILOBECQUEREL = 1.e+3*(1./(1.e+9 *1.));
  __constant f64cl_t MEGABECQUEREL = 1.e+6*(1./(1.e+9 *1.));
  __constant f64cl_t GIGABECQUEREL = 1.e+9*(1./(1.e+9 *1.));
  __constant f64cl_t MILLICURIE    = 1.e-3*(3.7e+10 * (1./(1.e+9 *1.)));
  __constant f64cl_t MICROCURIE    = 1.e-6*(3.7e+10 * (1./(1.e+9 *1.)));

  // Symbols definitions
  __constant f64cl_t Bq  = (1./(1.e+9 *1.));
  __constant f64cl_t kBq = (1.e+3*(1./(1.e+9 *1.)));
  __constant f64cl_t MBq = (1.e+6*(1./(1.e+9 *1.)));
  __constant f64cl_t GBq = (1.e+9*(1./(1.e+9 *1.)));
  __constant f64cl_t Ci  = (3.7e+10 * (1./(1.e+9 *1.)));
  __constant f64cl_t mCi = (1.e-3*(3.7e+10 * (1./(1.e+9 *1.))));
  __constant f64cl_t uCi = (1.e-6*(3.7e+10 * (1./(1.e+9 *1.))));

  // Absorbed dose [L^2][T^-2]
  __constant f64cl_t GRAY      =
    (1.e-6*1./1.602176487e-19)/(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1.
    /(1000.*1.0*1000.*1.0));
  __constant f64cl_t KILOGRAY  =
    1.e+3*((1.e-6*1./1.602176487e-19)/(1.e-6*1./1.602176487e-19*1.e+9 *1.
    *1.e+9 *1./(1000.*1.0*1000.*1.0)));
  __constant f64cl_t MILLIGRAY =
    1.e-3*((1.e-6*1./1.602176487e-19)/(1.e-6*1./1.602176487e-19*1.e+9 *1.
    *1.e+9 *1./(1000.*1.0*1000.*1.0)));
  __constant f64cl_t MICROGRAY =
    1.e-6*((1.e-6*1./1.602176487e-19)/(1.e-6*1./1.602176487e-19*1.e+9
    *1.*1.e+9 *1./(1000.*1.0*1000.*1.0)));

  // Luminous intensity [I]
  __constant f64cl_t CANDELA = 1.;

  // Luminous flux [I]
  __constant f64cl_t LUMEN = (1.)*(1.0);

  // Illuminance [I][L^-2]
  __constant f64cl_t LUX = ((1.)*(1.0))/(1000.*1.0*1000.*1.0);

  // Miscellaneous
  __constant f64cl_t PERCENT     = 0.01 ;
  __constant f64cl_t PERTHOUSAND = 0.001;
  __constant f64cl_t PERMILLION  = 0.000001;
#ifdef __cplusplus
}
#endif

#endif // End of GUARD_GGEMS_TOOLS_SYSTEMOFUNITS_HH
