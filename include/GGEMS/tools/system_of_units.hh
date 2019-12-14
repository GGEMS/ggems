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
  __constant double PI       = 3.141592653589793238463;
  __constant double PI_TWICE = 2.0 * 3.141592653589793238463;
  __constant double PI_HALF  = 3.141592653589793238463 / 2.0;
  __constant double PI_PI    = 3.141592653589793238463 * 3.141592653589793238463;

  // Lengths
  __constant double MILLIMETER  = 1.0;
  __constant double MILLIMETER2 = 1.0*1.0;
  __constant double MILLIMETER3 = 1.0*1.0*1.0;

  __constant double CENTIMETER  = 10.*1.0;
  __constant double CENTIMETER2 = 10.*1.0*10.*1.0;
  __constant double CENTIMETER3 = 10.*1.0*10.*1.0*10.*1.0;

  __constant double METER  = 1000.*1.0;
  __constant double METER2 = 1000.*1.0*1000.*1.0;
  __constant double METER3 = 1000.*1.0*1000.*1.0*1000.*1.0;

  __constant double KILOMETER  = 1000.*1000.*1.0;
  __constant double KILOMETER2 = 1000.*1000.*1.0*1000.*1000.*1.0;
  __constant double KILOMETER3 = 1000.*1000.*1.0*1000.*1000.*1.0*1000.*1000.*1.0;

  __constant double PARSEC = 96939420213600000*1000.*1.0/3.141592653589793238463;

  __constant double MICROMETER = 1.e-6 *1000.*1.0;
  __constant double NANOMETER  = 1.e-9 *1000.*1.0;
  __constant double ANGSTROM   = 1.e-10*1000.*1.0;
  __constant double FERMI      = 1.e-15*1000.*1.0;

  __constant double BARN      = 1.e-28*1000.*1.0;
  __constant double MILLIBARN = 1.e-3 *1.e-28*1000.*1.0;
  __constant double MICROBARN = 1.e-6 *1.e-28*1000.*1.0;
  __constant double NANOBARN  = 1.e-9 *1.e-28*1000.*1.0;
  __constant double PICOBARN  = 1.e-12*1.e-28*1000.*1.0;

  // Symbol definitions
  __constant double nm  = 1.e-9 *1000.*1.0;
  __constant double um  = 1.e-6 *1000.*1.0;

  __constant double mm = 1.0;
  __constant double mm2 = 1.0*1.0;
  __constant double mm3 = 1.0*1.0*1.0;

  __constant double cm  = 10.*1.0;
  __constant double cm2 = 10.*1.0*10.*1.0;
  __constant double cm3 = 10.*1.0*10.*1.0*10.*1.0;

  __constant double m  = 1000.*1.0;
  __constant double m2 = 1000.*1.0*1000.*1.0;
  __constant double m3 = 1000.*1.0*1000.*1.0*1000.*1.0;

  __constant double Km  = 1000.*1000.*1.0;
  __constant double Km2 = 1000.*1000.*1.0*1000.*1000.*1.0;
  __constant double Km3 = 1000.*1000.*1.0*1000.*1000.*1.0*1000.*1000.*1.0;

  __constant double pc = 96939420213600000*1000.*1.0/3.141592653589793238463;

  // Angles
  __constant double RADIAN      = 1.0;
  __constant double MILLIRADIAN = 1.e-3*1.0;
  __constant double DEGREE      = (3.141592653589793238463/180.0)*1.0;

  __constant double STERADIAN = 1.0;

  // Symbols definitions
  __constant double rad  = 1.0;
  __constant double mrad = 1.e-3*1.0;
  __constant double sr   = 1.0;
  __constant double deg  = (3.141592653589793238463/180.0)*1.0;

  // Time
  __constant double NANOSECOND  = 1.;
  __constant double SECOND      = 1.e+9 *1.;
  __constant double MILLISECOND = 1.e-3 *1.e+9 *1.;
  __constant double MICROSECOND = 1.e-6 *1.e+9 *1.;
  __constant double PICOSECOND  = 1.e-12*1.e+9 *1.;

  __constant double HERTZ     = 1./(1.e+9 *1.);
  __constant double KILOHERTZ = 1.e+3*(1./(1.e+9 *1.));
  __constant double MEGAHERTZ = 1.e+6*(1./(1.e+9 *1.));

  // Symbols definitions
  __constant double ns = 1.;
  __constant double s  = 1.e+9 *1.;
  __constant double ms = 1.e-3 *1.e+9 *1.;

  // Electric charge [Q]
  __constant double EPLUS   = 1. ;// positron charge
  __constant double ESI     = 1.602176487e-19;// positron charge in coulomb
  __constant double COULOMB = 1./1.602176487e-19;// coulomb = 6.24150 e+18 * eplus

  // Energy [E]
  __constant double MEGAELECTRONVOLT = 1. ;
  __constant double ELECTRONVOLT     = 1.e-6*1.;
  __constant double KILOELECTRONVOLT = 1.e-3*1.;
  __constant double GIGAELECTRONVOLT = 1.e+3*1.;
  __constant double TERAELECTRONVOLT = 1.e+6*1.;
  __constant double PETAELECTRONVOLT = 1.e+9*1.;

  __constant double JOULE = 1.e-6*1./1.602176487e-19;// joule = 6.24150 e+12 * MeV

  // Symbols definitions
  __constant double MeV = 1.;
  __constant double eV  = 1.e-6*1.;
  __constant double KeV = 1.e-3*1.;
  __constant double GeV = 1.e+3*1.;
  __constant double TeV = 1.e+6*1.;
  __constant double PeV = 1.e+9*1.;

  // Mass [E][T^2][L^-2]
  __constant double KILOGRAM  = 1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0);
  __constant double GRAM      = 1.e-3*(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0));
  __constant double MILLIGRAM = 1.e-3*1.e-3*(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0));

  // Symbols definitions
  __constant double  Kg = 1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0);
  __constant double  g  = 1.e-3*(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0));
  __constant double  Mg = 1.e-3*1.e-3*(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0));

  // Power [E][T^-1]
  __constant double WATT = 1.e-6*1./1.602176487e-19/(1.e+9 *1.);// watt = 6.24150 e+3 * MeV/ns

  // Force [E][L^-1]
  __constant double NEWTON = 1.e-6*1./1.602176487e-19/(1000.*1.0);// newton = 6.24150 e+9 * MeV/mm

  // Pressure [E][L^-3]
  __constant double PASCAL     = (1.e-6*1./1.602176487e-19/(1000.*1.0))/(1000.*1.0*1000.*1.0);   // pascal = 6.24150 e+3 * MeV/mm3
  __constant double BAR        = 100000*((1.e-6*1./1.602176487e-19/(1000.*1.0))/(1000.*1.0*1000.*1.0)); // bar    = 6.24150 e+8 * MeV/mm3
  __constant double ATMOSPHERE = 101325*((1.e-6*1./1.602176487e-19/(1000.*1.0))/(1000.*1.0*1000.*1.0)); // atm    = 6.32420 e+8 * MeV/mm3

  // Electric current [Q][T^-1]
  __constant double AMPERE      = (1./1.602176487e-19)/(1.e+9 *1.); // ampere = 6.24150 e+9 * eplus/ns
  __constant double MILLIAMPERE = 1.e-3*((1./1.602176487e-19)/(1.e+9 *1.));
  __constant double MICROAMPERE = 1.e-6*((1./1.602176487e-19)/(1.e+9 *1.));
  __constant double NANOAMPERE  = 1.e-9*((1./1.602176487e-19)/(1.e+9 *1.));

  // Electric potential [E][Q^-1]
  __constant double MEGAVOLT = (1.)/(1.);
  __constant double KILOVOLT = 1.e-3*((1.)/(1.));
  __constant double VOLT     = 1.e-6*((1.)/(1.));

  // Electric resistance [E][T][Q^-2]
  __constant double OHM = (1.e-6*((1.)/(1.)))/((1./1.602176487e-19)/(1.e+9 *1.));// ohm = 1.60217e-16*(MeV/eplus)/(eplus/ns)

  // Electric capacitance [Q^2][E^-1]
  __constant double FARAD = (1./1.602176487e-19)/(1.e-6*((1.)/(1.)));// farad = 6.24150e+24 * eplus/Megavolt
  __constant double MILLIFARAD = 1.e-3*((1./1.602176487e-19)/(1.e-6*((1.)/(1.))));
  __constant double MICROFARAD = 1.e-6*((1./1.602176487e-19)/(1.e-6*((1.)/(1.))));
  __constant double NANOFARAD = 1.e-9*((1./1.602176487e-19)/(1.e-6*((1.)/(1.))));
  __constant double PICOFARAD = 1.e-12*((1./1.602176487e-19)/(1.e-6*((1.)/(1.))));

  // Magnetic Flux [T][E][Q^-1]
  __constant double WEBER = (1.e-6*((1.)/(1.)))*(1.e+9 *1.);// weber = 1000*megavolt*ns

  // Magnetic Field [T][E][Q^-1][L^-2]
  __constant double TESLA     = (1.e-6*((1.)/(1.)))*(1.e+9 *1.)/(1000.*1.0*1000.*1.0);// tesla =0.001*megavolt*ns/mm2

  __constant double GAUSS     = 1.e-4*((1.e-6*((1.)/(1.)))*(1.e+9 *1.)/(1000.*1.0*1000.*1.0));
  __constant double KILOGAUSS = 1.e-1*((1.e-6*((1.)/(1.)))*(1.e+9 *1.)/(1000.*1.0*1000.*1.0));

  // Inductance [T^2][E][Q^-2]
  __constant double HENRY = ((1.e-6*((1.)/(1.)))*(1.e+9 *1.))/((1./1.602176487e-19)/(1.e+9 *1.));// henry = 1.60217e-7*MeV*(ns/eplus)**2

  // Temperature
  __constant double KELVIN = 1.;

  // Amount of substance
  __constant double MOLE = 1.;

  // Activity [T^-1]
  __constant double BECQUEREL     = 1./(1.e+9 *1.) ;
  __constant double CURIE         = 3.7e+10 * (1./(1.e+9 *1.));
  __constant double KILOBECQUEREL = 1.e+3*(1./(1.e+9 *1.));
  __constant double MEGABECQUEREL = 1.e+6*(1./(1.e+9 *1.));
  __constant double GIGABECQUEREL = 1.e+9*(1./(1.e+9 *1.));
  __constant double MILLICURIE    = 1.e-3*(3.7e+10 * (1./(1.e+9 *1.)));
  __constant double MICROCURIE    = 1.e-6*(3.7e+10 * (1./(1.e+9 *1.)));

  // Symbols definitions
  __constant double Bq  = (1./(1.e+9 *1.));
  __constant double kBq = (1.e+3*(1./(1.e+9 *1.)));
  __constant double MBq = (1.e+6*(1./(1.e+9 *1.)));
  __constant double GBq = (1.e+9*(1./(1.e+9 *1.)));
  __constant double Ci  = (3.7e+10 * (1./(1.e+9 *1.)));
  __constant double mCi = (1.e-3*(3.7e+10 * (1./(1.e+9 *1.))));
  __constant double uCi = (1.e-6*(3.7e+10 * (1./(1.e+9 *1.))));

  // Absorbed dose [L^2][T^-2]
  __constant double GRAY      = (1.e-6*1./1.602176487e-19)/(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0));
  __constant double KILOGRAY  = 1.e+3*((1.e-6*1./1.602176487e-19)/(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0)));
  __constant double MILLIGRAY = 1.e-3*((1.e-6*1./1.602176487e-19)/(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0)));
  __constant double MICROGRAY = 1.e-6*((1.e-6*1./1.602176487e-19)/(1.e-6*1./1.602176487e-19*1.e+9 *1.*1.e+9 *1./(1000.*1.0*1000.*1.0)));

  // Luminous intensity [I]
  __constant double CANDELA = 1.;

  // Luminous flux [I]
  __constant double LUMEN = (1.)*(1.0);

  // Illuminance [I][L^-2]
  __constant double LUX = ((1.)*(1.0))/(1000.*1.0*1000.*1.0);

  // Miscellaneous
  __constant double PERCENT     = 0.01 ;
  __constant double PERTHOUSAND = 0.001;
  __constant double PERMILLION  = 0.000001;
#ifdef __cplusplus
}
#endif

#endif // End of GUARD_GGEMS_TOOLS_SYSTEMOFUNITS_HH
