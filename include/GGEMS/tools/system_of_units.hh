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

#include <cmath>

/*!
  \namespace Units
  \brief namespace storing all the usefull physical units
*/
namespace Units
{
  // Pi definitions
  inline static constexpr double kPi    = 3.141592653589793238463;
  inline static constexpr double k2_Pi  = 2.0 * kPi;
  inline static constexpr double kPi_2  = kPi / 2.0;
  inline static constexpr double kPi_Pi = kPi * kPi;

  // Lengths
  inline static constexpr double kMillimeter  = 1.0;
  inline static constexpr double kMillimeter2 = kMillimeter*kMillimeter;
  inline static constexpr double kMillimeter3 =
    kMillimeter*kMillimeter*kMillimeter;

  inline static constexpr double kCentimeter  = 10.*kMillimeter;
  inline static constexpr double kCentimeter2 = kCentimeter*kCentimeter;
  inline static constexpr double kCentimeter3 =
    kCentimeter*kCentimeter*kCentimeter;

  inline static constexpr double kMeter  = 1000.*kMillimeter;
  inline static constexpr double kMeter2 = kMeter*kMeter;
  inline static constexpr double kMeter3 = kMeter*kMeter*kMeter;

  inline static constexpr double kKilometer  = 1000.*kMeter;
  inline static constexpr double kKilometer2 = kKilometer*kKilometer;
  inline static constexpr double kKilometer3 = kKilometer*kKilometer*kKilometer;

  inline static constexpr double kParsec = 96939420213600000*kMeter/kPi;

  inline static constexpr double kMicrometer = 1.e-6 *kMeter;
  inline static constexpr double kNanometer  = 1.e-9 *kMeter;
  inline static constexpr double kAngstrom   = 1.e-10*kMeter;
  inline static constexpr double kFermi      = 1.e-15*kMeter;

  inline static constexpr double kBarn      = 1.e-28*kMeter;
  inline static constexpr double kMillibarn = 1.e-3 *kBarn;
  inline static constexpr double kMicrobarn = 1.e-6 *kBarn;
  inline static constexpr double kNanobarn  = 1.e-9 *kBarn;
  inline static constexpr double kPicobarn  = 1.e-12*kBarn;

  // Symbol definitions
  inline static constexpr double nm  = kNanometer;
  inline static constexpr double um  = kMicrometer;

  inline static constexpr double mm  = kMillimeter;
  inline static constexpr double mm2 = kMillimeter2;
  inline static constexpr double mm3 = kMillimeter3;

  inline static constexpr double cm  = kCentimeter;
  inline static constexpr double cm2 = kCentimeter2;
  inline static constexpr double cm3 = kCentimeter3;

  inline static constexpr double m  = kMeter;
  inline static constexpr double m2 = kMeter2;
  inline static constexpr double m3 = kMeter3;

  inline static constexpr double km  = kKilometer;
  inline static constexpr double km2 = kKilometer2;
  inline static constexpr double km3 = kKilometer3;

  inline static constexpr double pc = kParsec;

  // Angles
  inline static constexpr double kRadian      = 1.0;
  inline static constexpr double kMilliradian = 1.e-3*kRadian;
  inline static constexpr double kDegree      = (kPi/180.0)*kRadian;

  inline static constexpr double kSteradian = 1.0;

  // Symbols definitions
  inline static constexpr double rad  = kRadian;
  inline static constexpr double mrad = kMilliradian;
  inline static constexpr double sr   = kSteradian;
  inline static constexpr double deg  = kDegree;

  // Time
  inline static constexpr double kNanosecond  = 1.;
  inline static constexpr double kSecond      = 1.e+9 *kNanosecond;
  inline static constexpr double kMillisecond = 1.e-3 *kSecond;
  inline static constexpr double kMicrosecond = 1.e-6 *kSecond;
  inline static constexpr double kPicosecond  = 1.e-12*kSecond;

  inline static constexpr double kHertz     = 1./kSecond;
  inline static constexpr double kKilohertz = 1.e+3*kHertz;
  inline static constexpr double kMegahertz = 1.e+6*kHertz;

  // Symbols definitions
  inline static constexpr double ns = kNanosecond;
  inline static constexpr double s  = kSecond;
  inline static constexpr double ms = kMillisecond;

  // Electric charge [Q]
  inline static constexpr double kEplus   = 1. ;// positron charge
  inline static constexpr double kESI     = 1.602176487e-19;// positron charge in coulomb
  inline static constexpr double kCoulomb = kEplus/kESI;// coulomb = 6.24150 e+18 * eplus

  // Energy [E]
  inline static constexpr double kMegaelectronvolt = 1. ;
  inline static constexpr double kElectronvolt     = 1.e-6*kMegaelectronvolt;
  inline static constexpr double kKiloelectronvolt = 1.e-3*kMegaelectronvolt;
  inline static constexpr double kGigaelectronvolt = 1.e+3*kMegaelectronvolt;
  inline static constexpr double kTeraelectronvolt = 1.e+6*kMegaelectronvolt;
  inline static constexpr double kPetaelectronvolt = 1.e+9*kMegaelectronvolt;

  inline static constexpr double kJoule = kElectronvolt/kESI;// joule = 6.24150 e+12 * MeV

  // Symbols definitions
  inline static constexpr double MeV = kMegaelectronvolt;
  inline static constexpr double eV  = kElectronvolt;
  inline static constexpr double keV = kKiloelectronvolt;
  inline static constexpr double GeV = kGigaelectronvolt;
  inline static constexpr double TeV = kTeraelectronvolt;
  inline static constexpr double PeV = kPetaelectronvolt;

  // Mass [E][T^2][L^-2]
  inline static constexpr double kKilogram  = kJoule*kSecond*kSecond/(kMeter*kMeter);
  inline static constexpr double kGram      = 1.e-3*kKilogram;
  inline static constexpr double kMilligram = 1.e-3*kGram;

  // Symbols definitions
  inline static constexpr double  kg = kKilogram;
  inline static constexpr double  g  = kGram;
  inline static constexpr double  mg = kMilligram;

  // Power [E][T^-1]
  inline static constexpr double kWatt = kJoule/kSecond;// watt = 6.24150 e+3 * MeV/ns

  // Force [E][L^-1]
  inline static constexpr double kNewton = kJoule/kMeter;// newton = 6.24150 e+9 * MeV/mm

  // Pressure [E][L^-3]
  inline static constexpr double kPascal     = kNewton/m2;   // pascal = 6.24150 e+3 * MeV/mm3
  inline static constexpr double kBar        = 100000*kPascal; // bar    = 6.24150 e+8 * MeV/mm3
  inline static constexpr double kAtmosphere = 101325*kPascal; // atm    = 6.32420 e+8 * MeV/mm3

  // Electric current [Q][T^-1]
  inline static constexpr double kAmpere      = kCoulomb/kSecond; // ampere = 6.24150 e+9 * eplus/ns
  inline static constexpr double kMilliampere = 1.e-3*kAmpere;
  inline static constexpr double kMicroampere = 1.e-6*kAmpere;
  inline static constexpr double kNanoampere  = 1.e-9*kAmpere;

  // Electric potential [E][Q^-1]
  inline static constexpr double kMegavolt = kMegaelectronvolt/kEplus;
  inline static constexpr double kKilovolt = 1.e-3*kMegavolt;
  inline static constexpr double kVolt     = 1.e-6*kMegavolt;

  // Electric resistance [E][T][Q^-2]
  inline static constexpr double kOhm = kVolt/kAmpere;// ohm = 1.60217e-16*(MeV/eplus)/(eplus/ns)

  // Electric capacitance [Q^2][E^-1]
  inline static constexpr double kFarad = kCoulomb/kVolt;// farad = 6.24150e+24 * eplus/Megavolt
  inline static constexpr double kMillifarad = 1.e-3*kFarad;
  inline static constexpr double kMicrofarad = 1.e-6*kFarad;
  inline static constexpr double kNanofarad = 1.e-9*kFarad;
  inline static constexpr double kPicofarad = 1.e-12*kFarad;

  // Magnetic Flux [T][E][Q^-1]
  inline static constexpr double kWeber = kVolt*kSecond;// weber = 1000*megavolt*ns

  // Magnetic Field [T][E][Q^-1][L^-2]
  inline static constexpr double kTesla     = kVolt*kSecond/kMeter2;// tesla =0.001*megavolt*ns/mm2

  inline static constexpr double kGauss     = 1.e-4*kTesla;
  inline static constexpr double kKilogauss = 1.e-1*kTesla;

  // Inductance [T^2][E][Q^-2]
  inline static constexpr double kHenry = kWeber/kAmpere;// henry = 1.60217e-7*MeV*(ns/eplus)**2

  // Temperature
  inline static constexpr double kKelvin = 1.;

  // Amount of substance
  inline static constexpr double kMole = 1.;

  // Activity [T^-1]
  inline static constexpr double kBecquerel     = 1./kSecond ;
  inline static constexpr double kCurie         = 3.7e+10 * kBecquerel;
  inline static constexpr double kKilobecquerel = 1.e+3*kBecquerel;
  inline static constexpr double kMegabecquerel = 1.e+6*kBecquerel;
  inline static constexpr double kGigabecquerel = 1.e+9*kBecquerel;
  inline static constexpr double kMillicurie    = 1.e-3*kCurie;
  inline static constexpr double kMicrocurie    = 1.e-6*kCurie;

  // Symbols definitions
  inline static constexpr double Bq  = kBecquerel;
  inline static constexpr double kBq = kKilobecquerel;
  inline static constexpr double MBq = kMegabecquerel;
  inline static constexpr double GBq = kGigabecquerel;
  inline static constexpr double Ci  = kCurie;
  inline static constexpr double mCi = kMillicurie;
  inline static constexpr double uCi = kMicrocurie;

  // Absorbed dose [L^2][T^-2]
  inline static constexpr double kGray      = kJoule/kKilogram;
  inline static constexpr double kKilogray  = 1.e+3*kGray;
  inline static constexpr double kMilligray = 1.e-3*kGray;
  inline static constexpr double kMicrogray = 1.e-6*kGray;

  // Luminous intensity [I]
  inline static constexpr double kCandela = 1.;

  // Luminous flux [I]
  inline static constexpr double kLumen = kCandela*kSteradian;

  // Illuminance [I][L^-2]
  inline static constexpr double kLux = kLumen/kMeter2;

  // Miscellaneous
  inline static constexpr double kPerCent     = 0.01 ;
  inline static constexpr double kPerThousand = 0.001;
  inline static constexpr double kPerMillion  = 0.000001;
}

#endif // End of GUARD_GGEMS_TOOLS_SYSTEMOFUNITS_HH