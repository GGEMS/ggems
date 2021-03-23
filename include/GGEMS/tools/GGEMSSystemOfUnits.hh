#ifndef GUARD_GGEMS_TOOLS_GGEMSSYSTEMOFUNITS_HH
#define GUARD_GGEMS_TOOLS_GGEMSSYSTEMOFUNITS_HH

// ************************************************************************
// * This file is part of GGEMS.                                          *
// *                                                                      *
// * GGEMS is free software: you can redistribute it and/or modify        *
// * it under the terms of the GNU General Public License as published by *
// * the Free Software Foundation, either version 3 of the License, or    *
// * (at your option) any later version.                                  *
// *                                                                      *
// * GGEMS is distributed in the hope that it will be useful,             *
// * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
// * GNU General Public License for more details.                         *
// *                                                                      *
// * You should have received a copy of the GNU General Public License    *
// * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
// *                                                                      *
// ************************************************************************

/*!
  \file GGEMSSystemOfUnits.hh

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

#ifndef __OPENCL_C_VERSION__
/*!
  \def __constant
  \brief __constant is known for OpenCL, but for C++ we define __constant as a constexpr
*/
#define __constant inline static constexpr
#include <algorithm>
#include <sstream>
#include <cfloat>
#include <iostream>
#include "GGEMS/tools/GGEMSTools.hh"
#endif

// Lengths [L] (mm)
__constant GGfloat nm  = 1.e-6f; /*!< Nanometer */
__constant GGfloat um  = 1.e-3f; /*!< Micrometer */
__constant GGfloat mm  = 1.0f; /*!< Millimeter (REFERENCE) */
__constant GGfloat mm2 = 1.0f; /*!< Squared millimeter (REFERENCE) */
__constant GGfloat mm3 = 1.0f; /*!< Cubic millimeter (REFERENCE) */
__constant GGfloat cm  = 10.f; /*!< Centimeter */
__constant GGfloat cm2 = 1.e2f; /*!< Squared centimeter */
__constant GGfloat cm3 = 1.e3f; /*!< Cubic centimeter */
__constant GGfloat m   = 1.e3f; /*!< Meter */
__constant GGfloat m2  = 1.e6f; /*!< Squared meter */
__constant GGfloat m3  = 1.e9f; /*!< Cubic meter */
__constant GGfloat km  = 1.e6f; /*!< Kilometer */
__constant GGfloat km2 = 1.e12f; /*!< Squared kilometer */
__constant GGfloat km3 = 1.e18f; /*!< Cubic kilometer */
__constant GGfloat pc  = 3.0856775807e+19f; /*!< Parsec */

// Cross section unit (mm2)
__constant GGfloat b = 1.e-22f; /*!< Barn */
__constant GGfloat mb = 1.e-25f; /*!< millibarn */
__constant GGfloat ub = 1.e-28f; /*!< microbarn */
__constant GGfloat nb = 1.e-31f; /*!< nanobarn */
__constant GGfloat pb = 1.e-34f; /*!< picobarn */

// Angles
__constant GGfloat rad  = 1.0f; /*!< Radian (REFERENCE) */
__constant GGfloat mrad = 1.e-3f; /*!< milliradian */
__constant GGfloat deg  = 3.141592653589793238463f/180.0f; /*!< Degree */

// Solid angle
__constant GGfloat sr   = 1.0f; /*!< Steradian (REFERENCE) */

// Time [T] (ns)
__constant GGfloat ns = 1.f; /*!< Nanosecond (REFERENCE) */
__constant GGfloat s  = 1.e+9f; /*!< Second */
__constant GGfloat ms = 1.e+6f; /*!< Millisecond */
__constant GGfloat us = 1.e+3f; /*!< Microsecond */
__constant GGfloat ps = 1.e-3f; /*!< Picosecond */

// Frequency [T^-1] (ns-1)
__constant GGfloat Hz  = 1.f/(1.e+9f); /*!< Hertz */
__constant GGfloat kHz = 1.e-6f; /*!< Kilohertz */
__constant GGfloat MHz = 1.e-3f; /*!< Megahertz */

// Electric charge [Q] (eplus)
__constant GGfloat eplus = 1.f; /*!< Positron charge */
__constant GGfloat qe    = 1.602176487e-19f; /*!< elementary charge in coulomb (C) */
__constant GGfloat C     = 1.f/1.602176487e-19f; /*!< Coulomb = 6.24150e+18 * eplus */

// Energy [E] (MeV)
__constant GGfloat eV    = 1.e-6f; /*!< Electronvolt */
__constant GGfloat keV   = 1.e-3f; /*!< kiloelectronvolt */
__constant GGfloat MeV   = 1.f; /*!< Megaelectronvolt (REFERENCE) */
__constant GGfloat GeV   = 1.e+3f; /*!< Gigaelectronvolt */
__constant GGfloat TeV   = 1.e+6f; /*!< Teraelectronvolt */
__constant GGfloat PeV   = 1.e+9f; /*!< Petaelectronvolt */
__constant GGfloat J     = 1.e-6f/1.602176487e-19f; /*!< Joule 6.24150 e+12 * MeV */

// Mass [E][T^2][L^-2] (MeV.ns2.mm-2)
__constant GGfloat kg = 6.241509704e+24f; /*!< Kilogram */
__constant GGfloat g  = 6.241509704e+21f; /*!< gram */
__constant GGfloat mg = 6.241509704e+18f; /*!< milligram */

// Power [E][T^-1] (MeV.ns-1)
__constant GGfloat W = 6.241509766e+3f; /*!< Watt */

// Force [E][L^-1] (MeV.mm-1)
__constant GGfloat N = 6.241509766e+9f; /*!< Newton */

// Pressure [E][L^-3] (MeV.mm-3)
__constant GGfloat Pa = 6.241509766e+3f; /*!< Pascal */
__constant GGfloat bar = 100000.0f*6.241509766e+3f; /*!< Bar */
__constant GGfloat atm = 101325.0f*6.241509766e+3f; /*!< Atmosphere */

// Electric current [Q][T^-1] (C.ns-1)
__constant GGfloat A  = 6.241509696e+9f; /*!< Ampere */
__constant GGfloat mA = 6.241509696e+6f; /*!< Milliampere */
__constant GGfloat uA = 6.241509696e+3f; /*!< Microampere */
__constant GGfloat nA = 6.241509696f; /*!< Nanoampere */

// Electric potential [E][Q^-1] 
__constant GGfloat MV = 1.0f; /*!< Megavolt (REFERENCE) */
__constant GGfloat kV = 1.e-3f; /*!< Kilovolt */
__constant GGfloat V  = 1.e-6f; /*!< Volt */

// Electric resistance [E][T][Q^-2] (MeV.ns.C-2)
__constant GGfloat OHM = 1.602176452e-16f; /*!< OHM 1.60217e-16*(MeV/eplus)/(eplus/ns) */

// Electric capacitance [Q^2][E^-1] (C.MV-1)
__constant GGfloat F  = 6.241509468e+24f; /*!< Farad */
__constant GGfloat mF = 6.241509468e+21f; /*!< millifarad */
__constant GGfloat uF = 6.241509468e+18f; /*!< microfarad */
__constant GGfloat nF = 6.241509468e+15f; /*!< nanofarad */
__constant GGfloat pF = 6.241509468e+12f; /*!< picofarad */

// Magnetic Flux [T][E][Q^-1] (ns.MV)
__constant GGfloat Wb = 1000.0f; /*!< Weber 1000*megavolt*ns */

// Magnetic Field [T][E][Q^-1][L^-2] (MV.ns.mm2)
__constant GGfloat T = 0.001f; /*!< Tesla 0.001*megavolt*ns/mm2 */
__constant GGfloat G = 1.e-7f; /*!< Gauss */
__constant GGfloat kG = 1.e-4f; /*!< Kilogauss */

// Inductance [T^2][E][Q^-2] (MeV.ns2.C-2)
__constant GGfloat H = 1.602176383e-07f; /*!< Henry 1.60217e-7*MeV*(ns/eplus)^2 */

// Temperature (K)
__constant GGfloat K = 1.0f; /*!< Kelvin (REFERENCE) */

// Amount of substance (mol)
__constant GGfloat mol = 1.0f; /*!< Mole (REFERENCE) */

// Activity [T^-1] (ns-1)
__constant GGfloat Bq  = 1.e-9f; /*!< Becquerel */
__constant GGfloat kBq = 1.e-6f; /*!< Kilobecquerel */
__constant GGfloat MBq = 1.e-3f; /*!< Megabecquerel */
__constant GGfloat GBq = 1.0f; /*!< Gigabecquerel (REFERENCE) */
__constant GGfloat Ci  = 3.7e+10f/1.e+9f; /*!< Curie (Bq.ns-1) */
__constant GGfloat mCi = 3.7e-2f; /*!< Millicurie */
__constant GGfloat uCi = 3.7e-5f; /*!< Microcurie */

// Absorbed dose [L^2][T^-2] (mm2.ns-2)
__constant GGfloat Gy  = 1.0e-12f; /*!< Gray */
__constant GGfloat kGy = 1.0e-9f; /*!< Kilogray */
__constant GGfloat mGy = 1.0e-15f; /*!< Milligray */
__constant GGfloat uGy = 1.0e-18f; /*!< Microgray */

// Luminous intensity [I] (cd)
__constant GGfloat cd = 1.0f; /*!< Candela (REFERENCE) */

// Luminous flux [I] (cd.sr)
__constant GGfloat lm = 1.0f; /*!< Lumen (REFERENCE) */

// Illuminance [I][L^-2] (cd.sr.mm-2)
__constant GGfloat lx = 1.e-6f; /*!< Lux */

// Miscellaneous
__constant GGfloat percent = 0.01f; /*!< Percent value */
__constant GGfloat perthousant = 0.001f; /*!< Perthousand value */
__constant GGfloat permillion  = 0.000001f; /*!< Permillion value */

#ifndef __OPENCL_C_VERSION__

/*!
  \namespace
  \brief empty namespace storing prefix unit
*/
namespace
{
  std::string prefix_unit[17] = {
    // yocto, zepto, atto, femto, pico, nano, micro, milli,
          "y",   "z",  "a",   "f",  "p",  "n",   "u",   "m",
    // N/A,
         "",
    // kilo, mega, giga, tera, peta, exa, zetta, yotta
         "k",  "M",  "G",  "T",  "P", "E",   "Z",   "Y"
  }; /*!< prefix units */
}

/*!
  \fn inline std::string BestDistanceUnit(T const& value)
  \tparam T - type of the value
  \param value - value to convert to best unit
  \brief Choose best distance unit
  \return string value and distance unit
*/
template <typename T>
inline std::string BestDistanceUnit(T const& value)
{
  std::string base_unit = "m"; // Base unit is meter (m)

  // Find index of prefix, milli is the reference
  GGint index = static_cast<GGint>(std::log(std::fabs(static_cast<GGdouble>(value)))/std::log(1000.0)) + 7;
  index = (index < 0) ? 0 : index;
  index = (index > 16) ? 16 : index;

  if (static_cast<GGdouble>(value) == 0.0) index = 7;

  std::ostringstream oss(std::stringstream::out);
  oss << static_cast<GGdouble>(value)/std::pow(1000, index - 7) << " " << ::prefix_unit[index]+base_unit;

  return oss.str();
}

/*!
  \fn inline T DistanceUnit(T const& value, std::string const& unit)
  \tparam T - type of the value to convert unit
  \param value - value to check
  \param unit - distance unit
  \brief Choose best distance unit
  \return value in the good unit
*/
template <typename T>
inline T DistanceUnit(T const& value, std::string const& unit)
{
  T new_value = static_cast<T>(0);
  if (unit == "nm") {
    new_value = static_cast<T>(value * nm);
  }
  else if (unit == "um") {
    new_value = static_cast<T>(value * um);
  }
  else if (unit == "mm") {
    new_value = static_cast<T>(value * mm);
  }
  else if (unit == "cm") {
    new_value = static_cast<T>(value * cm);
  }
  else if (unit == "m") {
    new_value = static_cast<T>(value * m);
  }
  else if (unit == "km") {
    new_value = static_cast<T>(value * km);
  }
  else if (unit == "pc") {
    new_value = static_cast<T>(value * pc);
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Unknown unit!!! You have choice between:" << std::endl;
    oss << "    - \"nm\": nanometer" << std::endl;
    oss << "    - \"um\": micrometer" << std::endl;
    oss << "    - \"mm\": millimeter" << std::endl;
    oss << "    - \"cm\": centimeter" << std::endl;
    oss << "    - \"m\": meter" << std::endl;
    oss << "    - \"km\": kilometer" << std::endl;
    oss << "    - \"pc\": parsec";
    GGEMSMisc::ThrowException("", "DistanceUnit", oss.str());
  }
  return new_value;
}

/*!
  \fn inline std::string BestEnergyUnit(T const& value)
  \tparam T - type of the value
  \param value - value to convert to best unit
  \brief Choose best energy unit, mega is the reference
  \return string value and energy unit
*/
template <typename T>
inline std::string BestEnergyUnit(T const& value)
{
  std::string base_unit = "eV"; // Base unit is electron volt (eV)

  // Find index of prefix
  GGint index = static_cast<GGint>(std::log(std::fabs(static_cast<GGdouble>(value)))/std::log(1000.0)) + 10;
  index = (index < 0) ? 0 : index;
  index = (index > 16) ? 16 : index;

  if (static_cast<GGdouble>(value) == 0.0) index = 8;

  std::ostringstream oss(std::stringstream::out);
  oss << static_cast<GGdouble>(value)/std::pow(1000, index - 10) << " " << ::prefix_unit[index]+base_unit;

  return oss.str();
}

/*!
  \fn inline T EnergyUnit(T const& value, std::string const& unit)
  \tparam T - type of the value to convert unit
  \param value - value to check
  \param unit - energy unit
  \brief Choose best energy unit
  \return value in the good unit
*/
template <typename T>
inline T EnergyUnit(T const& value, std::string const& unit)
{
  T new_value = static_cast<T>(0);
  if (unit == "eV") {
    new_value = static_cast<T>(value * eV);
  }
  else if (unit == "keV") {
    new_value = static_cast<T>(value * keV);
  }
  else if (unit == "MeV") {
    new_value = static_cast<T>(value * MeV);
  }
  else if (unit == "GeV") {
    new_value = static_cast<T>(value * GeV);
  }
  else if (unit == "TeV") {
    new_value = static_cast<T>(value * TeV);
  }
  else if (unit == "PeV") {
    new_value = static_cast<T>(value * PeV);
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Unknown unit!!! You have choice between:" << std::endl;
    oss << "    - \"eV\": electronvolt" << std::endl;
    oss << "    - \"keV\": kiloelectronvolt" << std::endl;
    oss << "    - \"MeV\": megaelectronvolt" << std::endl;
    oss << "    - \"GeV\": gigaelectronvolt" << std::endl;
    oss << "    - \"TeV\": teraelectronvolt" << std::endl;
    oss << "    - \"PeV\": petaelectronvolt" << std::endl;
    GGEMSMisc::ThrowException("", "EnergyUnit", oss.str());
  }
  return new_value;
}

/*!
  \fn inline T AngleUnit(T const& value, std::string const& unit)
  \tparam T - type of the value to convert unit
  \param value - value to check
  \param unit - angle unit
  \brief Choose best angle unit
  \return value in the good unit
*/
template <typename T>
inline T AngleUnit(T const& value, std::string const& unit)
{
  T new_value = static_cast<T>(0);
  if (unit == "rad") {
    new_value = static_cast<T>(value * rad);
  }
  else if (unit == "mrad") {
    new_value = static_cast<T>(value * mrad);
  }
  else if (unit == "deg") {
    new_value = static_cast<T>(value * deg);
  }
  else if (unit == "sr") {
    new_value = static_cast<T>(value * sr);
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Unknown unit!!! You have choice between:" << std::endl;
    oss << "    - \"rad\": radian" << std::endl;
    oss << "    - \"mrad\": milliradian" << std::endl;
    oss << "    - \"deg\": degree" << std::endl;
    oss << "    - \"sr\": steradian" << std::endl;
    GGEMSMisc::ThrowException("", "AngleUnit", oss.str());
  }
  return new_value;
}

/*!
  \fn inline T DensityUnit(T const& value, std::string const& unit)
  \tparam T - type of the value to convert unit
  \param value - value to check
  \param unit - density unit
  \brief Choose best density unit
  \return value in the good unit
*/
template <typename T>
inline T DensityUnit(T const& value, std::string const& unit)
{
  T new_value = static_cast<T>(0);
  if (unit == "g/cm3") {
    new_value = static_cast<T>(value * g/cm3);
  }
  else if (unit == "kg/m3") {
    new_value = static_cast<T>(value * kg/m3);
  }
  else if (unit == "mg/cm3") {
    new_value = static_cast<T>(value * mg/cm3);
  }
  else {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Unknown unit!!! You have choice between:" << std::endl;
    oss << "    - \"g/cm3\": gram per cubic centimeter" << std::endl;
    oss << "    - \"kg/m3\": kilogram per cubic meter" << std::endl;
    oss << "    - \"mg/cm3\": milligram per cubic centimeter" << std::endl;
    GGEMSMisc::ThrowException("", "DensityUnit", oss.str());
  }
  return new_value;
}

/*!
  \fn inline std::string BestDigitalUnit(GGulong const& value)
  \param value - value to convert to best unit
  \brief Choose best digital unit
  \return string value and energy unit
*/
inline std::string BestDigitalUnit(GGulong const& value)
{
  std::string base_unit = "B"; // Base unit is byte(B)

  // Find index of prefix
  GGint index = static_cast<GGint>(std::log(static_cast<GGdouble>(value))/std::log(1000.0)) + 8;
  index = (index < 8) ? 8 : index;
  index = (index > 16) ? 16 : index;

  if (value == 0) index = 8;

  std::ostringstream oss(std::stringstream::out);
  oss << static_cast<GGdouble>(value)/std::pow(1000, index - 8) << " " << ::prefix_unit[index]+base_unit;

  return oss.str();
}
#endif

#endif // End of GUARD_GGEMS_TOOLS_GGEMSSYSTEMOFUNITS_HH
