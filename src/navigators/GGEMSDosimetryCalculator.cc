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
  \file GGEMSDosimetryCalculator.cc

  \brief Class providing tools storing and computing dose in phantom

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Wednesday January 13, 2021
*/

#include "GGEMS/navigators/GGEMSDosimetryCalculator.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSDosimetryCalculator::GGEMSDosimetryCalculator(void)
{
  GGcout("GGEMSDosimetryCalculator", "GGEMSDosimetryCalculator", 3) << "Allocation of GGEMSDosimetryCalculator..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSDosimetryCalculator::~GGEMSDosimetryCalculator(void)
{
  GGcout("GGEMSDosimetryCalculator", "~GGEMSDosimetryCalculator", 3) << "Deallocation of GGEMSDosimetryCalculator..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::CheckParameters(void) const
{
  ;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSDosimetryCalculator::Initialize(void)
{
  GGcout("GGEMSDosimetryCalculator", "Initialize", 3) << "Initializing dosimetry calculator..." << GGendl;
}
