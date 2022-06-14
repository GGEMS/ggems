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
  \file GGEMSMeshedPhantom.cc

  \brief Child GGEMS class handling meshed phantom

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Tuesday June 14, 2022
*/

#include "GGEMS/navigators/GGEMSMeshedPhantom.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMeshedPhantom::GGEMSMeshedPhantom(std::string const& meshed_phantom_name)
: GGEMSNavigator(meshed_phantom_name),
  meshed_phantom_filename_("")
{
  GGcout("GGEMSMeshedPhantom", "GGEMSMeshedPhantom", 3) << "GGEMSMeshedPhantom creating..." << GGendl;

  GGcout("GGEMSMeshedPhantom", "GGEMSMeshedPhantom", 3) << "GGEMSMeshedPhantom created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMeshedPhantom::~GGEMSMeshedPhantom(void)
{
  GGcout("GGEMSMeshedPhantom", "~GGEMSMeshedPhantom", 3) << "GGEMSMeshedPhantom erasing..." << GGendl;

  GGcout("GGEMSMeshedPhantom", "~GGEMSMeshedPhantom", 3) << "GGEMSMeshedPhantom erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
