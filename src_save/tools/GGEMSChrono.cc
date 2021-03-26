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
  \file GGEMSChrono.cc

  \brief Structure storing static method computing/displaying the time

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Friday October 4, 2019
*/

#include "GGEMS/tools/GGEMSChrono.hh"
#include "GGEMS/tools/GGEMSPrint.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSChrono::DisplayTime(DurationNano const& duration, std::string const& displayed_text)
{
  #if __MINGW64__ || __clang__ || (_MSC_VER > 1800) || __GNUC__
  // Display the iteration time
  GGcout("GGEMSChrono", "DisplayTime", 0) << "Elapsed time (" << displayed_text << "): " << std::setfill( '0' ) << std::setw(2) << std::chrono::duration_cast<Hs>((duration)).count() << " hours " << std::setw(2) << std::chrono::duration_cast<Mins>((duration) % Hs(1)).count() << " mins " << std::setw(2) << std::chrono::duration_cast<Secs>((duration) % Mins(1) ).count() << " secs " << std::setw(3) << std::chrono::duration_cast<Ms>((duration) % Secs(1)).count() << " ms" << GGendl;
  #else
  GGcout("Chrono", "DisplayTime", 0) << "Elapsed time (" << displayedText << "): " << duration.count() / 1000000000.0f << "sec";
  #endif
}
