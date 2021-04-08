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
  \file GGEMSProfilerManager.cc

  \brief GGEMS class managing profiler data

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 16, 2021
*/

#include "GGEMS/tools/GGEMSProfilerManager.hh"
#include "GGEMS/tools/GGEMSProfiler.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSChrono.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSProfilerManager::GGEMSProfilerManager(void)
{
  GGcout("GGEMSProfilerManager", "GGEMSProfilerManager", 3) << "GGEMSProfilerManager creating..." << GGendl;

  profilers_.clear();

  GGcout("GGEMSProfilerManager", "GGEMSProfilerManager", 3) << "GGEMSProfilerManager created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSProfilerManager::~GGEMSProfilerManager(void)
{
  GGcout("GGEMSProfilerManager", "~GGEMSProfilerManager", 3) << "GGEMSProfilerManager erasing!!!" << GGendl;

  profilers_.clear();

  GGcout("GGEMSProfilerManager", "~GGEMSProfilerManager", 3) << "GGEMSProfilerManager erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProfilerManager::HandleEvent(cl::Event event, std::string const& profile_name)
{
  // Checking if profile exists already, if not, creating one
  if (profilers_.find(profile_name) == profilers_.end()) {
    GGEMSProfiler profiler;
    profilers_.insert(std::make_pair(profile_name, profiler));
  }

  // Storing event data in correct profiler
  profilers_[profile_name].HandleEvent(event);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProfilerManager::PrintSummaryProfile(void) const
{
  for (auto&& p: profilers_) GGEMSChrono::DisplayTime(p.second.GetSummaryTime(), p.first);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProfilerManager::Reset(void)
{
  profilers_.clear();
}
