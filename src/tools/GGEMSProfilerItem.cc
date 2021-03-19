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
  \file GGEMSProfilerItem.cc

  \brief GGEMS handling a specific item profiler

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 16, 2021
*/

#include "GGEMS/tools/GGEMSProfilerItem.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSProfilerItem::GGEMSProfilerItem(cl_event event)
{
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get time infos from event
  opencl_manager.CheckOpenCLError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(GGulong), &times_[GGEMSEventInfo::START], nullptr), "GGEMSProfilerItem", "GGEMSProfilerItem");
  opencl_manager.CheckOpenCLError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(GGulong), &times_[GGEMSEventInfo::END], nullptr), "GGEMSProfilerItem", "GGEMSProfilerItem");

  times_[GGEMSEventInfo::ELAPSED] = times_[GGEMSEventInfo::END] - times_[GGEMSEventInfo::START];
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProfilerItem::UpdateEvent(cl_event event)
{
  // Get time infos from event
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  opencl_manager.CheckOpenCLError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(GGulong), &times_[GGEMSEventInfo::START], nullptr), "GGEMSProfilerItem", "UpdateEvent");
  opencl_manager.CheckOpenCLError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(GGulong), &times_[GGEMSEventInfo::END], nullptr), "GGEMSProfilerItem", "UpdateEvent");
  times_[GGEMSEventInfo::ELAPSED] += times_[GGEMSEventInfo::END] - times_[GGEMSEventInfo::START];
}
