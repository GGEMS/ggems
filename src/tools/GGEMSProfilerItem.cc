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
  GGcout("GGEMSProfilerItem", "GGEMSProfilerItem", 3) << "Allocation of GGEMSProfilerItem..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get time infos from event
  opencl_manager.CheckOpenCLError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(GGulong), &times_[0], nullptr), "GGEMSProfilerItem", "GGEMSProfilerItem");
  opencl_manager.CheckOpenCLError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(GGulong), &times_[1], nullptr), "GGEMSProfilerItem", "GGEMSProfilerItem");
  opencl_manager.CheckOpenCLError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(GGulong), &times_[2], nullptr), "GGEMSProfilerItem", "GGEMSProfilerItem");
  opencl_manager.CheckOpenCLError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(GGulong), &times_[3], nullptr), "GGEMSProfilerItem", "GGEMSProfilerItem");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSProfilerItem::~GGEMSProfilerItem(void)
{
  GGcout("GGEMSProfilerItem", "~GGEMSProfilerItem", 3) << "Deallocation of GGEMSProfilerItem..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProfilerItem::PrintInfos(void) const
{
  // Getting moment in time of enqueueing command
  DurationNano queue_time = static_cast<DurationNano>(times_[GGEMSEventInfo::QUEUE]);

  // Getting submission time from queue to execution
  DurationNano submit_time = static_cast<DurationNano>(times_[GGEMSEventInfo::SUBMIT] - times_[GGEMSEventInfo::QUEUE]);

  // Getting the start time of execution
  DurationNano start_time = static_cast<DurationNano>(times_[GGEMSEventInfo::START] - times_[GGEMSEventInfo::SUBMIT]);

  // Getting the command time of execution
  DurationNano end_time = static_cast<DurationNano>(times_[GGEMSEventInfo::END] - times_[GGEMSEventInfo::START]);

  GGcout("GGEMSProfilerItem", "PrintInfos", 0) << "Time of enqueueing command: " << queue_time.count() << " ns" << GGendl;
  GGcout("GGEMSProfilerItem", "PrintInfos", 0) << "Time of submit command from queue: " << submit_time.count() << " ns" << GGendl;
  GGcout("GGEMSProfilerItem", "PrintInfos", 0) << "Time of start command from submit: " << start_time.count() << " ns" << GGendl;
  GGcout("GGEMSProfilerItem", "PrintInfos", 0) << "Time of execution command finish: " << end_time.count() << " ns" << GGendl;
}
