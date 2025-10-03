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
  \file GGEMSProfiler.cc

  \brief GGEMS class handling a specific profiler type

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 16, 2021
*/

/// \cond
#include <mutex>
/// \endcond

#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSProfiler.hh"

/*!
  \brief empty namespace storing mutex
*/
namespace {
  std::mutex mutex; /*!< Mutex variable */
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSProfiler::GGEMSProfiler(void)
: profiler_item_(nullptr)
{}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSProfiler::~GGEMSProfiler(void)
{
  if (profiler_item_) {
    delete profiler_item_;
    profiler_item_ = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProfiler::Callback(cl_event event, GGint event_command_exec_status, void* user_data)
{
  if (event_command_exec_status == CL_COMPLETE) {
    GGEMSProfiler* p = reinterpret_cast<GGEMSProfiler*>(user_data);
    // Call back Function has to be thread safe!!!
    ::mutex.lock();
    p->AddProfilerItem(event);
    ::mutex.unlock();
    clReleaseEvent(event);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProfiler::AddProfilerItem(cl_event event)
{
  if (!profiler_item_) {
    profiler_item_ = new GGEMSProfilerItem(event);
  }
  else {
    profiler_item_->UpdateEvent(event);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProfiler::HandleEvent(cl::Event& event)
{
  clRetainEvent(event());
  event.setCallback(CL_COMPLETE, reinterpret_cast<void (CL_CALLBACK*)(cl_event, GGint, void*)>(GGEMSProfiler::Callback), this);
}
