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

#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSProfiler.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSProfiler::GGEMSProfiler(void)
: profiler_items_(nullptr),
  number_of_data_(0)
{
  GGcout("GGEMSProfiler", "GGEMSProfiler", 3) << "Allocation of GGEMS Profiler..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSProfiler::~GGEMSProfiler(void)
{
  GGcout("GGEMSProfiler", "~GGEMSProfiler", 3) << "Deallocation of GGEMS Profiler..." << GGendl;

  if (profiler_items_) {
    for (std::size_t i = 0; i < number_of_data_; ++i) {
      delete profiler_items_[i];
      profiler_items_[i] = nullptr;
    }
    delete[] profiler_items_;
    profiler_items_ = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProfiler::CallBackFunction(cl_event event, GGint event_command_exec_status, void* user_data)
{
  if (event_command_exec_status == CL_COMPLETE) {
    GGEMSProfiler* p = reinterpret_cast<GGEMSProfiler*>(user_data);
    p->AddProfilerItem(event);
    clReleaseEvent(event);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProfiler::AddProfilerItem(cl_event event)
{
  if (number_of_data_ == 0) { // First profiler item
    profiler_items_ = new GGEMSProfilerItem*[1];
    profiler_items_[0] = new GGEMSProfilerItem(event);
  }
  else {
    GGEMSProfilerItem** tmp = new GGEMSProfilerItem*[number_of_data_+1];
    for (std::size_t i = 0; i < number_of_data_; ++i) {
      tmp[i] = profiler_items_[i];
    }

    tmp[number_of_data_] = new GGEMSProfilerItem(event);

    delete[] profiler_items_;
    profiler_items_ = tmp;
  }

  // increment profiler item
  number_of_data_++;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSProfiler::HandleEvent(cl::Event event)
{
  clRetainEvent(event());
  event.setCallback(CL_COMPLETE, reinterpret_cast<void (CL_CALLBACK*)(cl_event, GGint, void*)>(GGEMSProfiler::CallBackFunction), this);
}
