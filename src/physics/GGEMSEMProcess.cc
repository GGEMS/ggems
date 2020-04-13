/*!
  \file GGEMSEMProcess.cc

  \brief GGEMS mother class for electromagnectic process

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#include "GGEMS/physics/GGEMSEMProcess.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSEMProcess::GGEMSEMProcess()
:
  process_name_(""),
  primary_particle_(""),
  secondary_particle_(""),
  is_secondaries_(false),
  opencl_manager_(GGEMSOpenCLManager::GetInstance())
{
  GGcout("GGEMSEMProcess", "GGEMSEMProcess", 3) << "Allocation of GGEMSEMProcess..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSEMProcess::~GGEMSEMProcess(void)
{
  GGcout("GGEMSEMProcess", "~GGEMSEMProcess", 3) << "Deallocation of GGEMSEMProcess..." << GGendl;
}
