/*!
  \file GGEMSRangeCuts.cc

  \brief GGEMS class storing and converting the cut in energy cut

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday March 18, 2020
*/

#include "GGEMS/physics/GGEMSRangeCuts.hh"

#include "GGEMS/physics/GGEMSRangeCutsManager.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/global/GGEMSConstants.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRangeCuts::GGEMSRangeCuts(void)
: length_cut_photon_(GGEMSDefaultParams::PHOTON_CUT),
  length_cut_electron_(GGEMSDefaultParams::ELECTRON_CUT)
{
  GGcout("GGEMSRangeCuts", "GGEMSRangeCuts", 3) << "Allocation of GGEMSRangeCuts..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSRangeCuts::~GGEMSRangeCuts(void)
{
  GGcout("GGEMSRangeCuts", "~GGEMSRangeCuts", 3) << "Deallocation of GGEMSRangeCuts..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCuts::SetPhotonLengthCut(GGfloat const& cut)
{
  length_cut_photon_ = cut;
  if (length_cut_photon_ < GGEMSDefaultParams::PHOTON_CUT) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Cut length for photon " << cut << " mm is too small!!! Minimum value is " << GGEMSDefaultParams::PHOTON_CUT << " mm!!!";
    GGEMSMisc::ThrowException("GGEMSRangeCuts", "SetPhotonLengthCut", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSRangeCuts::SetElectronLengthCut(GGfloat const& cut)
{
  length_cut_electron_ = cut;
  if (length_cut_electron_ < GGEMSDefaultParams::ELECTRON_CUT) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "Cut length for electron " << cut << " mm is too small!!! Minimum value is " << GGEMSDefaultParams::ELECTRON_CUT << " mm!!!";
    GGEMSMisc::ThrowException("GGEMSRangeCuts", "SetElectronLengthCut", oss.str());
  }
}
