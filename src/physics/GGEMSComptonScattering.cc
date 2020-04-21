/*!
  \file GGEMSComptonScattering.cc

  \brief Compton Scattering process from standard model for Geant4 (G4...)

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 31, 2020
*/

#include "GGEMS/materials/GGEMSMaterials.hh"
#include "GGEMS/physics/GGEMSComptonScattering.hh"
#include "GGEMS/physics/GGEMSParticleCrossSectionsStack.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSComptonScattering::GGEMSComptonScattering(std::string const& primary_particle, bool const& is_secondary)
: GGEMSEMProcess()
{
  GGcout("GGEMSComptonScattering", "GGEMSComptonScattering", 3) << "Allocation of GGEMSComptonScattering..." << GGendl;

  process_name_ = "Compton";

  // Check type of primary particle
  if (primary_particle != "gamma") {
    std::ostringstream oss(std::ostringstream::out);
    oss << "For Compton scattering, incident particle has to be a 'gamma'";
    GGEMSMisc::ThrowException("GGEMSComptonScattering", "GGEMSComptonScattering", oss.str());
  }

  process_id_ = GGEMSProcess::COMPTON_SCATTERING;
  primary_particle_ = "gamma";
  secondary_particle_ = "e-";
  is_secondaries_ = is_secondary;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSComptonScattering::~GGEMSComptonScattering(void)
{
  GGcout("GGEMSComptonScattering", "~GGEMSComptonScattering", 3) << "Deallocation of GGEMSComptonScattering..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSComptonScattering::BuildCrossSectionTables(std::shared_ptr<cl::Buffer> particle_cross_sections, std::shared_ptr<cl::Buffer> material_tables)
{
  GGcout("GGEMSComptonScattering", "BuildCrossSectionTables", 3) << "Building cross section table for Compton scattering..." << GGendl;

  // Call mother
  GGEMSEMProcess::BuildCrossSectionTables(particle_cross_sections, material_tables);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSComptonScattering::ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number) const
{
  GGfloat cross_section_by_atom = 0.0f;

  if (atomic_number < 1 || energy < 1e-4f) {return cross_section_by_atom;}

  GGfloat p1Z = atomic_number * ( 2.7965e-23f +  1.9756e-27f*atomic_number + -3.9178e-29f*atomic_number*atomic_number);
  GGfloat p2Z = atomic_number * (-1.8300e-23f + -1.0205e-24f*atomic_number +  6.8241e-27f*atomic_number*atomic_number);
  GGfloat p3Z = atomic_number * ( 6.7527e-22f + -7.3913e-24f*atomic_number +  6.0480e-27f*atomic_number*atomic_number);
  GGfloat p4Z = atomic_number * (-1.9798e-21f +  2.7079e-24f*atomic_number +  3.0274e-26f*atomic_number*atomic_number);
  GGfloat T0 = (atomic_number < 1.5f)? 40.0e-3f : 15.0e-3f;
  GGfloat d1, d2, d3, d4, d5;

  d1 = fmaxf(energy, T0) / GGEMSPhysicalConstant::ELECTRON_MASS_C2; 
  cross_section_by_atom = p1Z*logf(1.0f+2.0f*d1)/d1+(p2Z+p3Z*d1+p4Z*d1*d1)/(1.0f+20.0f*d1+230.0f*d1*d1+440.0f*d1*d1*d1);

  if (energy < T0) {
    d1 = (T0+1.0e-3f) / GGEMSPhysicalConstant::ELECTRON_MASS_C2;
    d2 = p1Z*logf(1.0f+2.0f*d1)/d1+(p2Z+p3Z*d1+p4Z*d1*d1)/(1.0f+20.0f*d1+230.0f*d1*d1+440.0f*d1*d1*d1);
    d3 = (-T0 * (d2 - cross_section_by_atom)) / (cross_section_by_atom*1.0e-3f);
    d4 = (atomic_number > 1.5f)? 0.375f-0.0556f*logf(atomic_number) : 0.15f;
    d5 = logf(energy / T0);
    cross_section_by_atom *= expf(-d5 * (d3 + d4*d5));
  }
  
  return cross_section_by_atom; // in mm2
}
