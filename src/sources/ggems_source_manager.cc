/*!
  \file ggems_source_manager.cc

  \brief GGEMS class managing the source in GGEMS, every new sources in GGEMS
  inherit from this class

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 15, 2019
*/

#include <algorithm>
#include <sstream>

#include "GGEMS/sources/ggems_source_manager.hh"
#include "GGEMS/tools/print.hh"
#include "GGEMS/global/ggems_constants.hh"
#include "GGEMS/tools/functions.hh"
#include "GGEMS/maths/geometry_transformation.hh"

#ifdef _WIN32
#ifdef min
#undef min
#endif
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSourceManager* GGEMSSourceManager::p_current_source_ = nullptr;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSourceManager::GGEMSSourceManager()
: is_initialized_(false),
  particle_type_(99),
  p_kernel_get_primaries_(nullptr),
  p_particle_(nullptr),
  p_random_generator_(nullptr),
  opencl_manager_(OpenCLManager::GetInstance())
{
  GGEMScout("GGEMSSourceManager", "GGEMSSourceManager", 3)
    << "Allocation of GGEMSSourceManager..." << GGEMSendl;

  // Allocation of geometry transformation
  p_geometry_transformation_ = new GeometryTransformation;

  // Storing the pointer
  p_current_source_ = this;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSourceManager::~GGEMSSourceManager(void)
{
  if (p_geometry_transformation_) {
    delete p_geometry_transformation_;
    p_geometry_transformation_ = nullptr;
  }

  GGEMScout("GGEMSSourceDefinition", "~GGEMSSourceDefinition", 3)
    << "Deallocation of GGEMSSourceDefinition..." << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::DeleteInstance(void)
{
  if (p_current_source_)
  {
    delete p_current_source_;
    p_current_source_ = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

bool GGEMSSourceManager::IsReady(void) const
{
  return is_initialized_;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::SetPosition(float const& pos_x, float const& pos_y,
  float const& pos_z)
{
  p_geometry_transformation_->SetTranslation(
    MakeFloatXYZ(pos_x, pos_y, pos_z));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::SetParticle(Particle* const p_particle)
{
  p_particle_ = p_particle;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::SetRandomGenerator(
  RandomGenerator* const p_random_generator)
{
  p_random_generator_ = p_random_generator;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::SetSourceParticleType(char const* particle_type)
{
  // Convert the particle type in string
  std::string particle_type_str(particle_type);

  // Transform the string to lower character
  std::transform(particle_type_str.begin(), particle_type_str.end(),
    particle_type_str.begin(), ::tolower);

  if (!particle_type_str.compare("photon")) {
    particle_type_ = ParticleName::PHOTON;
  }
  else if (!particle_type_str.compare("electron")) {
    particle_type_ = ParticleName::ELECTRON;
  }
  else
  {
    Misc::ThrowException("GGEMSSourceManager", "SetParticleType",
      "Unknown particle!!!");
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::SetLocalAxis(
  float const& m00, float const& m01, float const& m02,
  float const& m10, float const& m11, float const& m12,
  float const& m20, float const& m21, float const& m22)
{
  p_geometry_transformation_->SetAxisTransformation(
    m00, m01, m02,
    m10, m11, m12,
    m20, m21, m22
  );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::SetRotation(float const& rx, float const& ry,
  float const& rz)
{
  p_geometry_transformation_->SetRotation(MakeFloatXYZ(rx, ry, rz));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::UpdateRotation(float const& rx, float const& ry,
  float const& rz)
{
  p_geometry_transformation_->SetRotation(MakeFloatXYZ(rx, ry, rz));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSourceManager::CheckParameters(void) const
{
  GGEMScout("GGEMSSourceManager", "CheckParameters", 3)
    << "Checking the mandatory parameters..." << GGEMSendl;

  // Checking the type of particles
  if (particle_type_ == 99) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a particle type for the source:" << std::endl;
    oss << "    - Photon" << std::endl;
    oss << "    - Electron" << std::endl;
    Misc::ThrowException("GGEMSSourceManager", "CheckParameters", oss.str());
  }

  // Checking the particle pointer
  if (!p_particle_) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "The particle pointer is empty in source manager!!!" << std::endl;
    Misc::ThrowException("GGEMSSourceManager", "CheckParameters", oss.str());
  }

  // Checking the random generator pointer
  if (!p_random_generator_) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "The random generator pointer is empty in source manager!!!" << std::endl;
    Misc::ThrowException("GGEMSSourceManager", "CheckParameters", oss.str());
  }

  // Checking the position of particles
  cl_float3 const kPosition = p_geometry_transformation_->GetPosition();
  if (Misc::IsEqual(kPosition.s[0], std::numeric_limits<float>::min()) ||
      Misc::IsEqual(kPosition.s[1], std::numeric_limits<float>::min()) ||
      Misc::IsEqual(kPosition.s[2], std::numeric_limits<float>::min())) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a position for the source!!!";
    Misc::ThrowException("GGEMSSourceManager", "CheckParameters", oss.str());
  }

  // Checking the rotation of particles
  cl_float3 const kRotation = p_geometry_transformation_->GetRotation();
  if (Misc::IsEqual(kRotation.s[0], std::numeric_limits<float>::min()) ||
      Misc::IsEqual(kRotation.s[1], std::numeric_limits<float>::min()) ||
      Misc::IsEqual(kRotation.s[2], std::numeric_limits<float>::min())) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a rotation for the source!!!";
    Misc::ThrowException("GGEMSSourceManager", "CheckParameters", oss.str());
  }
}
