/*!
  \file xray_source.cc

  \brief This class define a XRay source in GGEMS useful for CT/CBCT simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday October 22, 2019
*/

#include <sstream>

#include "GGEMS/sources/xray_source.hh"
#include "GGEMS/tools/print.hh"
#include "GGEMS/global/ggems_constants.hh"
#include "GGEMS/tools/functions.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

XRaySource::XRaySource(void)
: GGEMSSourceManager(),
  beam_aperture_(std::numeric_limits<float>::min())
{
  GGEMScout("XRaySource", "XRaySource", 1)
    << "Allocation of XRaySource..." << GGEMSendl;

  // Initialization of parameters
  focal_spot_size_ = Matrix::MakeFloatXYZ(
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::min()
  );

  // Initialization of local axis
  p_geometry_transformation_->SetAxisTransformation(
    0.0f, 0.0f, -1.0f,
    0.0f, 1.0f, 0.0f,
    1.0f, 0.0f, 0.0f
  );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

XRaySource::~XRaySource(void)
{
  GGEMScout("XRaySource", "~XRaySource", 1)
    << "Deallocation of XRaySource..." << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::GetPrimaries(cl::Buffer* p_primary_particles)
{
  if (p_primary_particles) std::cout << "Test" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::PrintInfos(void) const
{
  GGEMScout("XRaySource", "PrintInfos", 0) << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "XRaySource Infos:" << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "-----------------" << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "*Particle type: ";
  if (particle_type_ == ParticleName::PHOTON) {
    std::cout << "Photon" << std::endl;
  }
  if (particle_type_ == ParticleName::ELECTRON) {
    std::cout << "Electron" << std::endl;
  }
  GGEMScout("XRaySource", "PrintInfos", 0) << "*Position: " << "("
    << p_geometry_transformation_->GetPosition().s[0] << ", "
    << p_geometry_transformation_->GetPosition().s[1] << ", "
    << p_geometry_transformation_->GetPosition().s[2] << " ) m3" << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "*Rotation: " << "("
    << p_geometry_transformation_->GetRotation().s[0] << ", "
    << p_geometry_transformation_->GetRotation().s[1] << ", "
    << p_geometry_transformation_->GetRotation().s[2] << ") degree"
    << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "*Beam aperture: "
    << beam_aperture_ << " degrees" << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "*Focal spot size: " << "("
    << focal_spot_size_.s[0] << ", "
    << focal_spot_size_.s[1] << ", "
    << focal_spot_size_.s[2] << ") mm3" << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "*Local axis: " << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "[" << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "    "
    << p_geometry_transformation_->GetLocalAxis().m00_ << " "
    << p_geometry_transformation_->GetLocalAxis().m01_ << " "
    << p_geometry_transformation_->GetLocalAxis().m02_ << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "    "
    << p_geometry_transformation_->GetLocalAxis().m10_ << " "
    << p_geometry_transformation_->GetLocalAxis().m11_ << " "
    << p_geometry_transformation_->GetLocalAxis().m12_ << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "    "
    << p_geometry_transformation_->GetLocalAxis().m20_ << " "
    << p_geometry_transformation_->GetLocalAxis().m21_ << " "
    << p_geometry_transformation_->GetLocalAxis().m22_ << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << "]" << GGEMSendl;
  GGEMScout("XRaySource", "PrintInfos", 0) << GGEMSendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::CheckParameters(void) const
{
  GGEMScout("XRaySource", "CheckParameters", 1)
    << "Checking the mandatory parameters..." << GGEMSendl;

  // Checking the parameters of Source Manager
  GGEMSSourceManager::CheckParameters();

  // Checking the beam aperture
  if (Misc::IsEqual(beam_aperture_, std::numeric_limits<float>::min())) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a beam aperture for the source!!!";
    Misc::ThrowException("XRaySource", "CheckParameters", oss.str());
  }
  else if (beam_aperture_ < 0.0f) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "The beam aperture must be >= 0!!!";
    Misc::ThrowException("XRaySource", "CheckParameters", oss.str());
  }

  // Checking the focal spot size
  if (Misc::IsEqual(focal_spot_size_.s[0], std::numeric_limits<float>::min()) ||
      Misc::IsEqual(focal_spot_size_.s[1], std::numeric_limits<float>::min()) ||
      Misc::IsEqual(focal_spot_size_.s[2], std::numeric_limits<float>::min())) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "You have to set a focal spot size!!!";
    Misc::ThrowException("XRaySource", "CheckParameters", oss.str());
  }

  // Focal spot size must be a positive value
  if (focal_spot_size_.s[0] < 0.0f ||
      focal_spot_size_.s[1] < 0.0f ||
      focal_spot_size_.s[2] < 0.0f) {
    std::ostringstream oss(std::ostringstream::out);
    oss << "The focal spot size is a posivite value!!!";
    Misc::ThrowException("XRaySource", "CheckParameters", oss.str());
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::Initialize(void)
{
  // Check the mandatory parameters
  CheckParameters();

  // The source is initialized
  is_initialized_ = true;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::SetBeamAperture(float const& beam_aperture)
{
  beam_aperture_ = beam_aperture;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void XRaySource::SetFocalSpotSize(float const& width, float const& height,
  float const& depth)
{
  focal_spot_size_ = Matrix::MakeFloatXYZ(width, height, depth);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

XRaySource* create_ggems_xray_source(void)
{
  return XRaySource::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void delete_ggems_xray_source(void)
{
  GGEMSSourceManager::DeleteInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void initialize_xray_source(XRaySource* p_source_manager)
{
  p_source_manager->Initialize();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_position_xray_source(XRaySource* p_source_manager, float const pos_x,
  float const pos_y, float const pos_z)
{
  p_source_manager->SetPosition(pos_x, pos_y, pos_z);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_infos_xray_source(XRaySource* p_source_manager)
{
  p_source_manager->PrintInfos();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_particle_type_xray_source(XRaySource* p_source_manager,
  char const* particle_name)
{
  p_source_manager->SetParticleType(particle_name);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_beam_aperture_xray_source(XRaySource* p_source_manager,
  float const beam_aperture)
{
  p_source_manager->SetBeamAperture(beam_aperture);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_focal_spot_size_xray_source(XRaySource* p_source_manager,
  float const width, float const height, float const depth)
{
  p_source_manager->SetFocalSpotSize(width, height, depth);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_local_axis_xray_source(XRaySource* p_source_manager,
  float const m00, float const m01, float const m02,
  float const m10, float const m11, float const m12,
  float const m20, float const m21, float const m22)
{
  p_source_manager->SetLocalAxis(
    m00, m01, m02,
    m10, m11, m12,
    m20, m21, m22
  );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_rotation_xray_source(XRaySource* p_source_manager, float const rx,
  float const ry, float const rz)
{
  p_source_manager->SetRotation(rx, ry, rz);
}
