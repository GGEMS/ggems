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

#include "GGEMS/graphics/GGEMSOpenGLParticles.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/graphics/GGEMSOpenGLManager.hh"
#include "GGEMS/physics/GGEMSPrimaryParticles.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLParticles::GGEMSOpenGLParticles(void)
{
  GGcout("GGEMSOpenGLParticles", "GGEMSOpenGLParticles", 3) << "GGEMSOpenGLParticles creating..." << GGendl;

  GGcout("GGEMSOpenGLParticles", "GGEMSOpenGLParticles", 3) << "GGEMSOpenGLParticles created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOpenGLParticles::~GGEMSOpenGLParticles(void)
{
  GGcout("GGEMSOpenGLParticles", "~GGEMSOpenGLParticles", 3) << "GGEMSOpenGLParticles erasing..." << GGendl;

  GGcout("GGEMSOpenGLParticles", "~GGEMSOpenGLParticles", 3) << "GGEMSOpenGLParticles erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOpenGLParticles::CopyParticlePosition(GGsize const& source_index)
{
  // Getting primary particles
  GGEMSOpenGLManager& opengl_manager = GGEMSOpenGLManager::GetInstance();
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  cl::Buffer* primary_particles = source_manager.GetParticles()->GetPrimaryParticles(0);

  // Get the primary_particles pointer on OpenCL device
  GGEMSPrimaryParticles* primary_particles_device = opencl_manager.GetDeviceBuffer<GGEMSPrimaryParticles>(primary_particles, CL_TRUE, CL_MAP_READ, sizeof(GGEMSPrimaryParticles), 0);

  GGint number_of_displayed_particles = opengl_manager.GetNumberOfDisplayedParticles();

  // If very few particles check if number of simulated particles is inferior to displayed particles
  if (source_manager.GetNumberOfParticlesInBatch(source_index, 0, 0) < number_of_displayed_particles) {
    number_of_displayed_particles = static_cast<GGint>(source_manager.GetNumberOfParticlesInBatch(source_index, 0, 0));
  }

  for (GGint i = 0; i < number_of_displayed_particles; ++i) {
    std::cout << "*****" << std::endl;
    std::cout << primary_particles_device->stored_particles_gl_[i] << std::endl;
    GGint stored_particles = primary_particles_device->stored_particles_gl_[i];
    for (GGint j = 0; j < stored_particles; ++j) {
      std::cout << primary_particles_device->px_gl_[j] << " " << primary_particles_device->py_gl_[j] << " " << primary_particles_device->pz_gl_[j] << std::endl;
    }
  }

  // Release the pointers
  opencl_manager.ReleaseDeviceBuffer(primary_particles, primary_particles_device, 0);
}
