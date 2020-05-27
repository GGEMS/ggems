/*!
  \file GGEMSSolid.cc

  \brief GGEMS class for solid. This class store geometry about phantom or detector

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 2, 2020
*/

#include "GGEMS/geometries/GGEMSSolid.hh"
#include "GGEMS/io/GGEMSMHDImage.hh"
#include "GGEMS/sources/GGEMSSourceManager.hh"
#include "GGEMS/physics/GGEMSParticles.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolid::GGEMSSolid(void)
: solid_data_(nullptr),
  label_data_(nullptr),
  kernel_particle_solid_distance_(nullptr)
{
  GGcout("GGEMSSolid", "GGEMSSolid", 3) << "Allocation of GGEMSSolid..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get the RAM manager
  GGEMSRAMManager& ram_manager = GGEMSRAMManager::GetInstance();

  // Allocation of memory on OpenCL device for header data
  solid_data_ = opencl_manager.Allocate(nullptr, sizeof(GGEMSVoxelizedSolidData), CL_MEM_READ_WRITE);
  ram_manager.AddGeometryRAMMemory(sizeof(GGEMSVoxelizedSolidData));

  // Initializing OpenCL kernel
  InitializeKernel();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSolid::~GGEMSSolid(void)
{
  GGcout("GGEMSSolid", "~GGEMSSolid", 3) << "Deallocation of GGEMSSolid..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::InitializeKernel(void)
{
  GGcout("GGEMSSolid", "InitializeKernel", 3) << "Initializing kernel..." << GGendl;

  // Getting the path to kernel
  std::string const kOpenCLKernelPath = OPENCL_KERNEL_PATH;
  std::string const kFilename = kOpenCLKernelPath + "/ParticlesVoxelizedSolidDistance.cl";

  // Compiling the kernel
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  kernel_particle_solid_distance_ = opencl_manager.CompileKernel(kFilename, "particles_voxelized_solid_distance");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::DistanceFromParticle(void)
{
  // Getting the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  cl::CommandQueue* queue = opencl_manager.GetCommandQueue();
  cl::Event* event = opencl_manager.GetEvent();

  // Getting the buffer of primary particles from source
  GGEMSSourceManager& source_manager = GGEMSSourceManager::GetInstance();
  GGEMSParticles* particles = source_manager.GetParticles();
  cl::Buffer* primary_particles = particles->GetPrimaryParticles();

  // Getting the number of particles
  GGulong const kNumberOfParticles = particles->GetNumberOfParticles();

  // Set parameters for kernel
  kernel_particle_solid_distance_->setArg(0, *primary_particles);
  kernel_particle_solid_distance_->setArg(1, *solid_data_);

  // Define the number of work-item to launch
  cl::NDRange global(kNumberOfParticles);
  cl::NDRange offset(0);

  // Launching kernel
  cl_int kernel_status = queue->enqueueNDRangeKernel(*kernel_particle_solid_distance_, offset, global, cl::NullRange, nullptr, event);
  opencl_manager.CheckOpenCLError(kernel_status, "GGEMSSolid", "DistanceFromParticle");
  queue->finish(); // Wait until the kernel status is finish

  // Displaying time in kernel
  opencl_manager.DisplayElapsedTimeInKernel("DistanceFromParticle in GGEMSSolid");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::LoadPhantomImage(std::string const& phantom_filename, std::string const& range_data_filename, std::shared_ptr<GGEMSMaterials> materials)
{
  GGcout("GGEMSSolid", "LoadPhantomImage", 3) << "Loading image phantom from mhd file..." << GGendl;

  // Read MHD input file
  GGEMSMHDImage mhd_input_phantom;
  mhd_input_phantom.Read(phantom_filename, solid_data_);

  // Get the name of raw file from mhd reader
  std::string const kRawFilename = mhd_input_phantom.GetRawMDHfilename();

  // Get the type
  std::string const kDataType = mhd_input_phantom.GetDataMHDType();

  // Convert raw data to material id data
  if (!kDataType.compare("MET_CHAR")) {
    ConvertImageToLabel<char>(kRawFilename, range_data_filename, materials);
  }
  else if (!kDataType.compare("MET_UCHAR")) {
    ConvertImageToLabel<unsigned char>(kRawFilename, range_data_filename, materials);
  }
  else if (!kDataType.compare("MET_SHORT")) {
    ConvertImageToLabel<GGshort>(kRawFilename, range_data_filename, materials);
  }
  else if (!kDataType.compare("MET_USHORT")) {
    ConvertImageToLabel<GGushort>(kRawFilename, range_data_filename, materials);
  }
  else if (!kDataType.compare("MET_INT")) {
    ConvertImageToLabel<GGint>(kRawFilename, range_data_filename, materials);
  }
  else if (!kDataType.compare("MET_UINT")) {
    ConvertImageToLabel<GGuint>(kRawFilename, range_data_filename, materials);
  }
  else if (!kDataType.compare("MET_FLOAT")) {
    ConvertImageToLabel<GGfloat>(kRawFilename, range_data_filename, materials);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::ApplyOffset(GGfloat3 const& offset_xyz)
{
  GGcout("GGEMSSolid", "ApplyOffset", 3) << "Applyng the offset defined by the user..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device
  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_, sizeof(GGEMSVoxelizedSolidData));

  for (GGuint i = 0; i < 3; ++i ) {
    // Offset
    solid_data_device->offsets_xyz_.s[i] = offset_xyz.s[i];

    // Bounding box
    solid_data_device->border_min_xyz_.s[i] = -solid_data_device->offsets_xyz_.s[i];
    solid_data_device->border_max_xyz_.s[i] = solid_data_device->border_min_xyz_.s[i] + solid_data_device->number_of_voxels_xyz_.s[i] * solid_data_device->voxel_sizes_xyz_.s[i];
  }

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_, solid_data_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetGeometryTolerance(GGfloat const& tolerance)
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device
  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_, sizeof(GGEMSVoxelizedSolidData));

  // Storing the geometry tolerance
  solid_data_device->tolerance_ = tolerance;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_, solid_data_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::SetNavigatorID(std::size_t const& navigator_id)
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device
  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_, sizeof(GGEMSVoxelizedSolidData));

  // Storing the geometry tolerance
  solid_data_device->navigator_id_ = static_cast<GGint>(navigator_id);

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_, solid_data_device);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSolid::PrintInfos(void) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get pointer on OpenCL device
  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_, sizeof(GGEMSVoxelizedSolidData));

  GGcout("GGEMSSolid", "PrintInfos", 0) << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "GGEMSSolid Infos: " << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "--------------------------------------------" << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "*Dimension: " << solid_data_device->number_of_voxels_xyz_.s[0] << " " << solid_data_device->number_of_voxels_xyz_.s[1] << " " << solid_data_device->number_of_voxels_xyz_.s[2] << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "*Number of voxels: " << solid_data_device->number_of_voxels_ << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "*Size of voxels: (" << solid_data_device->voxel_sizes_xyz_.s[0] << "x" << solid_data_device->voxel_sizes_xyz_.s[1] << "x" << solid_data_device->voxel_sizes_xyz_.s[2] << ") mm3" << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "*Offset: (" << solid_data_device->offsets_xyz_.s[0] << "x" << solid_data_device->offsets_xyz_.s[1] << "x" << solid_data_device->offsets_xyz_.s[2] << ") mm3" << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "*Bounding box:" << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "    - X: " << solid_data_device->border_min_xyz_.s[0] << " <-> " << solid_data_device->border_max_xyz_.s[0] << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "    - Y: " << solid_data_device->border_min_xyz_.s[1] << " <-> " << solid_data_device->border_max_xyz_.s[1] << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "    - Z: " << solid_data_device->border_min_xyz_.s[2] << " <-> " << solid_data_device->border_max_xyz_.s[2] << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << "*Geometry tolerance: " << solid_data_device->tolerance_ << " mm" << GGendl;
  GGcout("GGEMSSolid", "PrintInfos", 0) << GGendl;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_, solid_data_device);
}
