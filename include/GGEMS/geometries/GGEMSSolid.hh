#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSSOLID_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSSOLID_HH

/*!
  \file GGEMSSolid.hh

  \brief GGEMS class for solid. This class store geometry about phantom or detector

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 2, 2020
*/

#include <limits>
#include <algorithm>

#include "GGEMS/io/GGEMSTextReader.hh"
#include "GGEMS/geometries/GGEMSVoxelizedSolidStack.hh"
#include "GGEMS/tools/GGEMSRAMManager.hh"
#include "GGEMS/navigators/GGEMSNavigatorManager.hh"

/*!
  \class GGEMSSolid
  \brief GGEMS class for solid (voxelized or analytical) informations
*/
class GGEMS_EXPORT GGEMSSolid
{
  public:
    /*!
      \brief GGEMSSolid constructor
    */
    GGEMSSolid(void);

    /*!
      \brief GGEMSSolid destructor
    */
    ~GGEMSSolid(void);

    /*!
      \fn GGEMSSolid(GGEMSSolid const& solid) = delete
      \param solid - reference on the GGEMS solid
      \brief Avoid copy by reference
    */
    GGEMSSolid(GGEMSSolid const& solid) = delete;

    /*!
      \fn GGEMSSolid& operator=(GGEMSSolid const& solid) = delete
      \param solid - reference on the GGEMS solid
      \brief Avoid assignement by reference
    */
    GGEMSSolid& operator=(GGEMSSolid const& solid) = delete;

    /*!
      \fn GGEMSSolid(GGEMSSolid const&& solid) = delete
      \param solid - rvalue reference on the GGEMS solid
      \brief Avoid copy by rvalue reference
    */
    GGEMSSolid(GGEMSSolid const&& solid) = delete;

    /*!
      \fn GGEMSSolid& operator=(GGEMSSolid const&& solid) = delete
      \param solid - rvalue reference on the GGEMS solid
      \brief Avoid copy by rvalue reference
    */
    GGEMSSolid& operator=(GGEMSSolid const&& solid) = delete;

    /*!
      \fn void LoadPhantomImage(std::string const& phantom_filename, std::string const& range_data_filename, std::shared_ptr<GGEMSMaterials> materials)
      \param phantom_filename - name of the MHD file containing the phantom
      \param range_data_filename - name of the file containing the range to material data
      \param materials - pointer on material for a phantom
      \brief load phantom image to GGEMS and create a volume of label in GGEMS
    */
    void LoadPhantomImage(std::string const& phantom_filename, std::string const& range_data_filename, std::shared_ptr<GGEMSMaterials> materials);

    /*!
      \fn void ApplyOffset(GGfloat3 const& offset_xyz)
      \param offset_xyz - offset in X, Y and Z
      \brief apply an offset defined by the user
    */
    void ApplyOffset(GGfloat3 const& offset_xyz);

    /*!
      \fn void SetGeometryTolerance(GGfloat const& tolerance)
      \param tolerance - geometry tolerance for computation
      \brief set the geometry tolerance
    */
    void SetGeometryTolerance(GGfloat const& tolerance);

    /*!
      \fn void SetNavigatorID(std::size_t const& navigator_id)
      \param navigator_id - index of the navigator
      \brief set the navigator index in solid data
    */
    void SetNavigatorID(std::size_t const& navigator_id);

    /*!
      \fn void DistanceFromParticle(void)
      \brief compute distance from particle position to solid and store this distance in OpenCL particle buffer
    */
    void DistanceFromParticle(void);

    /*!
      \fn void PrintInfos(void) const
      \brief printing infos about solid
    */
    void PrintInfos(void) const;

    /*!
      \fn inline std::shared_ptr<cl::Buffer> GetSolidData(void) const
      \brief get the informations about the solid geometry
      \return header data OpenCL pointer about solid
    */
    inline std::shared_ptr<cl::Buffer> GetSolidData(void) const {return solid_data_;};

  private:
    /*!
      \fn template <typename T> void ConvertImageToLabel(std::string const& raw_data_filename, std::string const& range_data_filename, std::shared_ptr<GGEMSMaterials> materials)
      \tparam T - type of data
      \param raw_data_filename - raw data filename from mhd
      \param range_data_filename - name of the file containing the range to material data
      \param materials - pointer on material for a phantom
      \brief convert image data to label data
    */
    template <typename T>
    void ConvertImageToLabel(std::string const& raw_data_filename, std::string const& range_data_filename, std::shared_ptr<GGEMSMaterials> materials);

    /*!
      \fn void InitializeKernel(void)
      \brief Initialize kernel for particle solid distance
    */
    void InitializeKernel(void);

  private:
    std::shared_ptr<cl::Buffer> solid_data_; /*!< Data about solid */
    std::shared_ptr<cl::Buffer> label_data_; /*!< Pointer storing the buffer about label data */
    std::shared_ptr<cl::Kernel> kernel_particle_solid_distance_; /*!< OpenCL kernel computing distance between particles and navigator(s) (solid(s)) */
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GGEMSSolid::ConvertImageToLabel(std::string const& raw_data_filename, std::string const& range_data_filename, std::shared_ptr<GGEMSMaterials> materials)
{
  GGcout("GGEMSSolid", "ConvertImageToLabel", 3) << "Converting image material data to label data..." << GGendl;

  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Get the RAM manager
  GGEMSRAMManager& ram_manager = GGEMSRAMManager::GetInstance();

  // Checking if file exists
  std::ifstream in_raw_stream(raw_data_filename, std::ios::in | std::ios::binary);
  GGEMSFileStream::CheckInputStream(in_raw_stream, raw_data_filename);

  // Get pointer on OpenCL device
  GGEMSVoxelizedSolidData* solid_data_device = opencl_manager.GetDeviceBuffer<GGEMSVoxelizedSolidData>(solid_data_, sizeof(GGEMSVoxelizedSolidData));

  // Get information about mhd file
  GGuint const kNumberOfVoxels = solid_data_device->number_of_voxels_;

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(solid_data_, solid_data_device);

  // Reading data to a tmp buffer
  std::vector<T> tmp_raw_data;
  tmp_raw_data.resize(kNumberOfVoxels);
  in_raw_stream.read(reinterpret_cast<char*>(&tmp_raw_data[0]), kNumberOfVoxels * sizeof(T));

  // Closing file
  in_raw_stream.close();

  // Allocating memory on OpenCL device
  label_data_ = opencl_manager.Allocate(nullptr, kNumberOfVoxels * sizeof(GGuchar), CL_MEM_READ_WRITE);
  ram_manager.AddGeometryRAMMemory(kNumberOfVoxels * sizeof(GGuchar));

  // Get pointer on OpenCL device
  GGuchar* label_data_device = opencl_manager.GetDeviceBuffer<GGuchar>(label_data_, kNumberOfVoxels * sizeof(GGuchar));

  // Set value to max of GGuchar
  std::fill(label_data_device, label_data_device + kNumberOfVoxels, std::numeric_limits<GGuchar>::max());

  // Opening range data file
  std::ifstream in_range_stream(range_data_filename, std::ios::in);
  GGEMSFileStream::CheckInputStream(in_range_stream, range_data_filename);

  // Values in the range file
  GGdouble first_label_value = 0.0;
  GGdouble last_label_value = 0.0;
  GGuchar label_index = 0;
  std::string material_name("");

  // Reading range file
  std::string line("");
  while (std::getline(in_range_stream, line)) {
    // Check if blank line
    if (GGEMSTextReader::IsBlankLine(line)) continue;

    // Getting the value in string stream
    std::istringstream iss = GGEMSRangeReader::ReadRangeMaterial(line);
    iss >> first_label_value >> last_label_value >> material_name;

    // Adding the material
    materials->AddMaterial(material_name);

    // Setting the label
    for (GGuint i = 0; i < kNumberOfVoxels; ++i) {
      // Getting the value of phantom
      GGdouble const kValue = static_cast<GGdouble>(tmp_raw_data[i]);
      if (((kValue == first_label_value) && (kValue == last_label_value)) || ((kValue >= first_label_value) && (kValue < last_label_value))) {
        label_data_device[i] = label_index;
      }
    }

    // Increment the label index
    ++label_index;
  }

  // Final loop checking if a value is still max of GGuchar
  bool all_converted = true;
  for (GGuint i = 0; i < kNumberOfVoxels; ++i) {
    if (label_data_device[i] == std::numeric_limits<GGuchar>::max()) all_converted = false;
  }

  // Closing file
  in_range_stream.close();
  tmp_raw_data.clear();

  // Release the pointer
  opencl_manager.ReleaseDeviceBuffer(label_data_, label_data_device);

  // Checking if all voxels converted
  if (all_converted) {
    GGcout("GGEMSSolid", "ConvertImageToLabel", 0) << "All your voxels are converted to label..." << GGendl;
  }
  else {
    GGEMSMisc::ThrowException("GGEMSSolid", "ConvertImageToLabel", "Errors(s) in the range data file!!!");
  }
}

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSSOLID_HH
