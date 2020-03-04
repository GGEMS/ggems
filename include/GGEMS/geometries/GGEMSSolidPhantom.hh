#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSSOLIDPHANTOM_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSSOLIDPHANTOM_HH

/*!
  \file GGEMSSolidPhantom.hh

  \brief GGEMS class for solid phantom. This class reads the phantom volume infos, the range data file

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 2, 2020
*/

#include <string>
#include <memory>
#include <fstream>
#include <limits>
#include <algorithm>

#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/io/GGEMSTextReader.hh"
#include "GGEMS/physics/GGEMSMaterials.hh"

/*!
  \class GGEMSSolidPhantom
  \brief GGEMS class for solid phantom informations
*/
class GGEMS_EXPORT GGEMSSolidPhantom
{
  public:
    /*!
      \brief GGEMSSolidPhantom constructor
    */
    explicit GGEMSSolidPhantom(std::shared_ptr<GGEMSMaterials> materials);

    /*!
      \brief GGEMSSolidPhantom destructor
    */
    ~GGEMSSolidPhantom(void);

    /*!
      \fn GGEMSSolidPhantom(GGEMSSolidPhantom const& solid_phantom) = delete
      \param solid_phantom - reference on the GGEMS solid phantom
      \brief Avoid copy by reference
    */
    GGEMSSolidPhantom(GGEMSSolidPhantom const& solid_phantom) = delete;

    /*!
      \fn GGEMSSolidPhantom& operator=(GGEMSSolidPhantom const& solid_phantom) = delete
      \param solid_phantom - reference on the GGEMS solid phantom
      \brief Avoid assignement by reference
    */
    GGEMSSolidPhantom& operator=(GGEMSSolidPhantom const& solid_phantom) = delete;

    /*!
      \fn GGEMSSolidPhantom(GGEMSSolidPhantom const&& solid_phantom) = delete
      \param solid_phantom - rvalue reference on the GGEMS solid phantom
      \brief Avoid copy by rvalue reference
    */
    GGEMSSolidPhantom(GGEMSSolidPhantom const&& solid_phantom) = delete;

    /*!
      \fn GGEMSSolidPhantom& operator=(GGEMSSolidPhantom const&& solid_phantom) = delete
      \param solid_phantom - rvalue reference on the GGEMS solid phantom
      \brief Avoid copy by rvalue reference
    */
    GGEMSSolidPhantom& operator=(GGEMSSolidPhantom const&& solid_phantom) = delete;

    /*!
      \fn void LoadPhantomImage(std::string const& phantom_filename, std::string const& range_data_filename)
      \param phantom_filename - name of the MHF file containing the phantom
      \param range_data_filename - name of the file containing the range to material data
      \brief load phantom image to GGEMS and create a volume of label in GGEMS
    */
    void LoadPhantomImage(std::string const& phantom_filename, std::string const& range_data_filename);

    /*!
      \fn void ApplyOffset(GGdouble3 const& offset_xyz)
      \param offset_xyz - offset in X, Y and Z
      \brief apply an offset defined by the user
    */
    void ApplyOffset(GGdouble3 const& offset_xyz);

    /*!
      \fn void PrintInfos(void) const
      \brief printing infos about solid phantom
    */
    void PrintInfos(void) const;

  private:
    /*!
      \fn template <typename T> void ConvertImageToLabel(std::string const& raw_data_filename, std::string const& range_data_filename)
      \tparam T - type of data
      \param raw_data_filename - raw data filename from mhd
      \param range_data_filename - name of the file containing the range to material data
      \brief convert image data to label data
    */
    template <typename T>
    void ConvertImageToLabel(std::string const& raw_data_filename, std::string const& range_data_filename);

  private:
    std::shared_ptr<cl::Buffer> solid_phantom_data_; /*!< Data about solid phantom */
    std::shared_ptr<cl::Buffer> label_data_; /*!< Pointer storing the buffer about label data */
    std::shared_ptr<GGEMSMaterials> materials_; /*!< Material of phantom */
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to OpenCL manager singleton */
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GGEMSSolidPhantom::ConvertImageToLabel(std::string const& raw_data_filename, std::string const& range_data_filename)
{
  GGcout("GGEMSSolidPhantom", "ConvertImageToLabel", 3) << "Converting image material data to label data..." << GGendl;

  // Checking if file exists
  std::ifstream in_raw_stream(raw_data_filename, std::ios::in | std::ios::binary);
  GGEMSFileStream::CheckInputStream(in_raw_stream, raw_data_filename);

  // Get pointer on OpenCL device
  GGEMSSolidPhantomData* solid_data = opencl_manager_.GetDeviceBuffer<GGEMSSolidPhantomData>(solid_phantom_data_, sizeof(GGEMSSolidPhantomData));

  // Get information about mhd file
  GGuint const kNumberOfVoxels = solid_data->number_of_voxels_;

  // Release the pointer
  opencl_manager_.ReleaseDeviceBuffer(solid_phantom_data_, solid_data);

  // Reading data to a tmp buffer
  std::unique_ptr<T[]> tmp_raw_data = std::make_unique<T[]>(kNumberOfVoxels);
  in_raw_stream.read(reinterpret_cast<char*>(tmp_raw_data.get()), kNumberOfVoxels * sizeof(T));

  // Closing file
  in_raw_stream.close();

  // Allocating memory on OpenCL device
  label_data_ = opencl_manager_.Allocate(nullptr, kNumberOfVoxels * sizeof(GGuchar), CL_MEM_READ_WRITE);
  opencl_manager_.AddRAMMemory(kNumberOfVoxels * sizeof(GGuchar));

  // Get pointer on OpenCL device
  GGuchar* label_data = opencl_manager_.GetDeviceBuffer<GGuchar>(label_data_, kNumberOfVoxels * sizeof(GGuchar));

  // Set value to max of GGushort$
  std::fill(label_data, label_data + kNumberOfVoxels, std::numeric_limits<GGuchar>::max());

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
    bool is_added = materials_->AddMaterial(material_name);
    if (!is_added) {
      GGwarn("GGEMSSolidPhantom", "ConvertImageToLabel", 0) << "The material '" << material_name << "' is already added!!!" << GGendl;
      continue;
    }

    // Setting the label
    for (GGuint i = 0; i < kNumberOfVoxels; ++i) {
      // Getting the value of phantom
      GGdouble const kValue = static_cast<GGdouble>(tmp_raw_data[i]);
      if (((kValue == first_label_value) && (kValue == last_label_value)) || ((kValue >= first_label_value) && (kValue < last_label_value))) {
        label_data[i] = label_index;
      }
    }

    // Increment the label index
    ++label_index;
  }

  // Final loop checking if a value is still max of GGuchar
  bool all_converted = true;
  for (GGuint i = 0; i < kNumberOfVoxels; ++i) {
    if (label_data[i] == std::numeric_limits<GGuchar>::max()) all_converted = false;
  }

  // Closing file
  in_range_stream.close();

  // Release the pointer
  opencl_manager_.ReleaseDeviceBuffer(label_data_, label_data);

  // Checking if all voxels converted
  if (all_converted) {
    GGcout("GGEMSSolidPhantom", "ConvertImageToLabel", 0) << "All your voxels are converted to label..." << GGendl;
  }
  else {
    GGEMSMisc::ThrowException("GGEMSSolidPhantom", "ConvertImageToLabel", "Errors(s) in the range data file!!!");
  }
}

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSSOLIDPHANTOM_HH
