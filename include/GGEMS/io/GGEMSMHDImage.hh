#ifndef GUARD_GGEMS_IO_GGEMSMHDIMAGE_HH
#define GUARD_GGEMS_IO_GGEMSMHDIMAGE_HH

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
  \file GGEMSMHDImage.hh

  \brief I/O class handling MHD image file

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday January 14, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <fstream>

#include "GGEMS/global/GGEMSOpenCLManager.hh"

/*!
  \class GGEMSMHDImage
  \brief I/O class handling MHD file
*/
class GGEMS_EXPORT GGEMSMHDImage
{
  public:
    /*!
      \brief GGEMSMHDImage constructor
    */
    GGEMSMHDImage(void);

    /*!
      \brief GGEMSMHDImage destructor
    */
    ~GGEMSMHDImage(void);

  public:
    /*!
      \fn GGEMSMHDImage(GGEMSMHDImage const& mhd) = delete
      \param mhd - reference on the mhd file
      \brief Avoid copy of the class by reference
    */
    GGEMSMHDImage(GGEMSMHDImage const& mhd) = delete;

    /*!
      \fn GGEMSMHDImage& operator=(GGEMSMHDImage const& mhd) = delete
      \param mhd - reference on the mhd file
      \brief Avoid assignement of the class by reference
    */
    GGEMSMHDImage& operator=(GGEMSMHDImage const& mhd) = delete;

    /*!
      \fn GGEMSMHDImage(GGEMSMHDImage const&& mhd) = delete
      \param mhd - rvalue reference on the mhd file
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMHDImage(GGEMSMHDImage const&& mhd) = delete;

    /*!
      \fn GGEMSMHDImage& operator=(GGEMSMHDImage const&& mhd) = delete
      \param mhd - rvalue reference on the mhd file
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSMHDImage& operator=(GGEMSMHDImage const&& mhd) = delete;

    /*!
      \fn void SetOutputFileName(std::string const& basename)
      \param basename - mhd file name
      \brief set the output filename (*.mhd)
    */
    void SetOutputFileName(std::string const& basename);

    /*!
      \fn void Read(std::string const& image_mhd_header_filename, cl::Buffer* solid_data, GGsize const& thread_index)
      \param image_mhd_header_filename - input mhd filename
      \param solid_data - pointer on solid data
      \param thread_index - index of the thread (= activated device index)
      \brief read the mhd header
    */
    void Read(std::string const& image_mhd_header_filename, cl::Buffer* solid_data, GGsize const& thread_index);

    /*!
      \fn void Write(cl::Buffer* image, GGsize const& thread_index) const
      \param image - image to write on output file
      \param thread_index - index of the thread (= activated device index)
      \brief Write mhd header/raw file
    */
    void Write(cl::Buffer* image, GGsize const& thread_index) const;

    /*!
      \fn template <typename T> void Write(T* image)
      \tparam T - type of the data
      \param image - image to write on output file
      \brief write the raw data to file
    */
    template<typename T>
    void Write(T* image);

    /*!
      \fn void SetElementSizes(GGfloat3 const& element_sizes)
      \param element_sizes - size of elements in X, Y, Z
      \brief set the size of the elements
    */
    void SetElementSizes(GGfloat3 const& element_sizes);

    /*!
      \fn void SetDimensions(GGsize3 const& dimensions)
      \param dimensions - dimensions of image in X, Y, Z
      \brief set the dimensions of the image
    */
    void SetDimensions(GGsize3 const& dimensions);

    /*!
      \fn void SetDataType(std::string const& data_type)
      \param data_type - type of data
      \brief set the type of data
    */
    void SetDataType(std::string const& data_type);

    /*!
      \fn std::string GetDataMHDType(void) const
      \brief get the mhd data type
      \return the type of data for mhd file
    */
    inline std::string GetDataMHDType(void) const {return mhd_data_type_;}

    /*!
      \fn std::string GetRawMDHfilename(void) const
      \brief get the filename of raw data
      \return the name of raw file
    */
    inline std::string GetRawMDHfilename(void) const {return mhd_raw_file_;}

    /*!
      \fn std::string GetOutputDirectory(void) const
      \brief get the output directory
      \return the name of output directory
    */
    inline std::string GetOutputDirectory(void) const {return output_dir_;}

  private:
    /*!
      \fn void CheckParameters(void) const
      \brief Check parameters before read/write MHD file
    */
    void CheckParameters(void) const;

    /*!
      \fn template <typename T> void WriteRaw(cl::Buffer* image, GGsize const& thread_index) const
      \tparam T - type of the data
      \param image - image to write on output file
      \param thread_index - index of the thread (= activated device index)
      \brief write the raw data to file
    */
    template <typename T>
    void WriteRaw(cl::Buffer* image, GGsize const& thread_index) const;

  private:
    std::string mhd_header_file_; /*!< Name of the MHD header file */
    std::string mhd_raw_file_; /*!< Name of the MHD raw file */
    std::string output_dir_; /*!< Output directory */
    std::string mhd_data_type_; /*!< Type of data */
    GGfloat3 element_sizes_; /*!< Size of elements */
    GGsize3 dimensions_; /*!< Dimension volume X, Y, Z */
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GGEMSMHDImage::Write(T* image)
{
  GGcout("GGEMSMHDImage", "Write", 1) << "Writing MHD Image " <<  mhd_header_file_ << "..." << GGendl;

  // Checking parameters before to write
  CheckParameters();

  // header data
  std::ofstream out_header_stream(mhd_header_file_, std::ios::out);
  out_header_stream << "ObjectType = Image" << std::endl;
  out_header_stream << "BinaryDataByteOrderMSB = False" << std::endl;
  out_header_stream << "NDims = 3" << std::endl;
  out_header_stream << "ElementSpacing = " << element_sizes_.s[0] << " " << element_sizes_.s[1] << " " << element_sizes_.s[2] << std::endl;
  out_header_stream << "DimSize = " << dimensions_.x_ << " " << dimensions_.y_ << " " << dimensions_.z_ << std::endl;
  out_header_stream << "ElementType = " << mhd_data_type_ << std::endl;
  out_header_stream << "ElementDataFile = " << mhd_raw_file_ << std::endl;
  out_header_stream.close();

  // raw data
  std::ofstream out_raw_stream(output_dir_+mhd_raw_file_, std::ios::out | std::ios::binary);

  // Writing data on file
  out_raw_stream.write(reinterpret_cast<char*>(image), static_cast<std::streamsize>(dimensions_.x_ * dimensions_.y_* dimensions_.z_ * sizeof(T)));

  out_raw_stream.close();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void GGEMSMHDImage::WriteRaw(cl::Buffer* image, GGsize const& thread_index) const
{
  // Get the OpenCL manager
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // raw data
  std::ofstream out_raw_stream(output_dir_+mhd_raw_file_, std::ios::out | std::ios::binary);

  // Mapping data
  T* data_image_device = opencl_manager.GetDeviceBuffer<T>(image, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, dimensions_.x_ * dimensions_.y_ * dimensions_.z_ * sizeof(T), thread_index);

  // Writing data on file
  out_raw_stream.write(reinterpret_cast<char*>(data_image_device), static_cast<std::streamsize>(dimensions_.x_ * dimensions_.y_* dimensions_.z_ * sizeof(T)));

  // Release the pointers
  opencl_manager.ReleaseDeviceBuffer(image, data_image_device, thread_index);
  out_raw_stream.close();
}

#endif // End of GUARD_GGEMS_IO_GGEMSMHDIMAGE_HH
