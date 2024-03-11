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
  \file GGEMSSTLReader.cc

  \brief I/O class handling STL mesh file

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday July 7, 2022
*/

#include <fstream>
#include <limits>

#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/io/GGEMSSTLReader.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSTLReader::GGEMSSTLReader(void)
{
  GGcout("GGEMSSTLReader", "GGEMSSTLReader", 3) << "GGEMSSTLReader creating..." << GGendl;

  GGcout("GGEMSSTLReader", "GGEMSSTLReader", 3) << "GGEMSSTLReader created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSTLReader::~GGEMSSTLReader(void)
{
  GGcout("GGEMSSTLReader", "~GGEMSSTLReader", 3) << "GGEMSSTLReader erasing!!!" << GGendl;

  GGcout("GGEMSSTLReader", "~GGEMSSTLReader", 3) << "GGEMSSTLReader erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSSTLReader::Read(std::string const& meshed_phantom_filename)
{
  GGcout("GGEMSSTLReader", "Read", 2) << "Reading STL Image..." << GGendl;

  //GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  //size_t number_activated_devices = opencl_manager.GetNumberOfActivatedDevice();

  // Opening STL file
  std::ifstream stl_stream(meshed_phantom_filename, std::ios::in | std::ios::binary);
  GGEMSFileStream::CheckInputStream(stl_stream, meshed_phantom_filename);

  stl_stream.read(reinterpret_cast<char*>(header_), sizeof(GGuchar) * 80);
  stl_stream.read(reinterpret_cast<char*>(&number_of_triangles_), sizeof(GGuint) * 1);

  // Allocating memory for triangles in each engine
/*  triangles_mp_ = new TriangleGPU*[number_of_engines];
  for (std::size_t i = 0; i < number_of_engines; ++i) {
    ocl::Engine const* engine = ocl_manager.GetEngines()[i];
    triangles_mp_[i] = engine->SVMAllocate<TriangleGPU>(CL_MEM_READ_WRITE, number_of_triangles_ * sizeof(TriangleGPU), 0);
  }

  // Min and max points
  primitives::Point3 lo(FLT_MAX, FLT_MAX, FLT_MAX);
  primitives::Point3 hi(FLT_MIN, FLT_MIN, FLT_MIN);

  float data[12]; // Vertices for triangle from STL file
  unsigned short octet_attribut; // Useless parameter from STL file
  shapes::Triangle* triangles_tmp = new shapes::Triangle[number_of_triangles_];

  for (unsigned int i = 0; i < number_of_triangles_; ++i) {
    stl_stream.read(reinterpret_cast<char*>(data), sizeof(float) * 12);
    stl_stream.read(reinterpret_cast<char*>(&octet_attribut), sizeof(unsigned short) * 1);
    triangles_tmp[i] = shapes::Triangle(
      primitives::Point3(data[3], data[4], data[5]),
      primitives::Point3(data[6], data[7], data[8]),
      primitives::Point3(data[9], data[10], data[11])
    );

    for (int j = 0; j < 3; ++j) { // Loop over points
      if (triangles_tmp[i].pts_[j].x_ < lo.x_) lo.x_ = triangles_tmp[i].pts_[j].x_;
      if (triangles_tmp[i].pts_[j].y_ < lo.y_) lo.y_ = triangles_tmp[i].pts_[j].y_;
      if (triangles_tmp[i].pts_[j].z_ < lo.z_) lo.z_ = triangles_tmp[i].pts_[j].z_;
      if (triangles_tmp[i].pts_[j].x_ > hi.x_) hi.x_ = triangles_tmp[i].pts_[j].x_;
      if (triangles_tmp[i].pts_[j].y_ > hi.y_) hi.y_ = triangles_tmp[i].pts_[j].y_;
      if (triangles_tmp[i].pts_[j].z_ > hi.z_) hi.z_ = triangles_tmp[i].pts_[j].z_;
    }
  }

  // Loading triangles to engines using 1 thread a engine
  std::thread* thread_triangle_loading = new std::thread[number_of_engines];
  for (std::size_t i = 0; i < number_of_engines; ++i) {
    thread_triangle_loading[i] = std::thread(&STLReader::LoadTriangleOnEngine, this, triangles_tmp, i);
  }

  // Joining thread
  for (std::size_t i = 0; i < number_of_engines; ++i) thread_triangle_loading[i].join();

  delete[] thread_triangle_loading;

  // Expand it by 10% so that all points are well interior
  float expanding_x = (hi.x_ - lo.x_) * 0.1f;
  float expanding_y = (hi.y_ - lo.y_) * 0.1f;
  float expanding_z = (hi.z_ - lo.z_) * 0.1f;

  lo.x_ -= expanding_x;
  hi.x_ += expanding_x;
  lo.y_ -= expanding_y;
  hi.y_ += expanding_y;
  lo.z_ -= expanding_z;
  hi.z_ += expanding_z;

  if (lo.x_ < 0.0f) lo.x_ = std::floor(lo.x_);
  else lo.x_ = std::ceil(lo.x_);

  if (lo.y_ < 0.0f) lo.y_ = std::floor(lo.y_);
  else lo.y_ = std::ceil(lo.y_);

  if (lo.z_ < 0.0f) lo.z_ = std::floor(lo.z_);
  else lo.z_ = std::ceil(lo.z_);

  if (hi.x_ < 0.0f) hi.x_ = std::floor(hi.x_);
  else hi.x_ = std::ceil(hi.x_);

  if (hi.y_ < 0.0f) hi.y_ = std::floor(hi.y_);
  else hi.y_ = std::ceil(hi.y_);

  if (hi.z_ < 0.0f) hi.z_ = std::floor(hi.z_);
  else hi.z_ = std::ceil(hi.z_);

  // Computing the center of octree box
  center_ = primitives::Point3(
    (hi.x_+lo.x_)*0.5f,
    (hi.y_+lo.y_)*0.5f,
    (hi.z_+lo.z_)*0.5f
  );

  // Computing half width for each axes
  half_width_[0] = (hi.x_-lo.x_)*0.5f;
  half_width_[1] = (hi.y_-lo.y_)*0.5f;
  half_width_[2] = (hi.z_-lo.z_)*0.5f;

  // Deleting triangles on host
  if (triangles_tmp) {
    delete[] triangles_tmp;
    triangles_tmp = nullptr;
  }
*/
  stl_stream.close();

  std::cout << "*****" << std::endl;
  std::string header;
  header.assign(header_, header_ + 80);
  std::cout << "Header: " << header_<< std::endl;
  std::cout << "Number of triangles: " << number_of_triangles_ << std::endl;
  std::cout << "Center: " << center_.x_ << " " << center_.y_ << " " << center_.z_ << std::endl;
  std::cout << "Half width: " << half_width_[0] << " " << half_width_[1] << " " << half_width_[2] << std::endl;
  std::cout << "*****" << std::endl;
}
