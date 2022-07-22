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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSSTLReader::GGEMSSTLReader(void)
: stl_filename_("")
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

  // Open STL file to stream
  std::ifstream stl_stream(meshed_phantom_filename, std::ios::in | std::ios::binary);
  GGEMSFileStream::CheckInputStream(stl_stream, meshed_phantom_filename);

  stl_filename_ = meshed_phantom_filename;

  GGcout("GGEMSSTLReader", "Read", 2) << "STL file loaded" << GGendl;

  // Reading file
  stl_stream.read(reinterpret_cast<char*>(header_), sizeof(GGuchar)*80);
  stl_stream.read(reinterpret_cast<char*>(&number_of_triangles_), sizeof(GGuint)*1);

  // Initializing max and min points of mes
  float high_point[] = {
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::min(),
    std::numeric_limits<float>::min()
  };

  float low_point[] = {
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max()
  };



  GGcout("GGEMSSTLReader", "Read", 1) << "STL filename: " << stl_filename_ << GGendl;
  GGcout("GGEMSSTLReader", "Read", 1) << "Header: " << header_ << GGendl;
  GGcout("GGEMSSTLReader", "Read", 1) << "Number of triangles: " << number_of_triangles_ << GGendl;



/*
  std::cout << "Center: " << center_.x_[0] << " " << center_.x_[1] << " " << center_.x_[2] << std::endl;
  std::cout << "Half width: " << half_width_[0] << " " << half_width_[1] << " " << half_width_[2] << std::endl;
  std::cout << "*****" << std::endl;
*/




  stl_stream.close();

/*
  // Min and max points
  Point lo(FLT_MAX, FLT_MAX, FLT_MAX);
  Point hi(FLT_MIN, FLT_MIN, FLT_MIN);

  float data[12];
  unsigned short octet_attribut;
  triangles_ = new Triangle[number_of_triangles_];
  for (unsigned int i = 0; i < number_of_triangles_; ++i) {
    stl_stream.read(reinterpret_cast<char*>(data), sizeof(float)*12);
    stl_stream.read(reinterpret_cast<char*>(&octet_attribut), sizeof(unsigned short)*1);
    triangles_[i] = Triangle(
      Point(data[3], data[4], data[5]),
      Point(data[6], data[7], data[8]),
      Point(data[9], data[10], data[11])
    );

    for (int j = 0; j < 3; ++j) { // Loop over points
      for (int k = 0; k < 3; ++k) { // Loop over dimensions
        if (triangles_[i].p_[j].x_[k] < lo.x_[k]) lo.x_[k] = triangles_[i].p_[j].x_[k];
        if (triangles_[i].p_[j].x_[k] > hi.x_[k]) hi.x_[k] = triangles_[i].p_[j].x_[k];
      }
    }
  }

  // Expand it by 10% so that all points are well interior
  for (int i = 0; i < 3; ++i) {
    float expanding = (hi.x_[i]-lo.x_[i])*0.1f;
    lo.x_[i] -= expanding;
    hi.x_[i] += expanding;

    // Selecting correct integer
    if (lo.x_[i] < 0.0f) lo.x_[i] = std::floor(lo.x_[i]);
    else lo.x_[i] = std::ceil(lo.x_[i]);

    if (hi.x_[i] < 0.0f) hi.x_[i] = std::floor(hi.x_[i]);
    else hi.x_[i] = std::ceil(hi.x_[i]);
  }

  // Computing the center of octree box
  center_ = Point(
    (hi.x_[0]+lo.x_[0])*0.5f,
    (hi.x_[1]+lo.x_[1])*0.5f,
    (hi.x_[2]+lo.x_[2])*0.5f
  );

  // Computing half width for each axes
  half_width_[0] = (hi.x_[0]-lo.x_[0])*0.5f;
  half_width_[1] = (hi.x_[1]-lo.x_[1])*0.5f;
  half_width_[2] = (hi.x_[2]-lo.x_[2])*0.5f;
*/

/*
  std::cout << "*****" << std::endl;
  std::cout << "STL filename: " << stl_filename_ << std::endl;
  std::cout << "Header: " << header_<< std::endl;
  std::cout << "Number of triangles: " << number_of_triangles_ << std::endl;
  std::cout << "Center: " << center_.x_[0] << " " << center_.x_[1] << " " << center_.x_[2] << std::endl;
  std::cout << "Half width: " << half_width_[0] << " " << half_width_[1] << " " << half_width_[2] << std::endl;
  std::cout << "*****" << std::endl;
*/
}
