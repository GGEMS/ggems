#ifndef GUARD_GGEMS_IO_GGEMSSTLREADER_HH
#define GUARD_GGEMS_IO_GGEMSSTLREADER_HH

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
  \file GGEMSSTLReader.hh

  \brief I/O class handling STL mesh file

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday July 7, 2022
*/

#include <string>

/*!
  \class GGEMSSTLReader
  \brief I/O class handling STL file
*/
class GGEMSSTLReader
{
  public:
    /*!
      \brief GGEMSSTLReader constructor
    */
    GGEMSSTLReader(void);

    /*!
      \brief GGEMSSTLReader destructor
    */
    ~GGEMSSTLReader(void);

    /*!
      \fn GGEMSSTLReader(GGEMSSTLReader const& stl) = delete
      \param stl - reference on the stl file
      \brief Avoid copy of the class by reference
    */
    GGEMSSTLReader(GGEMSSTLReader const& stl) = delete;

    /*!
      \fn GGEMSSTLReader(GGEMSSTLReader const&& stl) = delete
      \param stl - rvalue reference on the stl file
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSTLReader(GGEMSSTLReader const&& stl) = delete;

    /*!
      \fn GGEMSSTLReader& operator=(GGEMSSTLReader const& stl) = delete
      \param stl - reference on the stl file
      \brief Avoid assignement of the class by reference
    */
    GGEMSSTLReader& operator=(GGEMSSTLReader const& stl) = delete;

    /*!
      \fn GGEMSSTLReader& operator=(GGEMSSTLReader const&& stl) = delete
      \param stl - rvalue reference on the stl file
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSTLReader& operator=(GGEMSSTLReader const&& stl) = delete;

    /*!
      \fn void Read(std::string const& meshed_phantom_filename)
      \param meshed_phantom_filename - input stl file
      \brief read the stl file and store data
    */
    void Read(std::string const& meshed_phantom_filename);

  private:
    std::string stl_filename_;
    GGuchar header_[80];
    GGuint number_of_triangles_;
    /*Triangle* triangles_;
    Point center_;
    float half_width_[3];
*/
};

#endif // End of GUARD_GGEMS_IO_GGEMSSTLREADER_HH
