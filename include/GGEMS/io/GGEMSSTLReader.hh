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
#include <GGEMS/geometries/GGEMSPrimitiveGeometries.hh>

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
      \fn void Read(std::string const& meshed_phantom_filename, cl::Buffer* solid_data, GGsize const& thread_index)
      \param meshed_phantom_filename - input stl file
      \param solid_data - pointer on solid data
      \param thread_index - index of the thread (= activated device index)
      \brief read the stl file and load triangles
    */
    void Read(std::string const& meshed_phantom_filename);

    /*!
      \fn GGfloat* GetHalfWidth(void)
      \brief Get the half width of bounding box
      \return the half width of bounding box
    */
   // GGfloat* GetHalfWidth(void) { return half_width_; }

    /*!
      \fn GGEMSPoint3 GetCenter(void)
      \brief Get the center of the bounding box
      \return the center of the bounding box
    */
   // GGEMSPoint3 GetCenter(void) const { return center_; }

    /*!
      \fn void LoadTriangles(GGEMSTriangle3* triangles)
      \param triangles - SVM buffer storing triangles
      \brief Load triangles to SVM memory
    */
    void LoadTriangles(GGEMSTriangle3* triangles);

    /*!
      \fn GGuint GetNumberOfTriangles(void) const
      \brief Get the number of triangles from mesh file
      \return the number of triangles
    */
    GGuint GetNumberOfTriangles(void) const { return number_of_triangles_; }


  private:
    struct GGEMSMeshTriangle {
      GGEMSMeshTriangle(void) {}
      GGEMSMeshTriangle(GGEMSPoint3 const& p0, GGEMSPoint3 const& p1, GGEMSPoint3 const& p2);

      void MostSeparatedPointsOnAABB(GGEMSPoint3 pts[3], GGint& min, GGint& max);
      void SphereFromDistantPoints(GGEMSSphere3& s, GGEMSPoint3 pts[3]);
      void SphereOfSphereAndPoint(GGEMSSphere3& s, GGEMSPoint3& p);

      GGEMSPoint3        pts_[3];
      GGEMSSphere3       bounding_sphere_;
    };

  private:
    GGuchar            header_[80]; /*!< Header infos */
    GGuint             number_of_triangles_; /*< Number of triangles in mesh file */
    GGEMSMeshTriangle* triangles_; /*!< Triangle from STL file */
};

#endif // End of GUARD_GGEMS_IO_GGEMSSTLREADER_HH
