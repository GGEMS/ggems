#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSOCTREE_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSOCTREE_HH

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
  \file GGEMSOctree.hh

  \brief GGEMS class for Octree

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday September 25, 2024
*/

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/geometries/GGEMSPrimitiveGeometries.hh"

/*!
  \class GGEMSOctree
  \brief GGEMS class for Octree
*/
class GGEMS_EXPORT GGEMSOctree
{
  public:
    /*!
      \param max_depth - max depth of the octree
      \param half_width - half width of each side of the octree
      \brief GGEMSOctree constructor
    */
    GGEMSOctree(GGint const& max_depth, GGfloat const* half_width);

    /*!
      \brief GGEMSMOctree destructor
    */
    ~GGEMSOctree(void);

   /*!
      \fn GGEMSOctree(GGEMSOctree const& octree) = delete
      \param octree - reference on the GGEMS octree
      \brief Avoid copy by reference
    */ 
    GGEMSOctree(GGEMSOctree const& octree) = delete;

    /*!
      \fn GGEMSOctree(GGEMSOctree const&& octree) = delete
      \param octree - rvalue reference on the GGEMS octree
      \brief Avoid copy by rvalue reference
    */
    GGEMSOctree(GGEMSOctree const&& octree) = delete;

    /*!
      \fn GGEMSOctree& operator=(GGEMSOctree const& octree) = delete 
      \param octree - rvalue reference on the GGEMS octree
      \brief Avoid assignement by reference 
    */
    GGEMSOctree& operator=(GGEMSOctree const& octree) = delete;

    /*!
      \fn GGEMSOctree& operator=(GGEMSOctree const&& octree) = delete 
      \param octree - rvalue reference on the GGEMS octree
      \brief Avoid assignement by rvalue reference 
    */
    GGEMSOctree& operator=(GGEMSOctree const&& octree) = delete;

    /*!
      \fn void Build(GGEMSPoint3 const& center)
      \param center - center of node
      \brief Building octree
    */
    void Build(GGEMSPoint3 const& center);

    /*!
      \fn void InsertTriangles(GGEMSTriangles3** const& triangles, GGuint const& number_of_triangles)
      \param triangles - pointer to triangles
      \param number_of_triangles - number of the triangles
      \brief Insert triangles
    */
    void InsertTriangles(GGEMSTriangle3** triangles, GGuint const& number_of_triangles);

    /*!
      \fn GGEMSNode** GetNodes(void)
      \brief get the list of nodes
      \return the list of nodes
    */
    GGEMSNode** GetNodes(void) const {return nodes_;}

    /*!
      \fn GGint GetTotalNodes(void)
      \brief Get the total of nodes
      \return the total number of nodes
    */
    GGint GetTotalNodes(void) const {return total_nodes_;}

  private:
    /*!
      \fn void BuildOctreeOnDevice(GGEMSPoint3 const& center, GGsize const& thread_index)
      \param center - center of node
      \param thread_index - index of the thread
      \brief Building octree on OpenCL device
    */
    void BuildOctreeOnDevice(GGEMSPoint3 const& center, GGsize const& thread_index);

    /*!
      \fn void InsertTrianglesOnDevice(GGEMSTriangle3* triangle, GGuint number_of_triangles, GGsize const& thread_index)
      \param triangle - pointer to triangle
      \param number_of_triangles - number of the triangles
      \param thread_index - index of thread
      \brief Insert triangles to OpenCL device
    */
    void InsertTrianglesOnDevice(GGEMSTriangle3* triangle, GGuint number_of_triangles, GGsize const& thread_index);

  private:
    static constexpr GGint pmax_ = 8; /*!< Maximum number of levels */
    GGint                  max_depth_; /*!< Max depth defined by the user */
    GGfloat                half_width_[3]; /*!< half width of octree */
    GGint                  total_nodes_; /*!< Total nodes in the octree */
    GGEMSNode**            nodes_; /*!< Pointer of nodes */
    GGsize                 number_activated_devices_; /*!< Number of activated device */
};

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSVOXELIZEDSOLID_HH
