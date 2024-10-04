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
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \class GGEMSOctree
  \brief GGEMS class for Octree
*/
class GGEMS_EXPORT GGEMSOctree
{
  public:
    GGEMSOctree(GGint const& max_depth, GGfloat const* half_width);
    ~GGEMSOctree(void);

    GGEMSOctree(GGEMSOctree const& octree) = delete;
    GGEMSOctree(GGEMSOctree const&& octree) = delete;
    GGEMSOctree& operator=(GGEMSOctree const& octree) = delete;
    GGEMSOctree& operator=(GGEMSOctree const&& octree) = delete;

    void Build(GGEMSPoint3 const& center);

    void InsertTriangles(GGEMSTriangle3** triangles, GGuint const& number_of_triangles);

    GGEMSNode** GetNodes(void) const {return nodes_;}
    GGint GetTotalNodes(void) const {return total_nodes_;}

  private:
    void BuildOctreeOnDevice(GGEMSPoint3 const& center, GGsize const& thread_index);
    void InsertTrianglesOnDevice(GGEMSTriangle3* triangle, GGuint number_of_triangles, GGsize const& thread_index);

  private:
    static constexpr GGint pmax_ = 8; // maximum number of levels
    GGint                  max_depth_;
    GGfloat                half_width_[3];
    GGint                  total_nodes_;
    GGEMSNode**            nodes_;
    GGsize                 number_activated_devices_; /*!< Number of activated device */
};

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSVOXELIZEDSOLID_HH
