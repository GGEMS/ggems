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
  \file GGEMSOctree.cc

  \brief GGEMS class for Octree

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Wednesday September 25, 2024
*/

#include <thread>

#include "GGEMS/geometries/GGEMSOctree.hh"
#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/tools/GGEMSTools.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOctree::GGEMSOctree(GGint const& max_depth, GGfloat const* half_width)
: max_depth_(max_depth),
  total_nodes_(static_cast<int>(std::pow(8.0f, static_cast<float>(max_depth_))-1)/7)
{
  GGcout("GGEMSOctree", "GGEMSOctree", 3) << "GGEMSOctree creating..." << GGendl;

  if (max_depth_ > pmax_) {
    GGEMSMisc::ThrowException("GGEMSOctree", "GGEMSOctree", "Max. depth too large in Octree!!!");
  }

  half_width_[0] = half_width[0];
  half_width_[1] = half_width[1];
  half_width_[2] = half_width[2];

  // Creating all nodes to GPU
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();
  number_activated_devices_ = opencl_manager.GetNumberOfActivatedDevice();

  nodes_ = new GGEMSNode*[number_activated_devices_];

  // Allocating memory for each engine
  for (GGsize i = 0; i < number_activated_devices_; ++i) {
    nodes_[i] = opencl_manager.SVMAllocate<GGEMSNode>(
      total_nodes_ * sizeof(GGEMSNode),
      i,
      CL_MEM_READ_WRITE,
      0,
      "GGEMSOctree"
    );

    // Mapping nodes
    opencl_manager.GetSVMData(
      nodes_[i],
      total_nodes_ * sizeof(GGEMSNode),
      i,
      CL_TRUE,
      CL_MAP_WRITE
    );

    for (int j = 0; j < total_nodes_; ++j) {
      nodes_[i][j].node_depth_ = 0;
      nodes_[i][j].center_.x_ = 0.0f;
      nodes_[i][j].center_.y_ = 0.0f;
      nodes_[i][j].center_.z_ = 0.0f;
      nodes_[i][j].first_child_node_id_ = 0;
      nodes_[i][j].half_width_[0] = 0.0f;
      nodes_[i][j].half_width_[1] = 0.0f;
      nodes_[i][j].half_width_[2] = 0.0f;
      nodes_[i][j].triangle_list_ = nullptr;
    }

    opencl_manager.ReleaseSVMData(nodes_[i], i);
  }

  GGcout("GGEMSOctree", "GGEMSOctree", 3) << "GGEMSOctree created!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSOctree::~GGEMSOctree(void)
{
  GGcout("GGEMSMeshedPhantom", "~GGEMSOctree", 3) << "GGEMSOctree erasing..." << GGendl;

  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  if (nodes_) {
    for (GGsize i = 0; i < number_activated_devices_; ++i) {
      opencl_manager.SVMDeallocate(nodes_[i], total_nodes_ * sizeof(GGEMSNode), i, "GGEMSOctree");
    }
    delete[] nodes_;
    nodes_ = nullptr;
  }

  GGcout("GGEMSMeshedPhantom", "~GGEMSOctree", 3) << "GGEMSOctree erased!!!" << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOctree::Build(GGEMSPoint3 const& center)
{
  GGcout("GGEMSOctree", "Build", 3) << "Building octree..." << GGendl;

  // Build octree for each device
  std::thread* thread_octree = new std::thread[number_activated_devices_];
  for (std::size_t i = 0; i < number_activated_devices_; ++i)
    thread_octree[i] =
      std::thread(&GGEMSOctree::BuildOctreeOnDevice, this, center, i);

  // Joining threads
  for (std::size_t i = 0; i < number_activated_devices_; ++i) thread_octree[i].join();

  delete[] thread_octree;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOctree::BuildOctreeOnDevice(GGEMSPoint3 const& center, GGsize const& thread_index)
{
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Mapping first nodes
  opencl_manager.GetSVMData(
    nodes_[thread_index],
    total_nodes_ * sizeof(GGEMSNode),
    thread_index,
    CL_TRUE,
    CL_MAP_WRITE
  );

  // Setting the first node
  nodes_[thread_index][0].node_depth_ = 1;
  nodes_[thread_index][0].center_.x_ = center.x_;
  nodes_[thread_index][0].center_.y_ = center.y_;
  nodes_[thread_index][0].center_.z_ = center.z_;
  nodes_[thread_index][0].first_child_node_id_ = 1;
  nodes_[thread_index][0].half_width_[0] = half_width_[0];
  nodes_[thread_index][0].half_width_[1] = half_width_[1];
  nodes_[thread_index][0].half_width_[2] = half_width_[2];
  nodes_[thread_index][0].triangle_list_ = nullptr;

  // Loop over octree level
  GGint current_node_id = 1;
  for (GGint i = 2; i <= max_depth_; ++i) {
    GGint total_nodes_through_level = (static_cast<GGint>(std::pow(8, i)) - 1) / 7;
    GGint total_nodes_in_level = total_nodes_through_level - (static_cast<GGint>(std::pow(8.0, i - 1)) - 1) / 7;

    // Loop over nodes in level
    GGint node_cluster = 0;
    for (GGint j = 0; j < total_nodes_in_level; ++j) {
      GGint mother_node_id = (((current_node_id + 1) + 6) >> 3) - 1;

      GGfloat mother_node_center[3] = { // Center of mother node
        nodes_[thread_index][mother_node_id].center_.x_,
        nodes_[thread_index][mother_node_id].center_.y_,
        nodes_[thread_index][mother_node_id].center_.z_
      };

      GGfloat step[3] = {
        static_cast<GGfloat>(half_width_[0] / std::pow(2.0f, i - 1)),
        static_cast<GGfloat>(half_width_[1] / std::pow(2.0f, i - 1)),
        static_cast<GGfloat>(half_width_[2] / std::pow(2.0f, i - 1))
      };

      // Compute offset by cluster of 8
      GGfloat offset_x = ((node_cluster % 8) & 1) ? step[0] : -step[0];
      GGfloat offset_y = ((node_cluster % 8) & 2) ? step[1] : -step[1];
      GGfloat offset_z = ((node_cluster % 8) & 4) ? step[2] : -step[2];

      nodes_[thread_index][current_node_id].node_depth_ = i;

      nodes_[thread_index][current_node_id].center_.x_ = mother_node_center[0] + offset_x;
      nodes_[thread_index][current_node_id].center_.y_ = mother_node_center[1] + offset_y;
      nodes_[thread_index][current_node_id].center_.z_ = mother_node_center[2] + offset_z;

      nodes_[thread_index][current_node_id].first_child_node_id_ = 8 * (current_node_id + 1)  - 7;

      nodes_[thread_index][current_node_id].half_width_[0] = step[0];
      nodes_[thread_index][current_node_id].half_width_[1] = step[1];
      nodes_[thread_index][current_node_id].half_width_[2] = step[2];

      nodes_[thread_index][current_node_id].triangle_list_ = nullptr;

      current_node_id++;
      node_cluster++;
    }
  }

  opencl_manager.ReleaseSVMData(nodes_[thread_index], thread_index);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOctree::InsertTriangles(GGEMSTriangle3** triangles, GGuint const& number_of_triangles)
{
  GGcout("GGEMSOctree", "InsertTriangles", 3) << "Inserting triangles..." << GGendl;

  // Build octree for each device
  std::thread* thread_octree = new std::thread[number_activated_devices_];
  for (std::size_t i = 0; i < number_activated_devices_; ++i) {
    thread_octree[i] = std::thread(&GGEMSOctree::InsertTrianglesOnDevice, this,
      &triangles[i][0], number_of_triangles, i);
  }

  // Joining threads
  for (std::size_t i = 0; i < number_activated_devices_; ++i) thread_octree[i].join();

  delete[] thread_octree;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSOctree::InsertTrianglesOnDevice(GGEMSTriangle3* triangle, GGuint number_of_triangles, GGsize const& thread_index)
{
  GGEMSOpenCLManager& opencl_manager = GGEMSOpenCLManager::GetInstance();

  // Mapping nodes and triangles
  opencl_manager.GetSVMData(
    nodes_[thread_index],
    total_nodes_ * sizeof(GGEMSNode),
    thread_index,
    CL_TRUE,
    CL_MAP_WRITE
  );

  // Mapping triangles
  opencl_manager.GetSVMData(
    triangle,
    sizeof(GGEMSTriangle3) * number_of_triangles,
    thread_index,
    CL_TRUE,
    CL_MAP_WRITE
  );

  // Loop over triangles
  for (GGuint t = 0; t < number_of_triangles; ++t) {
    GGfloat triangle_center[3] = {
      (triangle[t].bounding_sphere_).center_.x_,
      (triangle[t].bounding_sphere_).center_.y_,
      (triangle[t].bounding_sphere_).center_.z_
    };

    // Start from first node
    GGint current_node_id = 0;
    GGint next_node_id = 0;
    // Loop over depth
    for (GGint i = 0; i < max_depth_; ++i) {
      GGfloat node_center[3] = {
        nodes_[thread_index][current_node_id].center_.x_,
        nodes_[thread_index][current_node_id].center_.y_,
        nodes_[thread_index][current_node_id].center_.z_
      };

      GGint index = 0, straddle = 0;
      for (GGint j = 0; j < 3; ++j) {
        GGfloat delta = triangle_center[j] - node_center[j];

        if (fabsf(delta) <= (triangle[t].bounding_sphere_).radius_) {
          straddle = 1;
          break;
        }

        if (delta > 0.0f) index |= (1 << j); // ZYX
      }

      if (!straddle && i != max_depth_ - 1) {
        // Compute index of next node
        next_node_id =
          nodes_[thread_index][current_node_id].first_child_node_id_ + index;
        
      } else { // Storing triangle
        // Straddling, or no child node to descend into, so link object into
        // linked list at this node
        triangle[t].next_triangle_
          = nodes_[thread_index][current_node_id].triangle_list_;
        nodes_[thread_index][current_node_id].triangle_list_ = &triangle[t];
        break;
      }

      current_node_id = next_node_id;
    }
  }

  opencl_manager.ReleaseSVMData(nodes_[thread_index], thread_index);
  opencl_manager.ReleaseSVMData(triangle, thread_index);
}
