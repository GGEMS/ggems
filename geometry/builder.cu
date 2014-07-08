// This file is part of GGEMS
//
// GGEMS is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// GGEMS is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GGEMS.  If not, see <http://www.gnu.org/licenses/>.
//
// GGEMS Copyright (C) 2013-2014 Julien Bert

#ifndef BUILDER_CU
#define BUILDER_CU

#include "builder.h"

///////// BVH class ////////////////////////////////////////////////////

BVH::BVH() {}

// Return the current node id
unsigned int BVH::get_current_id() {
    return cur_node_id;
}

// Update the tree address
void BVH::update_address() {
    ptr_nodes[0] = 0;
    unsigned int i=1;
    while (i < ptr_nodes.size()) {
        ptr_nodes[i] = ptr_nodes[i-1] + size_of_nodes[i-1];
        ++i;
    }
}

// Add the root
void BVH::add_root() {
    ptr_nodes.push_back(0);
    size_of_nodes.push_back(0);
    mother_node.push_back(0);
    cur_node_id = 0;
}

// Add a node
void BVH::add_node(unsigned int mother_id) {
    // New node ID
    cur_node_id++;

    // Insert this object into the tree
    child_nodes.insert(child_nodes.begin() + ptr_nodes[mother_id] + size_of_nodes[mother_id],
                       cur_node_id);

    // Update the tree
    size_of_nodes[mother_id]++;
    size_of_nodes.push_back(0);
    ptr_nodes.push_back(cur_node_id);
    mother_node.push_back(mother_id);

    // Update tree address
    update_address();
}

// Print the BVH
void BVH::print() {
    // print each node
    unsigned int i = 0;
    unsigned int j = 0;
    while (i < size_of_nodes.size()) {
        printf("(%i)--[%i]--(", mother_node[i], i);
        j=0; while (j < size_of_nodes[i]) {
            printf("%i,", child_nodes[ptr_nodes[i]+j]);
            ++j;
        }
        printf(")\n");
        ++i;
    }
    printf("\n");
}

///////// BVH class ////////////////////////////////////////////////////

GeometryBuilder::GeometryBuilder() {}

// Print the current world
void GeometryBuilder::print() {
    // Print out the tree structure
    World.tree.print();

    // Print out every object name
    unsigned int i;
    printf("List of object:\n");
    i=0; while (i < World.name_objects.size()) {
        printf("%i - %s\n", i, World.name_objects[i].c_str());
        ++i;
    }
    printf("\n");

    // Print out every material name
    printf("List of material:\n");
    i=0; while (i < World.materials_list.size()) {
        printf("%i - %s\n", i, World.materials_list[i].c_str());
        ++i;
    }
    printf("\n");

    // Print out each object contains on the tree
    i=0; while (i < World.ptr_objects.size()) {
        // Get obj address
        unsigned int address_obj = World.ptr_objects[i];

        // Object name
        printf("::: %s :::\n", World.name_objects[i].c_str());

        // Object type
        unsigned int params1 = (unsigned int)(World.data_objects[address_obj]);
        unsigned int params2 = (unsigned int)(World.data_objects[address_obj+1]);
        switch (params1) {
        case AABB:
            printf("type: AABB\n");
            printf("material: %s\n", World.materials_list[params2].c_str());
            printf("xmin: %f xmax: %f ymin: %f ymax: %f zmin: %f zmax: %f\n\n",
                    World.data_objects[address_obj+2], World.data_objects[address_obj+3],
                    World.data_objects[address_obj+4], World.data_objects[address_obj+5],
                    World.data_objects[address_obj+6], World.data_objects[address_obj+7]);
            break;

        case SPHERE:
            printf("type: SPHERE\n");
            printf("material: %s\n", World.materials_list[params2].c_str());
            printf("cx: %f cy: %f cz: %f radius: %f\n\n",
                    World.data_objects[address_obj+2], World.data_objects[address_obj+3],
                    World.data_objects[address_obj+4], World.data_objects[address_obj+5]);
            break;

        } // switch

        ++i;
    } // while
}

// Search and return the material index for a given material name
unsigned int GeometryBuilder::get_material_index(std::string material_name) {

    // Check if this material is already used, if it is return the corresponding index
    unsigned int index = 0;
    while (index < World.materials_list.size()) {
        if (World.materials_list[index] == material_name) return index;
        ++index;
    }

    // If it is not, add a new entry into the material table
    index = World.materials_list.size();
    World.materials_list.push_back(material_name);

    return index;
}

// Add the world
unsigned int GeometryBuilder::add_world(Aabb obj) {

    // Add the root tree
    World.tree.add_root();

    // Store the address to access to this object
    World.ptr_objects.push_back(World.data_objects.size());

    // Store the information of this object
    World.data_objects.push_back(AABB);                                  // Object Type
    World.data_objects.push_back(get_material_index(obj.material_name)); // Material index
    World.data_objects.push_back(obj.xmin);                              // AABB parameters
    World.data_objects.push_back(obj.xmax);
    World.data_objects.push_back(obj.ymin);
    World.data_objects.push_back(obj.ymax);
    World.data_objects.push_back(obj.zmin);
    World.data_objects.push_back(obj.zmax);

    World.name_objects.push_back(obj.object_name);                       // Name of this object

    // Store the size of this object
    World.size_of_objects.push_back(8);

    return World.tree.get_current_id();

}

// Add an AABB object into the world
unsigned int GeometryBuilder::add_object(Aabb obj, unsigned int mother_id) {

    // Add this object to the tree
    World.tree.add_node(mother_id);

    // Store the address to access to this object
    World.ptr_objects.push_back(World.data_objects.size());

    // Store the information of this object
    World.data_objects.push_back(AABB);                                  // Object Type
    World.data_objects.push_back(get_material_index(obj.material_name)); // Material index
    World.data_objects.push_back(obj.xmin);                              // AABB parameters
    World.data_objects.push_back(obj.xmax);
    World.data_objects.push_back(obj.ymin);
    World.data_objects.push_back(obj.ymax);
    World.data_objects.push_back(obj.zmin);
    World.data_objects.push_back(obj.zmax);

    World.name_objects.push_back(obj.object_name);                       // Name of this object

    // Store the size of this object
    World.size_of_objects.push_back(8);

    return World.tree.get_current_id();

}

// Add a Sphere object into the world
unsigned int GeometryBuilder::add_object(Sphere obj, unsigned int mother_id) {

    // Add this object to the tree
    World.tree.add_node(mother_id);

    // Store the address to access to this object
    World.ptr_objects.push_back(World.data_objects.size());

    // Store the information of this object
    World.data_objects.push_back(SPHERE);                                // Object Type
    World.data_objects.push_back(get_material_index(obj.material_name)); // Material index
    World.data_objects.push_back(obj.cx);                                // Sphere parameters
    World.data_objects.push_back(obj.cy);
    World.data_objects.push_back(obj.cz);
    World.data_objects.push_back(obj.radius);

    World.name_objects.push_back(obj.object_name);                       // Name of this object

    // Store the size of this object
    World.size_of_objects.push_back(6);

    return World.tree.get_current_id();

}




















#endif
