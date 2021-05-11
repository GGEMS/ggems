# ************************************************************************
# * This file is part of GGEMS.                                          *
# *                                                                      *
# * GGEMS is free software: you can redistribute it and/or modify        *
# * it under the terms of the GNU General Public License as published by *
# * the Free Software Foundation, either version 3 of the License, or    *
# * (at your option) any later version.                                  *
# *                                                                      *
# * GGEMS is distributed in the hope that it will be useful,             *
# * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
# * GNU General Public License for more details.                         *
# *                                                                      *
# * You should have received a copy of the GNU General Public License    *
# * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
# *                                                                      *
# ************************************************************************

#-------------------------------------------------------------------------------
# CMakeLists.txt
#
# CMakeLists.txt - Compile and build the 2_CT_scanner example
#
# Authors :
#   - Julien Bert <julien.bert@univ-brest.fr>
#   - Didier Benoit <didier.benoit@inserm.fr>
#
# Generated on : 14/12/2020
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Defining the project
PROJECT(CTScanner)

#-------------------------------------------------------------------------------
# Creating the executable
ADD_EXECUTABLE(ct_scanner ct_scanner.cc)
TARGET_LINK_LIBRARIES(ct_scanner ggems)

#-------------------------------------------------------------------------------
# Copy executable to ggems bin folder
INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ggems/examples)
INSTALL(TARGETS ct_scanner DESTINATION ggems/examples/2_CT_Scanner)
