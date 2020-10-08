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
  \file GGEMSMaterialsDatabaseManager.cc

  \brief GGEMS singleton class managing the material database

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday January 23, 2020
*/

#include "GGEMS/materials/GGEMSMaterialsDatabaseManager.hh"

#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/io/GGEMSTextReader.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterialsDatabaseManager::GGEMSMaterialsDatabaseManager(void)
{
  GGcout("GGEMSMaterialsDatabaseManager", "GGEMSMaterialsDatabaseManager", 3) << "Allocation of GGEMS materials manager..." << GGendl;

  // Loading GGEMS chemical elements
  LoadChemicalElements();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterialsDatabaseManager::~GGEMSMaterialsDatabaseManager(void)
{
  GGcout("GGEMSMaterialsDatabaseManager", "~GGEMSMaterialsDatabaseManager", 3) << "Deallocation of GGEMS materials manager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterialsDatabaseManager::SetMaterialsDatabase(char const* filename)
{
  // Converting char* to string
  std::string filename_str(filename);

  // Loading materials and elements in database
  if (!materials_.empty()) {
    GGwarn("GGEMSMaterialsDatabaseManager", "SetMaterialsDatabase", 0) << "Material database if already loaded!!!" << GGendl;
  }
  else {
    // Materials
    LoadMaterialsDatabase(filename_str);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterialsDatabaseManager::LoadMaterialsDatabase(std::string const& filename)
{
  GGcout("GGEMSMaterialsDatabaseManager", "LoadMaterialsDatabase", 0) << "Loading materials database in GGEMS..." << GGendl;

  // Opening the input file containing materials
  std::ifstream database_stream(filename, std::ios::in);
  GGEMSFileStream::CheckInputStream(database_stream, filename);

  // Reading database file
  std::string line("");
  while (std::getline(database_stream, line)) {
    // Skip comment
    GGEMSTextReader::SkipComment(database_stream, line);
    // Check if blank line
    if (GGEMSTextReader::IsBlankLine(line)) continue;

    // Remove space/tab from line
    GGEMSTextReader::RemoveSpace(line);

    // Creating a material and filling infos
    GGEMSSingleMaterial material;
    std::string const kMaterialName = GGEMSMaterialReader::ReadMaterialName(line);
    material.density_ = GGEMSMaterialReader::ReadMaterialDensity(line);
    material.nb_elements_ = static_cast<GGuchar>(GGEMSMaterialReader::ReadMaterialNumberOfElements(line));

    // Loop over number of elements
    for (GGushort i = 0; i < material.nb_elements_; ++i) {
      // Get next line element by element
      std::getline(database_stream, line);
      // Remove space/tab from line
      GGEMSTextReader::RemoveSpace(line);

      // Get infos and store them
      material.chemical_element_name_.push_back(GGEMSMaterialReader::ReadMaterialElementName(line));
      material.mixture_f_.push_back(GGEMSMaterialReader::ReadMaterialElementFraction(line));
    }

    // Storing the material
    GGcout("GGEMSMaterialsDatabaseManager", "LoadMaterialsDatabase", 3) << "Adding material: " << kMaterialName << "..." << GGendl;
    materials_.insert(std::make_pair(kMaterialName, material));
  }

  // Closing file stream
  database_stream.close();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterialsDatabaseManager::LoadChemicalElements(void)
{
  GGcout("GGEMSMaterialsDatabaseManager", "LoadChemicalElements", 0) << "Loading chemical elements in GGEMS..." << GGendl;

  // Name, Z (atomic number), M (molar mass g/mol), I (mean excitation energy eV), state, index for density correction
  AddChemicalElements("Hydrogen",        1,   1.00794258759021f,   19.2f,   GAS,  1);
  AddChemicalElements("Helium",          2,   4.00256163944925f,   41.8f,   GAS,  2);
  AddChemicalElements("Lithium",         3,   6.94107031304227f,   40.0f, SOLID,  3);
  AddChemicalElements("Beryllium",       4,   9.01225666906993f,   63.7f, SOLID,  4);
  AddChemicalElements("Boron",           5,  10.81197967347820f,   76.0f, SOLID,  5);
  AddChemicalElements("Carbon",          6,  12.01105016615290f,   81.0f, SOLID,  6);
  AddChemicalElements("Nitrogen",        7,  14.00672322714900f,   82.0f,   GAS,  7);
  AddChemicalElements("Oxygen",          8,  15.99936002559900f,   95.0f,   GAS,  8);
  AddChemicalElements("Fluorine",        9,  18.99856455290040f,  115.0f,   GAS,  9);
  AddChemicalElements("Neon",           10,  20.17959842599130f,  137.0f,   GAS, 10);
  AddChemicalElements("Sodium",         11,  22.98994712312160f,  149.0f, SOLID, 11);
  AddChemicalElements("Magnesium",      12,  24.30478196585180f,  156.0f, SOLID, 12);
  AddChemicalElements("Aluminium",      13,  26.98159025341940f,  166.0f, SOLID, 13);
  AddChemicalElements("Silicon",        14,  28.08537955384370f,  173.0f, SOLID, 14);
  AddChemicalElements("Phosphor",       15,  30.97381680019820f,  173.0f, SOLID, 15);
  AddChemicalElements("Sulfur",         16,  32.06605607551560f,  180.0f, SOLID, 16);
  AddChemicalElements("Chlorine",       17,  35.45285812600360f,  174.0f,   GAS, 17);
  AddChemicalElements("Argon",          18,  39.94762422601480f,  188.0f,   GAS, 18);
  AddChemicalElements("Potassium",      19,  39.09867270295300f,  190.0f, SOLID, 19);
  AddChemicalElements("Calcium",        20,  40.07775083662300f,  191.0f, SOLID, 20);
  AddChemicalElements("Scandium",       21,  44.95632813837990f,  216.0f, SOLID, 21);
  AddChemicalElements("Titanium",       22,  47.88021241403330f,  233.0f, SOLID, 22);
  AddChemicalElements("Vandium",        23,  50.94130675526020f,  245.0f, SOLID, 23);
  AddChemicalElements("Chromium",       24,  51.99644690946120f,  257.0f, SOLID, 24);
  AddChemicalElements("Manganese",      25,  54.93781039862880f,  272.0f, SOLID, 25);
  AddChemicalElements("Iron",           26,  55.84672222699540f,  286.0f, SOLID, 26);
  AddChemicalElements("Cobalt",         27,  58.93266397468080f,  297.0f, SOLID, 27);
  AddChemicalElements("Nickel",         28,  58.69036639557310f,  311.0f, SOLID, 28);
  AddChemicalElements("Copper",         29,  63.54632307827150f,  322.0f, SOLID, 29);
  AddChemicalElements("Zinc",           30,  65.38939384031910f,  330.0f, SOLID, 30);
  AddChemicalElements("Gallium",        31,  69.72245962844680f,  334.0f, SOLID, 31);
  AddChemicalElements("Germanium",      32,  72.61010641918720f,  350.0f, SOLID, 32);
  AddChemicalElements("Arsenic",        33,  74.92167279662170f,  347.0f, SOLID, 33);
  AddChemicalElements("Selenium",       34,  78.95959126799810f,  348.0f, SOLID, 34);
  AddChemicalElements("Bromine",        35,  79.90320297696500f,  343.0f,   GAS, 35);
  AddChemicalElements("Krypton",        36,  83.80083335273170f,  352.0f,   GAS, 36);
  AddChemicalElements("Rubidium",       37,  85.46811115474350f,  363.0f, SOLID, 37);
  AddChemicalElements("Strontium",      38,  87.62018953630470f,  366.0f, SOLID, 38);
  AddChemicalElements("Yttrium",        39,  88.90509950532290f,  379.0f, SOLID, 39);
  AddChemicalElements("Zirconium",      40,  91.22422915526360f,  393.0f, SOLID, 40);
  AddChemicalElements("Niobium",        41,  92.90731928393380f,  417.0f, SOLID, 41);
  AddChemicalElements("Molybdenum",     42,  95.94079082623290f,  424.0f, SOLID, 42);
  AddChemicalElements("Technetium",     43,  97.90751155536330f,  428.0f, SOLID, 43);
  AddChemicalElements("Ruthenium",      44, 101.07042771167400f,  441.0f, SOLID, 44);
  AddChemicalElements("Rhodium",        45, 102.90653799538100f,  449.0f, SOLID, 45);
  AddChemicalElements("Palladium",      46, 106.41989589358000f,  470.0f, SOLID, 46);
  AddChemicalElements("Silver",         47, 107.86743780409400f,  470.0f, SOLID, 47);
  AddChemicalElements("Cadmium",        48, 112.41217798594800f,  469.0f, SOLID, 48);
  AddChemicalElements("Indium",         49, 114.81863342393900f,  488.0f, SOLID, 49);
  AddChemicalElements("Tin",            50, 118.70845204178500f,  488.0f, SOLID, 50);
  AddChemicalElements("Antimony",       51, 121.75034018477400f,  487.0f, SOLID, 51);
  AddChemicalElements("Tellurium",      52, 127.60109933254800f,  485.0f, SOLID, 52);
  AddChemicalElements("Iodine",         53, 126.90355329949200f,  491.0f, SOLID, 53);
  AddChemicalElements("Xenon",          54, 131.29102844638900f,  482.0f,   GAS, 54);
  AddChemicalElements("Caesium",        55, 132.90481598724100f,  488.0f, SOLID, 55);
  AddChemicalElements("Barium",         56, 137.32558424679400f,  491.0f, SOLID, 56);
  AddChemicalElements("Lanthanum",      57, 138.90581211161200f,  501.0f, SOLID, 57);
  AddChemicalElements("Cerium",         58, 140.11354028264300f,  523.0f, SOLID, 58);
  AddChemicalElements("Praseodymium",   59, 140.90898235055300f,  535.0f, SOLID, 59);
  AddChemicalElements("Neodymium",      60, 144.24117123831000f,  546.0f, SOLID, 60);
  AddChemicalElements("Promethium",     61, 144.91376443198600f,  560.0f, SOLID, 61);
  AddChemicalElements("Samarium",       62, 150.36135228209700f,  574.0f, SOLID, 62);
  AddChemicalElements("Europium",       63, 151.96468630146900f,  580.0f, SOLID, 63);
  AddChemicalElements("Gadolinium",     64, 157.25202093417500f,  591.0f, SOLID, 64);
  AddChemicalElements("Terbium",        65, 158.92420537897300f,  614.0f, SOLID, 65);
  AddChemicalElements("Dysprosium",     66, 162.50153884033000f,  628.0f, SOLID, 66);
  AddChemicalElements("Holmium",        67, 164.93119661275600f,  650.0f, SOLID, 67);
  AddChemicalElements("Erbium",         68, 167.26109949575700f,  658.0f, SOLID, 68);
  AddChemicalElements("Thulium",        69, 168.93546175692900f,  674.0f, SOLID, 69);
  AddChemicalElements("Ytterbium",      70, 173.04031839418600f,  684.0f, SOLID, 70);
  AddChemicalElements("Lutetium",       71, 174.96734764286900f,  694.0f, SOLID, 71);
  AddChemicalElements("Hafnium",        72, 178.49174475680500f,  705.0f, SOLID, 72);
  AddChemicalElements("Tantalum",       73, 180.94836774657300f,  718.0f, SOLID, 73);
  AddChemicalElements("Tungsten",       74, 183.85093167701900f,  727.0f, SOLID, 74);
  AddChemicalElements("Rhenium",        75, 186.20586920899700f,  736.0f, SOLID, 75);
  AddChemicalElements("Osmium",         76, 190.19970969518000f,  746.0f, SOLID, 76);
  AddChemicalElements("Iridium",        77, 192.22127914523900f,  757.0f, SOLID, 77);
  AddChemicalElements("Platinum",       78, 195.07803121248500f,  790.0f, SOLID, 78);
  AddChemicalElements("Gold",           79, 196.96818589807500f,  790.0f, SOLID, 79);
  AddChemicalElements("Mercury",        80, 200.59174564966600f,  800.0f, SOLID, 80);
  AddChemicalElements("Thallium",       81, 204.38545583003200f,  810.0f, SOLID, 81);
  AddChemicalElements("Lead",           82, 207.20151610865400f,  823.0f, SOLID, 82);
  AddChemicalElements("Bismuth",        83, 208.97852305058300f,  823.0f, SOLID, 83);
  AddChemicalElements("Polonium",       84, 208.98121656922500f,  830.0f, SOLID, 84);
  AddChemicalElements("Astatine",       85, 209.98542454112000f,  825.0f, SOLID, -1);
  AddChemicalElements("Radon",          86, 222.01569599339100f,  794.0f,   GAS, 85);
  AddChemicalElements("Francium",       87, 223.01973852858200f,  827.0f, SOLID, -1);
  AddChemicalElements("Radium",         88, 226.02352699440100f,  826.0f, SOLID, 86);
  AddChemicalElements("Actinium",       89, 227.02923320238800f,  841.0f, SOLID, 87);
  AddChemicalElements("Thorium",        90, 232.03650707711300f,  847.0f, SOLID, 88);
  AddChemicalElements("Protactinium",   91, 231.03483294404400f,  878.0f, SOLID, 89);
  AddChemicalElements("Uranium",        92, 238.02747665002200f,  890.0f, SOLID, 90);
  AddChemicalElements("Neptunium",      93, 237.00000000000000f,  902.0f, SOLID, 91);
  AddChemicalElements("Plutonium",      94, 244.00000000000000f,  921.0f, SOLID, 92);
  AddChemicalElements("Americium",      95, 243.00000000000000f,  934.0f, SOLID, 93);
  AddChemicalElements("Curium",         96, 247.00000000000000f,  939.0f, SOLID, 94);
  AddChemicalElements("Berkelium",      97, 247.00000000000000f,  952.0f, SOLID, 95);
  AddChemicalElements("Californium",    98, 251.00000000000000f,  966.0f, SOLID, -1);
  AddChemicalElements("Einsteinium",    99, 252.00000000000000f,  980.0f, SOLID, -1);
  AddChemicalElements("Fermium",       100, 257.00000000000000f,  994.0f, SOLID, -1);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterialsDatabaseManager::AddChemicalElements(std::string const& element_name, GGuchar const& element_Z, GGfloat const& element_M, GGfloat const& element_I, GGuchar const& state, GGshort const& index_density_correction)
{
  GGcout("GGEMSMaterialsDatabaseManager", "AddChemicalElements", 3) << "Adding element: " << element_name << "..." << GGendl;

  // Creating chemical element and store it
  GGEMSChemicalElement element;
  element.atomic_number_Z_ = element_Z;
  element.molar_mass_M_ = element_M * (g / mol);
  element.mean_excitation_energy_I_ = element_I * eV;
  element.state_ = state;
  element.index_density_correction_ = index_density_correction;

  // No need to check if element already insert
  chemical_elements_.insert(std::make_pair(element_name, element));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGfloat GGEMSMaterialsDatabaseManager::GetRadiationLength(std::string const& material) const
{
  GGfloat inverse_radiation = 0.0f;
  GGfloat tsai_radiation = 0.0f;
  GGfloat zeff = 0.0f;
  GGfloat coulomb = 0.0f;

  static constexpr GGfloat l_rad_light[]  = {5.310f , 4.790f , 4.740f, 4.710f};
  static constexpr GGfloat lp_rad_light[] = {6.144f , 5.621f , 5.805f, 5.924f};
  static constexpr GGfloat k1 = 0.00830f;
  static constexpr GGfloat k2 = 0.20206f;
  static constexpr GGfloat k3 = 0.00200f;
  static constexpr GGfloat k4 = 0.03690f;

  // Getting the material infos from database
  GGEMSSingleMaterial const& kSingleMaterial = GetMaterial(material);

  // Loop over the chemical elements by material
  for (GGuchar i = 0; i < kSingleMaterial.nb_elements_; ++i) {
    // Getting the chemical element
    GGEMSChemicalElement const& kChemicalElement = GetChemicalElement(kSingleMaterial.chemical_element_name_[i]);

    // Z effective
    zeff = static_cast<GGfloat>(kChemicalElement.atomic_number_Z_);

    //  Compute Coulomb correction factor (Phys Rev. D50 3-1 (1994) page 1254)
    GGfloat az2 = (FINE_STRUCTURE_CONST*zeff)*(FINE_STRUCTURE_CONST*zeff);
    GGfloat az4 = az2 * az2;
    coulomb = ( k1*az4 + k2 + 1.0f/ (1.0f+az2) ) * az2 - ( k3*az4 + k4 ) * az4;

    //  Compute Tsai's Expression for the Radiation Length
    //  (Phys Rev. D50 3-1 (1994) page 1254)
    GGfloat const logZ3 = std::log(zeff) / 3.0f;

    GGfloat l_rad = 0.0f;
    GGfloat lp_rad = 0.0f;
    GGint iz = static_cast<GGint>(( zeff + 0.5f ) - 1);
    if (iz <= 3){
      l_rad = l_rad_light[iz];
      lp_rad = lp_rad_light[iz];
    }
    else {
      l_rad = std::log(184.15f) - logZ3;
      lp_rad = std::log(1194.0f) - 2.0f*logZ3;
    }

    tsai_radiation = 4.0f * ALPHA_RCL2 * zeff * ( zeff * ( l_rad - coulomb ) + lp_rad );
    inverse_radiation += GetAtomicNumberDensity(material, i) * tsai_radiation;
  }

  return (inverse_radiation <= 0.0f ? std::numeric_limits<GGfloat>::max() : 1.0f / inverse_radiation);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterialsDatabaseManager::PrintAvailableChemicalElements(void) const
{
  GGcout("GGEMSMaterialsDatabaseManager", "PrintAvailableChemicalElements", 3) << "Printing available chemical elements..." << GGendl;

  GGcout("GGEMSMaterialsDatabaseManager", "PrintAvailableChemicalElements", 0) << "Number of chemical elements in GGEMS: " << chemical_elements_.size() << GGendl;

  // Loop over the elements
  for (auto&& i : chemical_elements_) {
    GGcout("GGEMSMaterialsDatabaseManager", "PrintAvailableChemicalElements", 0) << "    * Chemical element: \"" << i.first << "\"" << GGendl;
    GGcout("GGEMSMaterialsDatabaseManager", "PrintAvailableChemicalElements", 0) << "        - Atomic number (Z): " << i.second.atomic_number_Z_ << GGendl;
    GGcout("GGEMSMaterialsDatabaseManager", "PrintAvailableChemicalElements", 0) << "        - Molar mass (M): " << i.second.molar_mass_M_ / (g / mol) << " g/mol" << GGendl;
    GGcout("GGEMSMaterialsDatabaseManager", "PrintAvailableChemicalElements", 0) << "        - Mean excitation energy (I): " << i.second.mean_excitation_energy_I_/eV << " eV" << GGendl;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterialsDatabaseManager::PrintAvailableMaterials(void) const
{
  GGcout("GGEMSMaterialsDatabaseManager", "PrintAvailableMaterials", 3) << "Printing available materials..." << GGendl;

  if (materials_.empty()) {
    GGcout("GGEMSMaterialsDatabaseManager", "PrintAvailableMaterials", 0) << "For moment the GGEMS material database is empty, provide your material file to GGEMS." << GGendl;
    return;
  }

  GGcout("GGEMSMaterialsDatabaseManager", "PrintAvailableMaterials", 0) << "Number of materials in GGEMS: " << materials_.size() << GGendl;

  // Loop over the materials
  for (auto&& i : materials_) {
    GGcout("GGEMSMaterialsDatabaseManager", "PrintAvailableMaterials", 0) << "    * Material: \"" << i.first << "\"" << GGendl;
    GGcout("GGEMSMaterialsDatabaseManager", "PrintAvailableMaterials", 0) << "        - Density: " << i.second.density_ / (g/cm3) << " g/cm3" << GGendl;
    GGcout("GGEMSMaterialsDatabaseManager", "PrintAvailableMaterials", 0) << "        - Number of elements: " << static_cast<GGushort>(i.second.nb_elements_) << GGendl;
    for (GGushort j = 0; j < i.second.nb_elements_; ++j) {
      GGcout("GGEMSMaterialsDatabaseManager", "PrintAvailableMaterials", 0) << "            * Element: " << i.second.chemical_element_name_.at(j) << ", fraction: " << i.second.mixture_f_.at(j) << GGendl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterialsDatabaseManager* get_instance_materials_manager(void)
{
  return &GGEMSMaterialsDatabaseManager::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_materials_ggems_materials_manager(GGEMSMaterialsDatabaseManager* ggems_materials_manager, char const* filename)
{
  ggems_materials_manager->SetMaterialsDatabase(filename);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_available_chemical_elements_ggems_materials_manager(GGEMSMaterialsDatabaseManager* ggems_materials_manager)
{
  ggems_materials_manager->PrintAvailableChemicalElements();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_available_materials_ggems_materials_manager(GGEMSMaterialsDatabaseManager* ggems_materials_manager)
{
  ggems_materials_manager->PrintAvailableMaterials();
}
