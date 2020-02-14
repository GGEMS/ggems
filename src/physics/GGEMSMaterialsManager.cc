/*!
  \file GGEMSMaterialsManager.cc

  \brief GGEMS singleton class managing the material database

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Thrusday January 23, 2020
*/

#include "GGEMS/tools/GGEMSPrint.hh"
#include "GGEMS/io/GGEMSTextReader.hh"
#include "GGEMS/physics/GGEMSMaterialsManager.hh"
#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
#include "GGEMS/tools/GGEMSTools.hh"

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterialsManager::GGEMSMaterialsManager(void)
{
  GGcout("GGEMSMaterialsManager", "GGEMSMaterialsManager", 3)
    << "Allocation of GGEMS materials manager..." << GGendl;

  // Loading GGEMS chemical elements
  LoadChemicalElements();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterialsManager::~GGEMSMaterialsManager(void)
{
  GGcout("GGEMSMaterialsManager", "~GGEMSMaterialsManager", 3)
    << "Deallocation of GGEMS materials manager..." << GGendl;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterialsManager::SetMaterialsDatabase(char const* filename)
{
  // Converting char* to string
  std::string filename_str(filename);

  // Loading materials and elements in database
  if (!materials_.empty()) {
    GGwarn("GGEMSMaterialsManager", "SetMaterialsDatabase", 0)
      << "Material database if already loaded!!!" << GGendl;
  }
  else {
    // Materials
    LoadMaterialsDatabase(filename_str);
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterialsManager::LoadMaterialsDatabase(std::string const& filename)
{
  GGcout("GGEMSMaterialsManager", "LoadMaterialsDatabase", 0)
    << "Loading materials database in GGEMS..." << GGendl;

  // Opening the input file containing materials
  std::ifstream database_stream(filename, std::ios::in);
  GGEMSFileStream::CheckInputStream(database_stream, filename);

  // Reading database file
  std::string line;
  while (std::getline(database_stream, line)) {
    // Skip comment
    GGEMSTextReader::SkipComment(database_stream, line);
    // Check if blank line
    if (GGEMSTextReader::IsBlankLine(line)) continue;

    // Remove space/tab from line
    GGEMSTextReader::RemoveSpace(line);

    // Creating a material and filling infos
    GGEMSSingleMaterial material;
    std::string const kMaterialName =
      GGEMSMaterialReader::ReadMaterialName(line);
    material.density_ = GGEMSMaterialReader::ReadMaterialDensity(line);
    material.nb_elements_ =
      GGEMSMaterialReader::ReadMaterialNumberOfElements(line);

    // Loop over number of elements
    for (GGushort i = 0; i < material.nb_elements_; ++i) {
      // Get next line element by element
      std::getline(database_stream, line);
      // Remove space/tab from line
      GGEMSTextReader::RemoveSpace(line);

      // Get infos and store them
      material.mixture_Z_.push_back(
        GGEMSMaterialReader::ReadMaterialElementName(line));
      material.mixture_f_.push_back(
        GGEMSMaterialReader::ReadMaterialElementFraction(line));
    }

    // Storing the material
    GGcout("GGEMSMaterialsManager", "LoadMaterialsDatabase", 3)
      << "Adding material: " << kMaterialName << "..." << GGendl;
    materials_.insert(std::make_pair(kMaterialName, material));
  }

  // Closing file stream
  database_stream.close();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterialsManager::LoadChemicalElements(void)
{
  GGcout("GGEMSMaterialsManager", "LoadChemicalElements", 0)
    << "Loading chemical elements in GGEMS..." << GGendl;

  // Name, Z (atomic number), A (atomic mass), I (mean excitation energy)
  AddChemicalElements("Hydrogen",      1,   1.00794258759021,  19.2);
  AddChemicalElements("Helium",        2,   4.00256163944925,  41.8);
  AddChemicalElements("Lithium",       3,   6.94107031304227,  40.0);
  AddChemicalElements("Beryllium",     4,   9.01225666906993,  63.7);
  AddChemicalElements("Boron",         5,  10.81197967347820,  76.0);
  AddChemicalElements("Carbon",        6,  12.01105016615290,  81.0);
  AddChemicalElements("Nitrogen",      7,  14.00672322714900,  82.0);
  AddChemicalElements("Oxygen",        8,  15.99936002559900,  95.0);
  AddChemicalElements("Fluorine",      9,  18.99856455290040, 115.0);
  AddChemicalElements("Neon",         10,  20.17959842599130, 137.0);
  AddChemicalElements("Sodium",       11,  22.98994712312160, 149.0);
  AddChemicalElements("Magnesium",    12,  24.30478196585180, 156.0);
  AddChemicalElements("Aluminium",    13,  26.98159025341940, 166.0);
  AddChemicalElements("Silicon",      14,  28.08537955384370, 173.0);
  AddChemicalElements("Phosphor",     15,  30.97381680019820, 173.0);
  AddChemicalElements("Sulfur",       16,  32.06605607551560, 180.0);
  AddChemicalElements("Chlorine",     17,  35.45285812600360, 174.0);
  AddChemicalElements("Argon",        18,  39.94762422601480, 188.0);
  AddChemicalElements("Potassium",    19,  39.09867270295300, 190.0);
  AddChemicalElements("Calcium",      20,  40.07775083662300, 191.0);
  AddChemicalElements("Scandium",     21,  44.95632813837990, 216.0);
  AddChemicalElements("Titanium",     22,  47.88021241403330, 233.0);
  AddChemicalElements("Vandium",      23,  50.94130675526020, 245.0);
  AddChemicalElements("Chromium",     24,  51.99644690946120, 257.0);
  AddChemicalElements("Manganese",    25,  54.93781039862880, 272.0);
  AddChemicalElements("Iron",         26,  55.84672222699540, 286.0);
  AddChemicalElements("Cobalt",       27,  58.93266397468080, 297.0);
  AddChemicalElements("Nickel",       28,  58.69036639557310, 311.0);
  AddChemicalElements("Copper",       29,  63.54632307827150, 322.0);
  AddChemicalElements("Zinc",         30,  65.38939384031910, 330.0);
  AddChemicalElements("Gallium",      31,  69.72245962844680, 334.0);
  AddChemicalElements("Germanium",    32,  72.61010641918720, 350.0);
  AddChemicalElements("Arsenic",      33,  74.92167279662170, 347.0);
  AddChemicalElements("Selenium",     34,  78.95959126799810, 348.0);
  AddChemicalElements("Bromine",      35,  79.90320297696500, 343.0);
  AddChemicalElements("Krypton",      36,  83.80083335273170, 352.0);
  AddChemicalElements("Rubidium",     37,  85.46811115474350, 363.0);
  AddChemicalElements("Strontium",    38,  87.62018953630470, 366.0);
  AddChemicalElements("Yttrium",      39,  88.90509950532290, 379.0);
  AddChemicalElements("Zirconium",    40,  91.22422915526360, 393.0);
  AddChemicalElements("Niobium",      41,  92.90731928393380, 417.0);
  AddChemicalElements("Molybdenum",   42,  95.94079082623290, 424.0);
  AddChemicalElements("Technetium",   43,  97.90751155536330, 428.0);
  AddChemicalElements("Ruthenium",    44, 101.07042771167400, 441.0);
  AddChemicalElements("Rhodium",      45, 102.90653799538100, 449.0);
  AddChemicalElements("Palladium",    46, 106.41989589358000, 470.0);
  AddChemicalElements("Silver",       47, 107.86743780409400, 470.0);
  AddChemicalElements("Cadmium",      48, 112.41217798594800, 469.0);
  AddChemicalElements("Indium",       49, 114.81863342393900, 488.0);
  AddChemicalElements("Tin",          50, 118.70845204178500, 488.0);
  AddChemicalElements("Antimony",     51, 121.75034018477400, 487.0);
  AddChemicalElements("Tellurium",    52, 127.60109933254800, 485.0);
  AddChemicalElements("Iodine",       53, 126.90355329949200, 491.0);
  AddChemicalElements("Xenon",        54, 131.29102844638900, 482.0);
  AddChemicalElements("Caesium",      55, 132.90481598724100, 488.0);
  AddChemicalElements("Barium",       56, 137.32558424679400, 491.0);
  AddChemicalElements("Lanthanum",    57, 138.90581211161200, 501.0);
  AddChemicalElements("Cerium",       58, 140.11354028264300, 523.0);
  AddChemicalElements("Praseodymium", 59, 140.90898235055300, 535.0);
  AddChemicalElements("Neodymium",    60, 144.24117123831000, 546.0);
  AddChemicalElements("Promethium",   61, 144.91376443198600, 560.0);
  AddChemicalElements("Samarium",     62, 150.36135228209700, 574.0);
  AddChemicalElements("Europium",     63, 151.96468630146900, 580.0);
  AddChemicalElements("Gadolinium",   64, 157.25202093417500, 591.0);
  AddChemicalElements("Terbium",      65, 158.92420537897300, 614.0);
  AddChemicalElements("Dysprosium",   66, 162.50153884033000, 628.0);
  AddChemicalElements("Holmium",      67, 164.93119661275600, 650.0);
  AddChemicalElements("Erbium",       68, 167.26109949575700, 658.0);
  AddChemicalElements("Thulium",      69, 168.93546175692900, 674.0);
  AddChemicalElements("Ytterbium",    70, 173.04031839418600, 684.0);
  AddChemicalElements("Lutetium",     71, 174.96734764286900, 694.0);
  AddChemicalElements("Hafnium",      72, 178.49174475680500, 705.0);
  AddChemicalElements("Tantalum",     73, 180.94836774657300, 718.0);
  AddChemicalElements("Tungsten",     74, 183.85093167701900, 727.0);
  AddChemicalElements("Rhenium",      75, 186.20586920899700, 736.0);
  AddChemicalElements("Osmium",       76, 190.19970969518000, 746.0);
  AddChemicalElements("Iridium",      77, 192.22127914523900, 757.0);
  AddChemicalElements("Platinum",     78, 195.07803121248500, 790.0);
  AddChemicalElements("Gold",         79, 196.96818589807500, 790.0);
  AddChemicalElements("Mercury",      80, 200.59174564966600, 800.0);
  AddChemicalElements("Thallium",     81, 204.38545583003200, 810.0);
  AddChemicalElements("Lead",         82, 207.20151610865400, 823.0);
  AddChemicalElements("Bismuth",      83, 208.97852305058300, 823.0);
  AddChemicalElements("Polonium",     84, 208.98121656922500, 830.0);
  AddChemicalElements("Astatine",     85, 209.98542454112000, 825.0);
  AddChemicalElements("Radon",        86, 222.01569599339100, 794.0);
  AddChemicalElements("Francium",     87, 223.01973852858200, 827.0);
  AddChemicalElements("Radium",       88, 226.02352699440100, 826.0);
  AddChemicalElements("Actinium",     89, 227.02923320238800, 841.0);
  AddChemicalElements("Thorium",      90, 232.03650707711300, 847.0);
  AddChemicalElements("Protactinium", 91, 231.03483294404400, 878.0);
  AddChemicalElements("Uranium",      92, 238.02747665002200, 890.0);
  AddChemicalElements("Neptunium",    93, 237.00000000000000, 902.0);
  AddChemicalElements("Plutonium",    94, 244.00000000000000, 921.0);
  AddChemicalElements("Americium",    95, 243.00000000000000, 934.0);
  AddChemicalElements("Curium",       96, 247.00000000000000, 939.0);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterialsManager::AddChemicalElements(
  std::string const& element_name, GGushort const& element_Z,
  GGdouble const& element_A, GGdouble const& element_I)
{
  GGcout("GGEMSMaterialsManager", "AddChemicalElements", 3)
    << "Adding element: " << element_name << "..." << GGendl;

  // Creating chemical element and store it
  GGEMSChemicalElement element;
  element.atomic_number_Z_ = element_Z;
  element.atomic_mass_A_ = element_A * (GGEMSUnits::g / GGEMSUnits::mol);
  element.mean_excitation_energy_I_ = element_I * GGEMSUnits::eV;

  chemical_elements_.insert(std::make_pair(element_name, element));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterialsManager::PrintAvailableChemicalElements(void) const
{
  GGcout("GGEMSMaterialsManager", "PrintAvailableChemicalElements", 3)
    << "Printing available chemical elements..." << GGendl;

  GGcout("GGEMSMaterialsManager", "PrintAvailableChemicalElements", 0)
    << "Number of chemical elements in GGEMS: " << chemical_elements_.size()
    << GGendl;

  // Loop over the elements
  for (auto&& i : chemical_elements_) {
    GGcout("GGEMSMaterialsManager", "PrintAvailableChemicalElements", 0)
      << "    * Chemical element: \"" << i.first << "\"" << GGendl;
    GGcout("GGEMSMaterialsManager", "PrintAvailableChemicalElements", 0)
      << "        - Atomic number (Z): " << i.second.atomic_number_Z_
      << GGendl;
    GGcout("GGEMSMaterialsManager", "PrintAvailableChemicalElements", 0)
      << "        - Mass atomic (A): "
      << i.second.atomic_mass_A_ / (GGEMSUnits::g / GGEMSUnits::mol) << " g/mol"
      << GGendl;
    GGcout("GGEMSMaterialsManager", "PrintAvailableChemicalElements", 0)
      << "        - Mean excitation energy (I): "
      << i.second.mean_excitation_energy_I_/GGEMSUnits::eV << " eV" << GGendl;
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GGEMSMaterialsManager::PrintAvailableMaterials(void) const
{
  GGcout("GGEMSMaterialsManager", "PrintAvailableMaterials", 3)
    << "Printing available materials..." << GGendl;

  if (materials_.empty()) {
    GGcout("GGEMSMaterialsManager", "PrintAvailableMaterials", 0)
      << "For moment the GGEMS material database is empty, provide your "
      << " material file to GGEMS." << GGendl;
      return;
  }

  GGcout("GGEMSMaterialsManager", "PrintAvailableMaterials", 0)
    << "Number of materials in GGEMS: " << materials_.size() << GGendl;

  // Loop over the materials
  for (auto&& i : materials_) {
    GGcout("GGEMSMaterialsManager", "PrintAvailableMaterials", 0)
      << "    * Material: \"" << i.first << "\"" << GGendl;
    GGcout("GGEMSMaterialsManager", "PrintAvailableMaterials", 0)
      << "        - Density: "
      << i.second.density_ / (GGEMSUnits::g/GGEMSUnits::cm3) << " g/cm3"
      << GGendl;
    GGcout("GGEMSMaterialsManager", "PrintAvailableMaterials", 0)
      << "        - Number of elements: " << i.second.nb_elements_ << GGendl;
    for (GGushort j = 0; j < i.second.nb_elements_; ++j) {
      GGcout("GGEMSMaterialsManager", "PrintAvailableMaterials", 0)
        << "            * Element: " << i.second.mixture_Z_.at(j)
        << ", fraction: " << i.second.mixture_f_.at(j) << GGendl;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

GGEMSMaterialsManager* get_instance_materials_manager(void)
{
  return &GGEMSMaterialsManager::GetInstance();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void set_materials_ggems_materials_manager(
  GGEMSMaterialsManager* p_ggems_materials_manager, char const* filename)
{
  p_ggems_materials_manager->SetMaterialsDatabase(filename);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_available_chemical_elements_ggems_materials_manager(
  GGEMSMaterialsManager* p_ggems_materials_manager)
{
  p_ggems_materials_manager->PrintAvailableChemicalElements();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void print_available_materials_ggems_materials_manager(
  GGEMSMaterialsManager* p_ggems_materials_manager)
{
  p_ggems_materials_manager->PrintAvailableMaterials();
}
