.. GGEMS documentation: Generality

.. sectionauthor:: Julien Bert

Generality
==========

System of units
---------------

Units recognized and used by GGEMS:

**Length:**

* millimeter or mm
* millimeter2 or mm2
* millimeter3 or mm3
* centimeter or cm
* centimeter2 or cm2
* centimeter3 or cm3
* meter or m
* meter2 or m2
* meter3 or m3
* kilometer or km
* kilometer2 or km2
* kilometer3 or km3
* parsec or pc
* micrometer or um
* nanometer or nm
* angstrom
* fermi
* barn
* millibarn
* microbarn
* nanobarn
* picobarn

**Angle**

* radian or rad
* milliradian or mrad
* degree or deg
* steradian or sr

**Time**

* nanosecond or ns
* second or s
* millisecond or ms
* microsecond
* picosecond
* hertz
* kilohertz
* megahertz

**Energy**

* megaelectronvolt or MeV
* electronvolt or eV
* kiloelectronvolt or keV
* gigaelectronvolt or GeV
* teraelectronvolt or TeV
* petaelectronvolt or PeV
* joule

**Mass**

* kilogram or kg
* gram or g
* milligram or mg

**Power**

* watt

**Force**

* newton

**Pressure**

* hep_pascal
* bar
* atmosphere

**Electric current**

* ampere
* milliampere
* microampere
* nanoampere

**Electric potential**

* megavolt
* kilovolt
* volt

**Electric resistance**

* ohm

**Electric capacitance**

* farad
* millifarad
* microfarad
* nanofarad
* picofarad

**Magnetic flux**

* weber

**Magnetic field**

* tesla
* gauss
* kilogauss

**Inductance**

* henry

**Temperature**

* kelvin

**Amount of substance**

* mole

**Activity**

* becquerel or Bq
* kilobecquerel or kBq
* megabecquerel or MBq
* gigabecquerel or GBq
* curie or Ci
* millicurie or mCi
* microcurie or uCi

**Absorbed dose**

* gray
* kilogray
* milligray
* microgray

**Luminous flux**

* candela

**Luminous flux**

* lumen

**Illuminance**

* lux

**Miscellaneous**

* perCent
* perThousand
* perMillion

Reference frame
---------------

MHD file format
---------------

GGEMS uses MHD file format to load phantom and CT image. This file format is composed of two files. One is containing the raw data of the image in binary format (.raw), and the second one containing information about the image in text format (.mhd). In the .mhd file only these keywords are used by GGEMS::

    ObjectType = Image
    NDims = 3
    BinaryData = True
    CompressedData = False
    Offset = 100.000000 100.000000 100.000000
    ElementSpacing = 4.000000 4.000000 4.000000
    DimSize = 128 63 46
    ElementType = MET_FLOAT
    ElementDataFile = phantom_pet_hu.raw

``TODO: talk about offset``

Simulation structure
--------------------


Last update: |today|  -  Release: |release|.