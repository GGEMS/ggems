# GGEMS

**GPU Geant4-based Monte Carlo Simulations**

*OpenCL-accelerated Monte Carlo particle transport for medical imaging*

GGEMS is a standalone Monte Carlo simulation engine developed in C++ with a Python interface. Its physics models are derived from Geant4 and implemented using OpenCL to enable high-performance simulations on compatible CPU and GPU architectures.

GGEMS is primarily designed for medical imaging applications, particularly computed tomography (CT) and cone-beam computed tomography (CBCT). It also provides capabilities for photon transport, dosimetry, and other specialized imaging configurations.

The project prioritizes computational performance and specialized simulation workflows rather than the general-purpose flexibility of frameworks such as Geant4 or GATE.

## Features & Limitations

GGEMS provides a set of specialized tools for Monte Carlo simulations in medical imaging, with an emphasis on photon transport and computational performance.

### Supported applications

- **CT / CBCT imaging** — Simulation of X-ray imaging systems using flat or curved detectors, including primary and scattered radiation.
- **SPECT imaging** — SPECT configurations are possible using user-defined meshed collimator geometries. No dedicated SPECT example or ready-to-use simulation workflow is currently provided.
- **Photon dosimetry** — Energy deposition and dose calculations in voxelized geometries.

### Core capabilities

- Photon transport using physics models derived from Geant4, including Compton scattering, photoelectric absorption, and Rayleigh scattering.
- X-ray sources and voxelized radioactive source distributions.
- Geometric navigation through simple solids, voxelized volumes, and meshed geometries.
- Configurable CT/CBCT detector geometries.
- OpenCL execution on compatible CPU and GPU devices, including multi-device configurations.
- Single- or double-precision calculations for dosimetry.
- Optional 3D visualization using OpenGL.

### Known limitations

**Particle transport**

GGEMS 1.3 focuses on photon transport. It does not provide the general-purpose particle transport capabilities available in frameworks such as Geant4 or GATE.

**Detector electronics and digitization**

GGEMS records detector hits but does not simulate detector readout electronics or digitization processes.

Consequently, GGEMS does not directly generate *singles*. Additional processing is required to model the electronic detector response and convert hits into singles.

**Specialized simulation workflows**

GGEMS is designed for a defined set of imaging and dosimetry applications. More complex configurations, such as SPECT systems with custom collimators, require users to construct and configure the appropriate geometries themselves.

## Requirements & Supported Platforms

GGEMS uses OpenCL to execute Monte Carlo simulations on compatible CPU and GPU devices.

The project integrates the official Khronos OpenCL SDK through Git submodules. **The NVIDIA CUDA Toolkit is no longer required to build GGEMS.**

### Build requirements

- **Operating system:** Windows or Linux.
- **C++ compiler:** A compiler supporting C++17 (MSVC, GCC, or Clang).
- **Build system:** CMake.
- **Version control:** Git, including Git submodule support.
- **Python:** Python 3 with setuptools.
- **OpenCL:** A compatible device driver and runtime supporting the compilation and execution of OpenCL C 2.0 kernels.

The OpenCL headers and ICD Loader are provided through the Khronos SDK submodules. A working OpenCL driver from the hardware vendor is still required.

### Validated GPU platforms

The current GGEMS 1.3.1 development version has been successfully tested on the following devices:

| Operating system | GPU | Architecture |
|---|---|---|
| Windows 11 | NVIDIA RTX 4000 SFF Ada Generation | NVIDIA Ada |
| Windows 11 | AMD Radeon AI PRO R9700 | AMD RDNA4 |

GGEMS is not restricted to these devices. Other OpenCL-compatible CPUs and GPUs may be used, provided that their drivers support the required OpenCL features.

However, compatibility and performance are not guaranteed on untested hardware.

### Linux support

GGEMS has also been used on Linux, including Ubuntu 24.04 LTS.

The latest Khronos OpenCL SDK integration has not yet been revalidated on Linux.

### Optional OpenGL visualization

GGEMS can be built **with or without OpenGL visualization**.

When enabled, visualization requires the following additional libraries:

- OpenGL
- GLFW
- GLEW
- GLM

These dependencies are not required for simulations without visualization.

Both installation configurations are described in the next section.

## Installation

GGEMS can be built and installed on Windows and Linux using CMake and Python.

### 1. Install an OpenCL runtime

**A working OpenCL implementation from your hardware vendor is required before running GGEMS.**

Depending on the devices you intend to use, install the appropriate driver or runtime:

| Hardware | OpenCL implementation |
|---|---|
| NVIDIA GPU | NVIDIA driver with OpenCL support |
| AMD GPU | AMD driver with OpenCL support |
| Intel CPU | Intel OpenCL CPU Runtime |
| Intel GPU | Intel graphics driver with OpenCL support |

Multiple OpenCL implementations can coexist on the same machine, allowing GGEMS to discover devices from different vendors.

GGEMS requires support for compiling and executing OpenCL C 2.0 kernels on the selected device.

**Note:** The Khronos OpenCL SDK included with GGEMS provides the development headers and ICD Loader, but does not replace the hardware vendor's OpenCL implementation.

### 2. Get GGEMS

Clone the repository with all its Git submodules:

```bash
git clone --recurse-submodules https://github.com/GGEMS/ggems.git
cd ggems
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

**The CUDA Toolkit is not required to build GGEMS.**

### 3. Install GGEMS

Choose the configuration that matches your needs.

#### Option A — Without OpenGL (recommended)

For simulations without graphical visualization:

```bash
python -m pip install .
```

This is the default configuration.

No OpenGL, GLFW, GLEW or GLM dependencies are required.

#### Option B — With OpenGL visualization

GGEMS supports optional OpenGL visualization of simulation geometries and particle trajectories.

Install the additional dependencies:

- OpenGL
- GLFW
- GLEW
- GLM

Then use the existing GGEMS build interface:

```bash
python setup.py build_ext --opengl=ON install
```

OpenGL visualization is optional. Particle transport is performed using OpenCL in both configurations.

### 4. Advanced CMake build

GGEMS can also be configured and compiled directly with CMake.

Without visualization:

```bash
cmake -S . -B build -DOPENGL_VISUALIZATION=OFF
cmake --build build --config Release --parallel
```

With visualization, configure CMake using `-DOPENGL_VISUALIZATION=ON` and ensure the required OpenGL dependencies are available.

### 5. Further documentation

For advanced build options and platform-specific installation instructions, see the [GGEMS installation documentation](https://doc.ggems.fr/v1.3/building_and_installing.html).

After installation, follow the Quick Start section to verify device detection and run your first simulation.

## Quick Start

Once GGEMS is installed, you can verify your OpenCL configuration and run your first Monte Carlo simulation using the examples provided with the repository.

### 1. Check available OpenCL devices

GGEMS can list the OpenCL devices detected on your system.

Run the following command:

```bash
python -c "from ggems import GGEMSOpenCLManager; GGEMSOpenCLManager().print_infos()"
```

This displays information about the available OpenCL platforms and devices, including their identifiers and hardware capabilities.

Make sure that your intended CPU or GPU appears in the list.

### 2. Run your first calculation

GGEMS provides a simple example for calculating photon interaction cross sections.

From the repository root:

```bash
cd examples/0_Cross_Sections
```

Run a Compton scattering calculation in water at 100 keV:

```bash
python cross_sections.py --device 0 --material Water --process Compton --energy 0.1
```

The energy is specified in MeV (0.1 MeV = 100 keV).

Replace device `0` with the appropriate OpenCL device identifier for your system.

The example prints material properties, photon interaction cross sections and attenuation coefficients.

### 3. Run a CT simulation

To explore a complete Monte Carlo imaging simulation, GGEMS provides a CT scanner example.

From the repository root:

```bash
cd examples/2_CT_Scanner
```

Run the simulation with 100,000 primary particles:

```bash
python ct_scanner.py --device 0 --nparticles 100000
```

The example generates a voxelized phantom, configures an X-ray source and CT detector, and performs photon transport through the geometry.

Simulation outputs are written to the example's `data` directory.

**Note:** Run the examples from their respective directories because they use relative paths to access input data and write results.

### 4. Explore more examples

Additional examples are available in the [`examples/`](https://github.com/GGEMS/ggems/tree/master/examples) directory, covering:

- Photon cross sections and attenuation
- CT imaging
- Voxelized phantom generation
- Photon dosimetry
- Particle tracking
- OpenGL visualization
- Meshed geometries
- Voxelized sources

For more advanced configurations and API details, refer to the documentation.

---

## Documentation & Community

### Documentation

The GGEMS documentation provides detailed information about installation, simulation configuration, geometry, sources, detectors and the Python interface.

**Official resources:**

- [GGEMS website](https://ggems.fr)
- [GGEMS documentation (Sphinx)](https://doc.ggems.fr/v1.3/)
- [Installation guide](https://doc.ggems.fr/v1.3/building_and_installing.html)
- [GGEMS community forum](https://forum.ggems.fr)

### Issues & Contributions

Bug reports, feature requests and technical discussions are welcome through [GitHub Issues](https://github.com/GGEMS/ggems/issues).

Contributions can be submitted through GitHub Pull Requests. Please review the [contribution guidelines](https://github.com/GGEMS/ggems/blob/master/CONTRIBUTING.md) before contributing.

### License

GGEMS is free and open-source software distributed under the terms of the **GNU General Public License, version 3 or later (GPL-3.0-or-later)**.

See the [LICENSE.md](https://github.com/GGEMS/ggems/blob/master/LICENSE.md) file for the full license text.
