# Contributing to GGEMS

Thank you for your interest in contributing to **GGEMS (GPU Geant4-based Monte Carlo Simulations)**!

We welcome bug reports, reproducible test cases, documentation improvements, and code contributions. Our goal is to keep GGEMS reliable, efficient, and useful for scientific simulations.

## Before You Start

- Check the [existing issues](https://github.com/GGEMS/ggems/issues) and [pull requests](https://github.com/GGEMS/ggems/pulls) to avoid duplicate work.
- For substantial changes or new features, please open an issue first to discuss the proposed approach.
- Read our [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting Bugs

Please report bugs through [GitHub Issues](https://github.com/GGEMS/ggems/issues). A useful report should include:

- **GGEMS version or commit:** the release or Git commit used.
- **Environment:** operating system, compiler, CMake and Python versions, when relevant.
- **OpenCL configuration:** device model, hardware vendor, OpenCL driver/runtime, and device identifier.
- **Reproduction steps:** a minimal example, exact commands, and relevant simulation parameters (including the random seed, if applicable).
- **Expected and actual behavior:** a clear description of the difference.
- **Evidence:** relevant error messages, logs, or images.

Please remove passwords, personal information, and sensitive research data from reports and attachments.

## Building from Source

GGEMS uses CMake and the Khronos OpenCL SDK, included as Git submodules. Clone the repository with its submodules:

```bash
git clone --recurse-submodules https://github.com/GGEMS/ggems.git
cd ggems
```

For an existing clone, initialize or update the submodules with:

```bash
git submodule update --init --recursive
```

A compatible OpenCL driver and runtime from your hardware vendor are required to run GGEMS. The included Khronos SDK does not provide a device implementation.

To build GGEMS without optional OpenGL visualization:

```bash
cmake -S . -B build -DOPENGL_VISUALIZATION=OFF
cmake --build build --config Release --parallel
```

For Python installation, optional OpenGL dependencies, and platform-specific requirements, see the [README](README.md) and the [installation documentation](https://doc.ggems.fr/v1.3/building_and_installing.html).

## Submitting a Pull Request

1. Fork the repository and create a topic branch from `master`.
2. Keep your changes focused on one problem or feature.
3. Include a reproducible example or test when practical, especially for numerical or geometry-related fixes.
4. Test the affected functionality on the OpenCL hardware available to you, and describe the results and environment in the pull request.
5. Update documentation and examples when changing user-visible behavior or the Python API.
6. Avoid committing generated build files, simulation outputs, virtual environments, or unrelated formatting changes.
7. Open a pull request against `master`, describe what changed and why, and link any related issue.

A contribution does not need to support every GPU vendor, but please state clearly which devices and platforms you tested.

## Coding Style

- Follow the conventions already used in the surrounding code.
- Use **two spaces for indentation**, not tabs, where applicable.
- Write source comments, documentation, and pull request descriptions in **English**.
- Keep changes compatible with the project's existing C++17 code and OpenCL C 2.0 kernels.
- Prefer small, reviewable changes over unrelated refactoring.

## License

GGEMS is distributed under the **GNU General Public License, version 3 or later (GPL-3.0-or-later)**.

By submitting a contribution, you agree that it will be distributed under the same license. See [LICENSE.md](LICENSE.md) for details.
