import os
import pathlib
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):
    user_options = build_ext_orig.user_options + \
            [('generator=', None, 'The CMake generator to use.'),
             ('opengl=', None, 'Whether to build with OpenGL visualization.'),
             ('examples=', None, 'Whether to build examples.')]

    def initialize_options(self):
        super().initialize_options()
        self.generator = None
        self.opengl = "OFF"
        self.examples = "OFF"
    
    def finalize_options(self):
        super().finalize_options()
        self.generator = self.distribution.get_command_obj('build_ext').generator
        self.opengl = self.distribution.get_command_obj('build_ext').opengl
        self.examples = self.distribution.get_command_obj('build_ext').examples

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # Config
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + config,
        ]
        
        # OpenGL
        cmake_args.append('-DOPENGL_VISUALIZATION=' + self.opengl)

        # Examples
        cmake_args.append('-DBUILD_EXAMPLES=' + self.examples)
  
        # Output Directory
        if sys.platform == 'win32':
            cmake_args.extend([
                '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG=' + str(extdir.parent.absolute()),
                '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=' + str(extdir.parent.absolute())
                ])
        else:
            cmake_args.extend([
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute())
                ])
        
        # Generator
        if self.generator is not None:
            cmake_args.append('-G' + self.generator)

        build_args = [
            '--config', config,
            '--parallel', '8',
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)

        os.chdir(str(cwd))


setup(
    name='ggems',
    version='1.2',
    packages=['ggems'],
    package_dir={'ggems': 'python_module'},
    ext_modules=[CMakeExtension('.')],
    cmdclass={
        'build_ext': build_ext,
    }
)