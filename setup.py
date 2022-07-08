import os
import pathlib
import sys

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):
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

        config = 'Debug' if self.debug else 'Release'

        cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + config,
            '-DBUILD_EXAMPLES=OFF',
        ]

        if sys.platform == 'win32':
            cmake_args.extend([
                '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG=' + str(extdir.parent.absolute()),
                '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=' + str(extdir.parent.absolute())
                ])
        else:
            cmake_args.extend([
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute())
                ])


        build_args = [
            '--config', config,
            '--parallel', '8',
        ]

        print(cmake_args, build_args)

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