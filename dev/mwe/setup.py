"""
This is a script to build a MWE of a build issue I'm having getting
scikit-build to properly install pyxccd in development mode.

Usage:
    python setup.py mwe generate
    pip install -e .
    python setup.py mwe analyize
    pip uninstall pkg_mwe

    rm -rf _skbuild

    python setup.py mwe generate with_cxx && pip install -e . --verbose && python -c "from pkg_mwe.myalgo_cython import call_myalgo; print(call_myalgo())"

    python setup.py mwe analyize

    python ~/code/pyxccd/dev/mwe/setup.py mwe analyize --mod_name=pyxccd --repo_dpath=$HOME/code/pyxccd
"""
import sys


class ProjectStructure():
    def __init__(self, repo_dpath='.', mod_name='pkg_mwe'):
        import ubelt as ub
        self.root = ub.Path(repo_dpath)
        self.mod_name = mod_name
        self.cxx_path    = (self.root / 'src' / 'cxx')
        self.python_path = (self.root / 'src' / 'python')
        self.mod_dpath = (self.python_path / self.mod_name)

    def generate(self, with_cxx=0):
        self.python_path.delete().ensuredir()
        self.cxx_path.delete()
        (self.root / 'CMakeLists.txt').delete()
        self.mod_dpath.ensuredir()
        (self.mod_dpath / '__init__.py').write_text('__version__ = "1.0.0"')

        # Give the MWE a CXX extension
        WITH_CXX = with_cxx
        if WITH_CXX:
            self.cxx_path.ensuredir()
            import ubelt as ub

            (self.root / 'CMakeLists.txt').write_text(ub.codeblock(
                r'''
                cmake_minimum_required(VERSION 3.13.0)
                project(pkg_mwe LANGUAGES C Fortran)


                find_package(PythonInterp REQUIRED)
                find_package(PythonLibs REQUIRED)

                ###
                # Private helper function to execute `python -c "<cmd>"`
                #
                # Runs a python command and populates an outvar with the result of stdout.
                # Be careful of indentation if `cmd` is multiline.
                #
                function(pycmd outvar cmd)
                  execute_process(
                    COMMAND "${PYTHON_EXECUTABLE}" -c "${cmd}"
                    RESULT_VARIABLE _exitcode
                    OUTPUT_VARIABLE _output)
                  if(NOT ${_exitcode} EQUAL 0)
                    message(ERROR "Failed when running python code: \"\"\"
                ${cmd}\"\"\"")
                    message(FATAL_ERROR "Python command failed with error code: ${_exitcode}")
                  endif()
                  # Remove supurflous newlines (artifacts of print)
                  string(STRIP "${_output}" _output)
                  set(${outvar} "${_output}" PARENT_SCOPE)
                endfunction()

                ###
                # Find scikit-build and include its cmake resource scripts
                #
                if (NOT SKBUILD)
                  pycmd(skbuild_location "import os, skbuild; print(os.path.dirname(skbuild.__file__))")
                  set(skbuild_cmake_dir "${skbuild_location}/resources/cmake")
                  # If skbuild is not the driver, then we need to include its utilities in our CMAKE_MODULE_PATH
                  list(APPEND CMAKE_MODULE_PATH ${skbuild_cmake_dir})
                endif()

                find_package(PythonExtensions REQUIRED)
                find_package(Cython REQUIRED)
                find_package(NumPy REQUIRED)

                # Backend C library
                add_subdirectory("src/cxx")

                # Cython library
                add_subdirectory("src/python/pkg_mwe")
                '''))

            (self.cxx_path / 'myalgo.h').write_text(ub.codeblock(
                '''
                #ifndef MYALGO_H
                #define MYALGO_H
                int myalgo(long *arr1, long *arr2, size_t num);
                #endif MYALGO_H
                '''))
            (self.cxx_path / 'myalgo.c').write_text(ub.codeblock(
                r'''
                #include <string.h>
                long myalgo(long *arr1, long *arr2, size_t num)
                {
                    for (int i = 0 ; i < num ; i++ )
                    {
                        arr2[i] = arr1[i] + arr2[i];
                    }
                    return 1;
                }
                '''))
            cmake_list_cxx = self.cxx_path / 'CMakeLists.txt'
            cmake_list_cxx.write_text(ub.codeblock(
                '''
                set(MYALGO_MODULE_NAME "myalgo")
                list(APPEND MYALGO_SOURCES "myalgo.h" "myalgo.c")
                add_library(${MYALGO_MODULE_NAME} STATIC ${MYALGO_SOURCES})
                '''))

            (self.mod_dpath / 'myalgo_cython.pyx').write_text(ub.codeblock(
                '''
                import numpy as np
                cimport numpy as np
                cdef extern from "../../cxx/myalgo.h":
                    cdef int myalgo(long *arr1, long *arr2, size_t num);

                def call_myalgo():
                    """
                    This is a docstring
                    """
                    cdef int result;
                    cdef np.ndarray[np.int64_t, ndim=1] arr1
                    cdef np.ndarray[np.int64_t, ndim=1] arr2
                    arr1 = np.array([1, 2, 3], dtype=np.int64)
                    arr2 = np.array([4, 6, 9], dtype=np.int64)
                    cdef long [:] arr1_view = arr1
                    cdef long [:] arr2_view = arr2
                    cdef size_t num = len(arr1)
                    print(f'arr1={arr1}')
                    print(f'arr2={arr2}')
                    print('calling my algo')
                    result = myalgo(&arr1_view[0], &arr2_view[0], num)
                    print(f'arr1={arr1}')
                    print(f'arr2={arr2}')
                    return result
                '''))

            (self.mod_dpath / 'CMakeLists.txt').write_text(ub.codeblock(
                '''
                set(cython_source "myalgo_cython.pyx")
                set(PYMYALGO_MODULE_NAME "myalgo_cython")

                # Translate Cython into C/C++
                add_cython_target(${PYMYALGO_MODULE_NAME} "${cython_source}" C OUTPUT_VAR sources)

                # Add other C sources
                list(APPEND sources )

                # Create C++ library. Specify include dirs and link libs as normal
                add_library(${PYMYALGO_MODULE_NAME} MODULE ${sources})
                target_include_directories(
                    ${PYMYALGO_MODULE_NAME}
                    PUBLIC
                    ${NumPy_INCLUDE_DIRS}
                    ${PYTHON_INCLUDE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}
                )

                # TODO: not sure why this isn't set in the global scope?
                # Hack around it: just hard code the module name
                set(MYALGO_MODULE_NAME "myalgo")

                # TODO: linking to the MYALGO shared object isn't working 100% yet.
                target_link_libraries(${PYMYALGO_MODULE_NAME} ${MYALGO_MODULE_NAME})

                target_compile_definitions(${PYMYALGO_MODULE_NAME} PUBLIC
                    "NPY_NO_DEPRECATED_API"
                    #"NPY_1_7_API_VERSION=0x00000007"
                )

                # Transform the C++ library into an importable python module
                python_extension_module(${PYMYALGO_MODULE_NAME})

                # Install the C++ module to the correct relative location
                # (this will be an inplace build if you use `pip install -e`)
                #file(RELATIVE_PATH pymyalgo_install_dest "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")

                # My "normal" method of setting install targets does not seem to work here. Hacking it.
                # NOTE: skbuild *seems* to place libraries in a data dir *unless* the install destination
                # corresponds exactly to the <package_dir>/<package_name> specified implicitly in setup.py
                set(pymyalgo_install_dest "src/python/pkg_mwe")
                #install(TARGETS ${MYALGO_MODULE_NAME} LIBRARY DESTINATION "${pymyalgo_install_dest}")
                install(TARGETS ${PYMYALGO_MODULE_NAME} LIBRARY DESTINATION "${pymyalgo_install_dest}")
                '''
            ))

    def analyize(self):
        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.table import Table
        import distutils.sysconfig
        import ubelt as ub
        import xdev

        console = Console()

        def rich_file_content(fpath, lexer='bash'):
            import os
            text = fpath.read_text()
            return Panel(Syntax(text, lexer), title=os.fspath(fpath))

        def print_egg_path_content(egg_info_dpath, color='blue'):
            blocklist = {'requires.txt'}
            fpaths = egg_info_dpath.ls()
            table = Table(f'[{color}]' + str(egg_info_dpath))
            for fpath in fpaths:
                if fpath.name not in blocklist:
                    panel = rich_file_content(fpath)
                    table.add_row(panel)
            console.print(table)

        print('\n')
        print('Repo Structure:')
        directory_blocklist = ['.*', '.git', 'dist', '_skbuild', 'dev']
        xdev.tree_repr(self.root, max_files=None, dirblocklist=directory_blocklist)

        print('\n')
        print('Content of the EGG Link:')
        site_dpath = ub.Path(distutils.sysconfig.get_python_lib())
        egg_link_fpaths = list(site_dpath.glob(self.mod_name.replace('_', '*') + '*.egg-link'))
        assert len(egg_link_fpaths) == 1
        egg_link_fpath = egg_link_fpaths[0]
        console.print(rich_file_content(egg_link_fpath))

        # Note: (recently 2022-08-ish) python switched to a new type of
        # This is not present in setuptools==63.2.0 but is in 65.3.0
        # editable install. TODO: incomporate this.
        # editable_fpaths = list(site_dpath.glob('__editable__*' + self.mod_name.replace('_', '*') + '*'))

        print('\n')
        print('Check easy-install.pth')
        easy_install_fpath = site_dpath / 'easy-install.pth'
        assert easy_install_fpath.exists()
        easy_install_text = easy_install_fpath.read_text()
        abs_path = self.mod_dpath.absolute().parent
        print(f'abs_path={abs_path}')
        if str(abs_path)  in easy_install_text:
            console.print('[green] Easy install dpath is good')
        else:
            console.print('[red] Easy install dpath is bad')
            console.print(rich_file_content(easy_install_fpath))

        expected_egg_info_dpath = self.root / f'src/python/{self.mod_name}.egg-info'
        all_egg_infos = [ub.Path(e).resolve() for e in xdev.find('*.egg-info', dirblocklist=directory_blocklist)]
        other_egg_infos = set(all_egg_infos) - {expected_egg_info_dpath.resolve()}
        print('expected_egg_info_dpath = {}'.format(ub.repr2(expected_egg_info_dpath, nl=1)))
        if expected_egg_info_dpath.exists():
            console.print('[green] Egg info exists in expected location')
            egg_info_dpath = expected_egg_info_dpath
            print_egg_path_content(egg_info_dpath, color='green')
        else:
            console.print('[red] Egg info exists in expected location')
            print(f'other_egg_infos={other_egg_infos}')

        if other_egg_infos:
            console.print('[red] THERE ARE UNEXEPCTED EGG INFOS')
            for egg_info_dpath in other_egg_infos:
                print_egg_path_content(egg_info_dpath, color='red')

        print('\n')
        print('Test to ensure we can import the module')
        command = f'python -c "import {self.mod_name}; print({self.mod_name})"'
        info = ub.cmd(command, verbose=3)
        if info['ret'] != 0:
            raise Exception('failed to import')
        assert str(self.mod_dpath) in info['out']


def mwe_cli():
    import scriptconfig as scfg
    class AnalyizeConfig(scfg.DataConfig):
        mod_name = 'pkg_mwe'
        repo_dpath = '.'
    config = AnalyizeConfig.cli()
    self = ProjectStructure(**config)
    if 'generate' in sys.argv:
        self.generate(with_cxx='with_cxx' in sys.argv)
    if 'analyize' in sys.argv:
        self.analyize()


if __name__ == '__main__':
    if 'mwe' in sys.argv:
        mwe_cli()
        sys.exit(0)

    import os
    DISABLE_C_EXTENSIONS = os.environ.get('DISABLE_C_EXTENSIONS', '')
    if DISABLE_C_EXTENSIONS == '1':
        from setuptools import setup
    else:
        from skbuild import setup
    from setuptools import find_packages

    packages = find_packages('./src/python')
    setup(
        package_dir={
            '': 'src/python',
        },
        install_requires=['packaging', 'ubelt'],
        name='pkg_mwe',
        version="1.0.0",
        description='MWE of build issue',
        packages=packages,
        include_package_data=True,
    )
