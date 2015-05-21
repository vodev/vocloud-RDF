# Random Forest for VO-CLOUD

The wrapper is designed to run under `Python3` or higher and does not offer some functionality
under lower versions. However, running the cudaTree package is supported only under Python 2.7, due
to the technical limitations of the library.

To run H2O, the user have to have Java installed. The H2O instance can run prior to the execution of
the wrapper, or if there is no instance running, the wrapper will start it provided it finds an
h2o.jar file in the root folder of the wrapper.

To run the wrapper, use the command `python3 runRF [input_file1, ...]`, which will run it in
`Python3`. For `Python2`, just change the initial command for python2.
