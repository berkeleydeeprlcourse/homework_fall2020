Welcome to MuJoCo Pro version 1.50.

The full documentation is available at http://www.mujoco.org/book
The most relevant chapters are Overview, MJCF Models, and MuJoCo Pro.

Here we provide brief notes to get you started:


The activation key (which you should have received with your license) is a
plain-text file whose path must be passed to the mj_activate() function.
The code samples assume that it is called mjkey.txt in the bin directory.

Once you have mjkey.txt in the bin directory, run:
  simulate ../model/humanoid.xml  (or ./simulate on Linux and OSX)
to see MuJoCo Pro in action.

On Linux, you can use LD_LIBRARY_PATH to point the dynamic linker to the
.so files, or copy them to a directory that is already in the linker path.
On OSX, the MuJoCo Pro dynamic library is compiled with @executable_path/
to avoid the need for installation in a predefined directory.

In general, the directory structure we have provided is merely a suggestion;
feel free to re-organize it if needed. MuJoCo Pro does not have an installer
and does not write any files outside the executable directory.

The makefile in the sample directory generates binaries in the bin directory.
These binaries are pre-compiled and included in the software distribution.

While the software distribution contains only one model (humanoid.xml),
additional models are available at http://www.mujoco.org/forum under Resources.
