# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/liuchang/tools/C++/cmake/bin/cmake

# The command to remove a file.
RM = /home/liuchang/tools/C++/cmake/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liuchang/codes/computer_vision/bayers_matting

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liuchang/codes/computer_vision/bayers_matting/build

# Include any dependencies generated for this target.
include CMakeFiles/executor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/executor.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/executor.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/executor.dir/flags.make

CMakeFiles/executor.dir/src/bayers_matting.cpp.o: CMakeFiles/executor.dir/flags.make
CMakeFiles/executor.dir/src/bayers_matting.cpp.o: ../src/bayers_matting.cpp
CMakeFiles/executor.dir/src/bayers_matting.cpp.o: CMakeFiles/executor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liuchang/codes/computer_vision/bayers_matting/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/executor.dir/src/bayers_matting.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/executor.dir/src/bayers_matting.cpp.o -MF CMakeFiles/executor.dir/src/bayers_matting.cpp.o.d -o CMakeFiles/executor.dir/src/bayers_matting.cpp.o -c /home/liuchang/codes/computer_vision/bayers_matting/src/bayers_matting.cpp

CMakeFiles/executor.dir/src/bayers_matting.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/executor.dir/src/bayers_matting.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liuchang/codes/computer_vision/bayers_matting/src/bayers_matting.cpp > CMakeFiles/executor.dir/src/bayers_matting.cpp.i

CMakeFiles/executor.dir/src/bayers_matting.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/executor.dir/src/bayers_matting.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liuchang/codes/computer_vision/bayers_matting/src/bayers_matting.cpp -o CMakeFiles/executor.dir/src/bayers_matting.cpp.s

# Object files for target executor
executor_OBJECTS = \
"CMakeFiles/executor.dir/src/bayers_matting.cpp.o"

# External object files for target executor
executor_EXTERNAL_OBJECTS =

../bin/executor: CMakeFiles/executor.dir/src/bayers_matting.cpp.o
../bin/executor: CMakeFiles/executor.dir/build.make
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_gapi.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_stitching.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_alphamat.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_aruco.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_bgsegm.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_bioinspired.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_ccalib.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_cudabgsegm.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_cudafeatures2d.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_cudaobjdetect.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_cudastereo.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_dnn_objdetect.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_dnn_superres.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_dpm.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_face.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_freetype.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_fuzzy.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_hdf.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_hfs.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_img_hash.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_intensity_transform.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_line_descriptor.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_mcc.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_quality.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_rapid.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_reg.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_rgbd.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_saliency.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_sfm.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_stereo.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_structured_light.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_superres.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_surface_matching.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_tracking.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_videostab.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_xfeatures2d.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_xobjdetect.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_xphoto.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_shape.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_highgui.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_datasets.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_plot.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_text.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_ml.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_phase_unwrapping.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_cudacodec.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_videoio.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_cudaoptflow.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_cudalegacy.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_cudawarping.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_optflow.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_ximgproc.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_video.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_dnn.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_imgcodecs.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_objdetect.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_calib3d.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_features2d.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_flann.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_photo.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_cudaimgproc.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_cudafilters.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_imgproc.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_cudaarithm.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_core.so.4.5.2
../bin/executor: /usr/local/opencv/opencv-4.5.2/lib/libopencv_cudev.so.4.5.2
../bin/executor: CMakeFiles/executor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liuchang/codes/computer_vision/bayers_matting/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/executor"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/executor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/executor.dir/build: ../bin/executor
.PHONY : CMakeFiles/executor.dir/build

CMakeFiles/executor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/executor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/executor.dir/clean

CMakeFiles/executor.dir/depend:
	cd /home/liuchang/codes/computer_vision/bayers_matting/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liuchang/codes/computer_vision/bayers_matting /home/liuchang/codes/computer_vision/bayers_matting /home/liuchang/codes/computer_vision/bayers_matting/build /home/liuchang/codes/computer_vision/bayers_matting/build /home/liuchang/codes/computer_vision/bayers_matting/build/CMakeFiles/executor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/executor.dir/depend
