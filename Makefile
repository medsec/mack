SHELL = /bin/sh

### Adjust to your system:
# Library path of used libraries
LIBPATH =-L/usr/lib -L/usr/local/cuda/lib64
# Compiler optimization level
OPTIMIZATION_LEVEL=2
### Required libraries
LIBS =-lboost_system -lboost_filesystem -lexpat
CUDA_LIBS =-lcudart

### Compiler settings
CXX =g++-4.6
CXXFLAGS =-Wall -O$(OPTIMIZATION_LEVEL)
CUDA =/usr/local/cuda/include 

NVXX =nvcc
NVXXFLAGS = -arch=sm_20

LDXXFLAGS =$(LIBS) $(CUDA_LIBS) $(LIBPATH)

# Disabling builtin rules
.SUFFIXES:

# Directories

SOURCE_PATH = src
PROGRAMS_SOURCE_PATH = programs
OBJ_PATH = obj
BIN = bin

TESTS_SOURCE_PATH = tests/src
TESTS_OBJ_PATH = tests/obj
TESTS_BIN_PATH = tests

CONFIGURATION_SOURCE_PATH = configuration/src
CONFIGURATION_PROGRAMS_PATH = configuration/programs
CONFIGURATION_OBJ_PATH = configuration/obj
CONFIGURATION_BIN_PATH = configuration/bin

# ---------------------------------- #

MKDIR = mkdir

mkdirs = $(if $(wildcard $(1)),,$(1) $(call mkdirs,$(abspath $(dir $(1)))))

SOURCE_DIRS = $(sort $(dir $(shell find $(SOURCE_PATH)/)))
SOURCES = $(filter %.cpp %.cu, $(foreach dir,$(SOURCE_DIRS),$(wildcard $(dir)*)))
SOURCES_INCLUDE = $(foreach directory,$(sort $(dir $(SOURCES))),-I$(directory))
OBJECTS = $(patsubst %,$(OBJ_PATH)/%,$(subst .cpp,.o,$(subst .cu,.o,$(SOURCES))))
OBJECTS_LINK_PATTERN = $(foreach directory,$(sort $(dir $(OBJECTS))),$(directory)*.o)

PROGRAM_SOURCE_DIRS = $(sort $(dir $(shell find $(PROGRAMS_SOURCE_PATH)/)))
PROGRAM_SOURCES = $(filter %.cpp %.cu, $(foreach dir,$(PROGRAM_SOURCE_DIRS),$(wildcard $(dir)*)))
PROGRAM_OBJECTS = $(patsubst %,$(OBJ_PATH)/%,$(subst .cpp,,$(subst .cu,,$(PROGRAM_SOURCES))))
PROGRAMS = $(patsubst %,$(BIN)/%,$(subst $(PROGRAMS_SOURCE_PATH)/,,$(subst .cpp,,$(subst .cu,,$(PROGRAM_SOURCES)))))

CONFIGURATION_SOURCE_DIRS = $(sort $(dir $(shell find $(CONFIGURATION_SOURCE_PATH)/)))
CONFIGURATION_SOURCES = $(filter %.cpp %.cu, $(foreach dir,$(CONFIGURATION_SOURCE_DIRS),$(wildcard $(dir)*)))
CONFIGURATION_SOURCES_INCLUDE = $(foreach directory,$(sort $(dir $(CONFIGURATION_SOURCES))),-I$(directory))
CONFIGURATION_OBJECTS = $(patsubst %,$(CONFIGURATION_OBJ_PATH)/%,$(subst $(CONFIGURATION_SOURCE_PATH)/,,$(subst .cpp,.o,$(subst .cu,.o,$(CONFIGURATION_SOURCES)))))
CONFIGURATION_OBJECTS_LINK_PATTERN = $(foreach directory,$(sort $(dir $(CONFIGURATION_OBJECTS))),$(directory)*.o)

TESTS_DIRS = $(sort $(dir $(shell find $(TESTS_SOURCE_PATH)/)))
TESTS_SOURCES = $(filter %.cpp %.cu, $(foreach dir,$(TESTS_DIRS),$(wildcard $(dir)*)))
TESTS_SOURCES_INCLUDE = $(foreach directory,$(sort $(dir $(TESTS_SOURCES))),-I$(directory))
TESTS_OBJECTS = $(patsubst %,$(TESTS_OBJ_PATH)/%,$(subst $(TESTS_SOURCE_PATH)/,src/,$(subst .cpp,.o,$(subst .cu,.o,$(TESTS_SOURCES)))))
TESTS_OBJECTS_LINK_PATTERN = $(foreach directory,$(sort $(dir $(TESTS_OBJECTS))),$(directory)*.o)

NEW_OBJECT_DIRS = $(sort $(foreach dir,$(sort $(foreach obj,$(OBJECTS) $(SOURCE_PATH)/mack/options/types/,$(dir $(obj)))),$(call mkdirs,$(abspath $(dir)))))
NEW_PROGRAM_OBJECT_DIRS = $(sort $(foreach dir,$(sort $(foreach obj,$(PROGRAM_OBJECTS),$(dir $(obj)))),$(call mkdirs,$(abspath $(dir)))))
NEW_CONFIGURATION_OBJECT_DIRS = $(sort $(foreach dir,$(sort $(foreach obj,$(CONFIGURATION_OBJECTS),$(dir $(obj)))),$(call mkdirs,$(abspath $(dir))))) \
																$(call mkdirs,$(abspath $(CONFIGURATION_BIN_PATH)))
NEW_TESTS_OBJECT_DIRS = $(sort $(foreach dir,$(sort $(foreach obj,$(TESTS_OBJECTS),$(dir $(obj)))),$(call mkdirs,$(abspath $(dir)))))
NEW_BIN_DIRS = $(sort $(foreach dir,$(sort $(foreach obj,$(PROGRAMS),$(dir $(obj)))),$(call mkdirs,$(abspath $(dir)))))
NEW_PROGRAM_DIRS = $(sort $(NEW_BIN_DIRS) $(NEW_PROGRAM_OBJECT_DIRS))

PARSE_DOXYGEN = $(CONFIGURATION_BIN_PATH)/parse_doxygen

# targets
standard:
	@echo "Usage:"
	@echo "> make update"
	@echo "    Updates and parses the doxygen documentation."
	@echo "> make programs"
	@echo "    Compiles the Mack the Knife programs."
	@echo "    Requires 'make update' to be run before."
	@echo "> make tests"
	@echo "    Compiles and runs the test suite"
	@echo "> make all"
	@echo "    Runs make update, programs and tests"
	@echo "> make clean"
	@echo "    Deletes all documentation, configuration, object files and binaries"

programs: program_dirs obj_dirs $(PROGRAMS)

all:
	make update
	make programs
	make tests

update: configuration_obj_dirs obj_dirs $(PARSE_DOXYGEN)
	doxygen doc/doxygen.conf
	$(PARSE_DOXYGEN)

tests: obj_dirs tests_obj_dirs $(TESTS_BIN_PATH)/run_tests
	$(TESTS_BIN_PATH)/run_tests

clean:
	rm -rf $(OBJ_PATH) $(CONFIGURATION_OBJ_PATH) $(TESTS_OBJ_PATH) $(BIN) $(CONFIGURATION_BIN_PATH) $(TESTS_BIN_PATH)/run_tests doc/html doc/xml doc/latex doc/man
	rm -rf $(SOURCE_PATH)/mack/options/types $(SOURCE_PATH)/mack/options/programs.cpp

include Makefile.targets

# programs
$(BIN)/%:$(OBJ_PATH)/$(PROGRAMS_SOURCE_PATH)/%.o $(OBJECTS)
	$(NVXX) $(NVXXFLAGS) $(LDXXFLAGS) $(OBJECTS_LINK_PATTERN) $< -o $@

# configuration programs
$(PARSE_DOXYGEN):$(CONFIGURATION_PROGRAMS_PATH)/parse_doxygen.cpp $(CONFIGURATION_OBJECTS) $(OBJ_PATH)/$(SOURCE_PATH)/mack/core/xml_parser.o $(OBJ_PATH)/$(SOURCE_PATH)/mack/core/files.o
	$(CXX) $(CXXFLAGS) $(LIBS) $(LIBPATH) -I$(SOURCE_PATH) $(SOURCES_INCLUDE) -I$(CONFIGURATION_SOURCE_PATH) $(CONFIGURATION_SOURCES_INCLUDE) $(CONFIGURATION_OBJECTS_LINK_PATTERN) $(OBJ_PATH)/$(SOURCE_PATH)/mack/core/files.o $(OBJ_PATH)/$(SOURCE_PATH)/mack/core/xml_parser.o $(CONFIGURATION_PROGRAMS_PATH)/parse_doxygen.cpp -o $@

# test program
$(TESTS_OBJ_PATH)/run_tests.o:tests/main.cpp $(OBJECTS) $(TESTS_OBJECTS)
	$(CXX) $(CXXFLAGS) -I$(SOURCE_PATH) $(SOURCES_INCLUDE) -I$(TESTS_SOURCE_PATH) $(TESTS_SOURCES_INCLUDE) -c $< -o $@

$(TESTS_BIN_PATH)/run_tests: $(TESTS_OBJ_PATH)/run_tests.o $(OBJECTS) $(TESTS_OBJECTS)
	$(NVXX) $(NVXXFLAGS) $(LDXXFLAGS) -lunittest++ -I$(CUDA) $(OBJECTS_LINK_PATTERN) $(TESTS_OBJECTS_LINK_PATTERN) $< -o $@

# objects
$(OBJ_PATH)/$(SOURCE_PATH)/%.o:$(SOURCE_PATH)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(SOURCE_PATH) $(SOURCES_INCLUDE) -c $< -o $@

$(OBJ_PATH)/$(SOURCE_PATH)/%.o:$(SOURCE_PATH)/%.cu
	$(NVXX) $(NVXXFLAGS) -I$(SOURCE_PATH) -I$(CUDA) $(SOURCES_INCLUDE) -dc $< -o $@

# program objects
$(OBJ_PATH)/$(PROGRAMS_SOURCE_PATH)/%.o:$(PROGRAMS_SOURCE_PATH)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(SOURCE_PATH) $(SOURCES_INCLUDE) -c $< -o $@

$(OBJ_PATH)/$(PROGRAMS_SOURCE_PATH)/%.o:$(PROGRAMS_SOURCE_PATH)/%.cu
	$(NVXX) $(NVXXFLAGS) -I$(SOURCE_PATH) -I$(CUDA) $(SOURCES_INCLUDE) -dc $< -o $@

# configuration objects
$(CONFIGURATION_OBJ_PATH)/%.o:$(CONFIGURATION_SOURCE_PATH)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(SOURCE_PATH) $(SOURCES_INCLUDE) -I$(CONFIGURATION_SOURCE_PATH) $(CONFIGURATION_SOURCES_INCLUDE) -c $< -o $@

$(CONFIGURATION_OBJ_PATH)/%.o:$(CONFIGURATION_SOURCE_PATH)/%.cu
	$(NVXX) $(NVXXFLAGS) -I$(SOURCE_PATH) -I$(CUDA) $(SOURCES_INCLUDE) -I$(CONFIGURATION_SOURCE_PATH) $(CONFIGURATION_SOURCES_INCLUDE) -dc $< -o $@

# test objects
$(TESTS_OBJ_PATH)/src/%.o:$(TESTS_SOURCE_PATH)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(SOURCE_PATH) $(SOURCES_INCLUDE) -I$(TESTS_SOURCE_PATH) $(TESTS_SOURCES_INCLUDE) -c $< -o $@

$(TESTS_OBJ_PATH)/src/%.o:$(TESTS_SOURCE_PATH)/%.cu
	$(NVXX) $(NVXXFLAGS) -I$(SOURCE_PATH) -I$(CUDA) $(SOURCES_INCLUDE) -I$(TESTS_SOURCE_PATH) $(TESTS_SOURCES_INCLUDE) -dc $< -o $@

# directories
program_dirs:
	$(if $(strip $(NEW_PROGRAM_DIRS)),$(MKDIR) $(NEW_PROGRAM_DIRS))

obj_dirs:
	$(if $(strip $(NEW_OBJECT_DIRS)),$(MKDIR) $(NEW_OBJECT_DIRS))

configuration_obj_dirs:
	$(if $(strip $(NEW_CONFIGURATION_OBJECT_DIRS)),$(MKDIR) $(NEW_CONFIGURATION_OBJECT_DIRS))

tests_obj_dirs:
	$(if $(strip $(NEW_TESTS_OBJECT_DIRS)),$(MKDIR) $(NEW_TESTS_OBJECT_DIRS))

