#ifndef DEFAULT_LOADER_H_
#define DEFAULT_LOADER_H_

#include <mack/target_loader.hpp>
#include <stdio.h>

/**
 * @file mack/targetloaders/default_loader.hpp
 */

namespace mack {
namespace targetloaders{

/**
 * @class Default_Loader
 * @is_of_type{targetloaders}
 * @brief This is the default target loader. It loads targets from plain files.
 *
 * It does not matter which sizes the targets are, it determines it while loading.
 * All targets must have the same size and use the same algorithm to work fine.
 * The number of the targets and its size should be much smaller than GPU's memory.
 */

class Default_Loader : public mack::targetloaders::Target_Loader {
public:
	unsigned char* load_file_32(const char* file_path);
	unsigned char* load_file_8(const char* file_path);

	virtual ~Default_Loader();
};

}
}

#endif /* DEFAULT_LOADER_H_ */
