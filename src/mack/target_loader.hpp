#ifndef TARGET_LOADER_H_
#define TARGET_LOADER_H_

#include <stdio.h>


namespace mack {
namespace targetloaders {

/**
 * @option_type{targetloaders,Target Loaders}
 * @option_type_class{mack::targetloaders::Target_Loader}
 */
/**
 * @class Target_Loader
 * @brief This is the target_loader base class.
 */
class Target_Loader {
public:
	/**
	 * @brief The init method initializes the targetloader with the correct target size. This
	 * is very important for cutting the loaded file into targets.
	 * @param target_size the target size.
	 */
	void init(size_t target_size);

	/**
	 * @brief counts the lines of the file with given filename
	 * @param file_path the path of the file to load
	 */
	size_t line_count(const char* file_path);
	/**
	 * @brief a loading method which returns the file enties in 32 bit blocks.
	 * This method must be implemented from inherited classes.
	 * @param file_path the path of the file to load
	 * @return an array of characters with the targets
	 */
	virtual unsigned char* load_file_32(const char* file_path) = 0;
	/**
	 * @brief a loading method which returns the file enties in 8 bit blocks.
	 * This method must be implemented from inherited classes.
	 * @param file_path the path of the file to load
	 * @return an array of characters with the targets
	 */
	virtual unsigned char* load_file_8(const char* file_path) = 0;
	/**
	 * @brief Returns the number of the targets.
	 * @returns number of the targets.
	 */
	size_t get_target_count();
	/**
	 * @brief Getter for the targets.
	 * @param index the position of the target
	 * @param targets the targets array
	 * @param target_size the size of one target
	 * @returns the target at position index
	 */
	unsigned char* get_target(size_t index, unsigned char* targets, size_t target_size);
	/**
	 * @brief Getter for the targets.
	 * @param index the position of the target
	 * @param targets the targets array
	 * @returns the target at position index using the internal target size
	 */
	unsigned char* get_target(size_t index, unsigned char* targets);

	virtual ~Target_Loader();

protected:
	size_t _target_count;
	size_t _target_size;
};

}
}

#endif /* TARGET_LOADER_H_ */
