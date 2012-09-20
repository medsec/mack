#ifndef CRACKER_H_
#define CRACKER_H_

#include <string>
#include <mack/callback.hpp>
#include <mack/target_loader.hpp>



namespace mack {
namespace core {

#define VALUE_SIZE 20

/**
 * @option_type{crackers,Crackers}
 * @option_type_class{mack::core::Cracker}
 */
/**
 * @class Cracker
 * @brief This is the base class for all Crackers
 *
 * @details A Cracker tries to find preimages to the Targets it gets with the target loader.
 * Found preimages are then processed by a callback class that e.g. writes them to the Standard Output.
 * Derived Classes are implemented as template classes for the Class Algorithm to process every possible
 * algorithm implemented.
 *
 * @date 13.09.2012
 * @version 0.1
 */
class Cracker {
public:
	Cracker() { };

	/**
	 * @brief Prepares and launches to Kernel for the cracking process
	 * @param callback callback to be used for the found targets
	 * @param target_loader target_loader that is used for loading the targets
	 */
    virtual void crack(mack::callbacks::Callback* callback, mack::targetloaders::Target_Loader* target_loader) const = 0;
	virtual ~Cracker() { };

};

} // namespace core
} // namespace mack

#endif /* CRACKER_H_ */
