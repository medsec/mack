#ifndef CALLBACK_HPP_
#define CALLBACK_HPP_

#include <string.h>
#include <stdio.h>
#include <mack/core/algorithm.cuh>


namespace mack {
namespace callbacks {

/**
 * @option_type{callbacks,Target Found Callbacks}
 * @brief Classes for handling found targets.
 * @option_type_class{mack::callbacks::Callback}
 */
/**
 * @class Callback
 * @brief Classes for handling found targets.
 * @author Paul Kramer
 * @date 09.08.2012
 */
class Callback {
public:
	/**
	 * @brief Callback call method. This method must be implemented by inheriting classes
	 * and will be called from the crackers crack methods.
	 * @param target the target to do sth. with
	 * @param candidate the candidate corresponding to the target or in other words the message of the hash
	 * @param targetlength the length of the target, depending on the hash algorithm
	 */
	virtual void call(unsigned char* target, mack::core::candidate* candidate, unsigned int targetlength) = 0;
};

}
}

#endif /* CALLBACK_HPP_ */
