/*
 * container.h
 *
 *  Created on: 31.08.2012
 *      Author: Felix Trojan
 *
 *      Saves all Targets:candidates in a Map
 */
#include <string>

#ifndef CONTAINER_CALLBACK_H_
#define CONTAINER_CALLBACK_H_

#include <mack/callback.hpp>
#include <map>

/**
 * @file mack/callbacks/container_callback.h
 */

namespace mack {
namespace callbacks {

/**
 * @class container_callback
 * @is_of_type{callbacks}
 * @brief This callback prints the results to the console and stores it into a stl map.
 *
 * @details Only the found targets will be printed in form:
 * found 'bla': 0x128ecf542a35ac5270a87dc740918404
 *
 * Only the found targets will be stored in the map in form:
 * map[hash] = candidate
 *
 * @author Felix Trojan
 * @date 31.08.2012
 * @version 0.1
 */
class container_callback : public Callback {
public:
	/**
	 * @brief the callback constructor
	 */
	container_callback();
	/**
	 * @brief the callback destructor
	 */
	virtual ~container_callback();
	/**
	 * @brief The callback call.
	 * @details This function prints a found statement on the console for all found targets
	 * and writes the into a stl map.
	 */
	void call(unsigned char* target, mack::core::candidate* candidate, unsigned int targetlength);
	std::map<std::string,mack::core::candidate> get_container();

private:
	std::map<std::string,mack::core::candidate> _container;
};

}
}

#endif /* CONTAINER_CALLBACK_H_ */
