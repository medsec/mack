/*
 * consoleoutput.h
 *
 *  Created on: 09.08.2012
 *      Author: Paul Kramer
 */

#ifndef CONSOLEOUTPUT_H_
#define CONSOLEOUTPUT_H_

#include <mack/callback.hpp>
/**
 * @file mack/callbacks/consoleoutput.h
 */

namespace mack {
namespace callbacks {

/**
 * @class Console_Output
 * @is_of_type{callbacks}
 * @brief This callback prints the results to the console.
 *
 * @details Only the found targets will be printed in form:
 * found 'bla': 0x128ecf542a35ac5270a87dc740918404
 *
 * @author Paul Kramer
 * @date 09.08.2012
 * @version 0.1
 */
class Console_Output : public Callback{
public:
	/**
	 * @brief the callback constructor
	 */
	Console_Output();
	/**
	 * @brief the callback destructor
	 */
	virtual ~Console_Output();
	/**
	 * @brief The callback call.
	 * @details This function prints a found statement on the console for all found targets.
	 */
	void call(unsigned char* target, mack::core::candidate* candidate, unsigned int targetlength);
};

}
}

#endif /* CONSOLEOUTPUT_H_ */
