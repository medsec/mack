#ifndef __MACK_OPTIONS_HANDLERS_NO_INTERACTION_HANDLER_HPP__
#define __MACK_OPTIONS_HANDLERS_NO_INTERACTION_HANDLER_HPP__

#include <mack/options/handlers/handler.hpp>

namespace mack {
namespace options {
namespace handlers {

/**
 * @brief An interaction handler which allows for no interaction.
 * @details If this handler is used, the program will be run only
 * once. If an error occurs, the program will print the error message and
 * terminate.
 * @is_of_type{option_handlers}
 *
 * @author Johannes Kiesel
 * @date Aug 10 2012
 * @see mack::options
 * @see mack::options::handlers::handler
 */
class no_interaction_handler : public handler {

  public:

    /**
     * @brief Constructor for the no_interaction_handler.
     * @param error_output_stream a stream to write error messages to
     */
    no_interaction_handler(
        std::ostream& error_output_stream = std::cerr);

    virtual
    ~no_interaction_handler();

    virtual
    values const*
    run();

  private:

    /**
     * @brief If the handler has already been run.
     */
    bool _has_been_run;

    /**
     * @brief The stream to print the error message to.
     */
    std::ostream& _error_stream;

}; // class no_interaction_handler

} // namespace handlers
} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_HANDLERS_NO_INTERACTION_HANDLER_HPP__ */

