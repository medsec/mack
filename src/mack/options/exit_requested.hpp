#ifndef __MACK_OPTIONS_EXIT_REQUESTED_HPP__
#define __MACK_OPTIONS_EXIT_REQUESTED_HPP__

namespace mack {
namespace options {

/**
 * @brief Thrown in order to indicate that program exit was requested.
 *
 * @author Johannes Kiesel
 * @date Aug 08 2012
 */
struct exit_requested
{
  public:

    /**
     * @brief Constructor for a request of exit.
     * @param exit an exit code (either <tt>EXIT_SUCCESS</tt> or
     * <tt>EXIT_FAILURE</tt> of <tt>cstdlib</tt>)
     */
    exit_requested(
        const int exit);

    /**
     * @brief The exit code signaling how the exit was requested.
     * @details This is <tt>EXIT_SUCCESS</tt> (from <tt>cstdlib</tt>) if
     * everything worked fine.
     */
    int exit_code;
};

} // namespace options
} // namespace mack

#endif /* __MACK_OPTIONS_EXIT_REQUESTED_HPP__ */
