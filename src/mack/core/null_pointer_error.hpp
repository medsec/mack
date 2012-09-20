#ifndef __MACK_CORE_NULL_POINTER_ERROR_HPP__
#define __MACK_CORE_NULL_POINTER_ERROR_HPP__

#include <exception>
#include <boost/exception/exception.hpp>

namespace mack {
namespace core {

/**
 * @brief An exception which is thrown if a pointer is <tt>null</tt> but must
 * not be so.
 *
 * @author Johannes Kiesel
 * @date Aug 24th, 2012
 */
struct null_pointer_error : virtual std::exception, virtual boost::exception
{
};

} // namespace core
} // namespace mack

#endif /* __MACK_CORE_NULL_POINTER_ERROR_HPP__ */
