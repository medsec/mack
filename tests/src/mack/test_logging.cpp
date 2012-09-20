#include <unittest++/UnitTest++.h>

#include <mack/logging.hpp>
#include <mack/logging/stream_logger.hpp>

#include <string>

SUITE(mack_logging)
{
  TEST(TestClearLoggers)
  {
    mack::logging::set_log_level_to_debug();
    mack::logging::clear_loggers();
    CHECK(!mack::logging::is_debug_on());
    CHECK(!mack::logging::is_info_on());
    CHECK(!mack::logging::is_warning_on());
    CHECK(!mack::logging::is_error_on());
  }

  TEST(TestSetDebugLogger)
  {
    mack::logging::set_log_level_to_debug();
    mack::logging::clear_loggers();
    std::stringstream stream;

    mack::logging::set_debug_logger(new mack::logging::stream_logger(stream));
    LOG_DEBUG("debug");
    LOG_INFO("info");
    LOG_WARNING("warning");
    LOG_ERROR("error");

    CHECK_EQUAL(stream.str(), "debug\ninfo\nwarning\nerror\n");
  }

  TEST(TestSetInfoLogger)
  {
    mack::logging::set_log_level_to_debug();
    mack::logging::clear_loggers();
    std::stringstream stream;

    mack::logging::set_info_logger(new mack::logging::stream_logger(stream));
    LOG_DEBUG("debug");
    LOG_INFO("info");
    LOG_WARNING("warning");
    LOG_ERROR("error");

    CHECK_EQUAL(stream.str(), "info\nwarning\nerror\n");
  }

  TEST(TestSetWarningLogger)
  {
    mack::logging::set_log_level_to_debug();
    mack::logging::clear_loggers();
    std::stringstream stream;

    mack::logging::set_warning_logger(new mack::logging::stream_logger(stream));
    LOG_DEBUG("debug");
    LOG_INFO("info");
    LOG_WARNING("warning");
    LOG_ERROR("error");

    CHECK_EQUAL(stream.str(), "warning\nerror\n");
  }

  TEST(TestSetErrorLogger)
  {
    mack::logging::set_log_level_to_debug();
    mack::logging::clear_loggers();
    std::stringstream stream;

    mack::logging::set_error_logger(new mack::logging::stream_logger(stream));
    LOG_DEBUG("debug");
    LOG_INFO("info");
    LOG_WARNING("warning");
    LOG_ERROR("error");

    CHECK_EQUAL(stream.str(), "error\n");
  }

  TEST(TestSetLogLevelToDebug)
  {
    mack::logging::clear_loggers();
    std::stringstream stream;
    mack::logging::set_debug_logger(new mack::logging::stream_logger(stream));

    mack::logging::set_log_level_to_debug();

    LOG_DEBUG("debug");
    LOG_INFO("info");
    LOG_WARNING("warning");
    LOG_ERROR("error");

    CHECK_EQUAL(stream.str(), "debug\ninfo\nwarning\nerror\n");
  }

  TEST(TestSetLogLevelToInfo)
  {
    mack::logging::clear_loggers();
    std::stringstream stream;
    mack::logging::set_debug_logger(new mack::logging::stream_logger(stream));

    mack::logging::set_log_level_to_info();

    LOG_DEBUG("debug");
    LOG_INFO("info");
    LOG_WARNING("warning");
    LOG_ERROR("error");

    CHECK_EQUAL(stream.str(), "info\nwarning\nerror\n");
  }

  TEST(TestSetLogLevelToWarning)
  {
    mack::logging::clear_loggers();
    std::stringstream stream;
    mack::logging::set_debug_logger(new mack::logging::stream_logger(stream));

    mack::logging::set_log_level_to_warning();

    LOG_DEBUG("debug");
    LOG_INFO("info");
    LOG_WARNING("warning");
    LOG_ERROR("error");

    CHECK_EQUAL(stream.str(), "warning\nerror\n");
  }

  TEST(TestSetLogLevelToError)
  {
    mack::logging::clear_loggers();
    std::stringstream stream;
    mack::logging::set_debug_logger(new mack::logging::stream_logger(stream));

    mack::logging::set_log_level_to_error();

    LOG_DEBUG("debug");
    LOG_INFO("info");
    LOG_WARNING("warning");
    LOG_ERROR("error");

    CHECK_EQUAL(stream.str(), "error\n");
  }

  TEST(TestLogStream)
  {
    mack::logging::clear_loggers();
    std::stringstream stream;
    mack::logging::set_debug_logger(new mack::logging::stream_logger(stream));

    mack::logging::set_log_level_to_debug();

    LOG_STREAM_DEBUG("debug");
    LOG_STREAM_DEBUG("part 1, " << 2 << ", part 3");

    CHECK_EQUAL(stream.str(), "debug\npart 1, 2, part 3\n");
  }
}

