#include "logging.hpp"

#include <set>

#include <mack/logging/stream_logger.hpp>

enum log_level_t { DEBUG, INFO, WARNING, ERROR };

mack::logging::logger* debug_logger = new mack::logging::stream_logger(std::cout);

mack::logging::logger* info_logger = debug_logger;

mack::logging::logger* warning_logger = new mack::logging::stream_logger(std::cerr);

mack::logging::logger* error_logger = warning_logger;

log_level_t log_level = INFO;

void
set_logger(
    mack::logging::logger* logger,
    log_level_t level)
{
  std::set<mack::logging::logger*> old_loggers;
  std::set<mack::logging::logger*> kept_loggers;
  if (level <= DEBUG)
  {
    old_loggers.insert(debug_logger);
    debug_logger = logger;
  }
  else
  {
    kept_loggers.insert(debug_logger);
  }

  if (level <= INFO)
  {
    old_loggers.insert(info_logger);
    info_logger = logger;
  }
  else
  {
    kept_loggers.insert(info_logger);
  }

  if (level <= WARNING)
  {
    old_loggers.insert(warning_logger);
    warning_logger = logger;
  }
  else
  {
    kept_loggers.insert(warning_logger);
  }

  if (level <= ERROR)
  {
    old_loggers.insert(error_logger);
    error_logger = logger;
  }
  else
  {
    kept_loggers.insert(error_logger);
  }


  for (std::set<mack::logging::logger*>::iterator old_loggers_it =
        old_loggers.begin();
      old_loggers_it != old_loggers.end();
      ++old_loggers_it)
  {
    if (*old_loggers_it != NULL && kept_loggers.count(*old_loggers_it) == 0)
    {
      delete *old_loggers_it;
    }
  }
}

void
mack::logging::set_debug_logger(
    logger* logger)
{
  set_logger(logger, DEBUG);
}

void
mack::logging::set_info_logger(
    logger* logger)
{
  set_logger(logger, INFO);
}

void
mack::logging::set_warning_logger(
    logger* logger)
{
  set_logger(logger, WARNING);
}

void
mack::logging::set_error_logger(
    logger* logger)
{
  set_logger(logger, ERROR);
}

void
mack::logging::clear_loggers()
{
  set_logger(NULL, DEBUG);
}

void
mack::logging::set_log_level_to_debug()
{
  log_level = DEBUG;
}

void
mack::logging::set_log_level_to_warning()
{
  log_level = WARNING;
}

void
mack::logging::set_log_level_to_info()
{
  log_level = INFO;
}

void
mack::logging::set_log_level_to_error()
{
  log_level = ERROR;
}

void
mack::logging::debug(
    std::string const& message)
{
  if (log_level <= DEBUG && debug_logger != NULL)
  {
    debug_logger->log(message);
  }
}

void
mack::logging::info(
    std::string const& message)
{
  if (log_level <= INFO && info_logger != NULL)
  {
    info_logger->log(message);
  }
}

void
mack::logging::warning(
    std::string const& message)
{
  if (log_level <= WARNING && warning_logger != NULL)
  {
    warning_logger->log(message);
  }
}
  
void
mack::logging::error(
    std::string const& message)
{
  if (log_level <= ERROR && error_logger != NULL)
  {
    error_logger->log(message);
  }
}

bool
mack::logging::is_debug_on()
{
  return log_level <= DEBUG && debug_logger != NULL;
}

bool
mack::logging::is_info_on()
{
  return log_level <= INFO && info_logger != NULL;
}

bool
mack::logging::is_warning_on()
{
  return log_level <= WARNING && warning_logger != NULL;
}

bool
mack::logging::is_error_on()
{
  return log_level <= ERROR && error_logger != NULL;
}
