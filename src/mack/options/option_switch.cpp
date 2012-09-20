#include "option_switch.hpp"

#include <vector>

mack::options::option_switch::option_switch(
    std::string const& short_flag,
    std::string const& long_flag,
    std::string const& brief_description,
    std::string const& detailed_description)
  : mack::options::selection_option(
      short_flag,
      long_flag,
      brief_description,
      detailed_description,
      mack::options::option_switch::create_true_false_vector(),
      "true",
      "false")
{
}

mack::options::option_switch::~option_switch()
{
}

