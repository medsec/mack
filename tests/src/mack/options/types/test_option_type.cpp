#include "test_option_type.hpp"

#include <mack/options/option.hpp>
#include <mack/options/selection_option.hpp>
#include <mack/options/option_switch.hpp>
#include <mack/options/type_option.hpp>

#include <mack/options/exceptions.hpp>
#include <boost/throw_exception.hpp>

// #include <tests/mack/options/recursive_class>
// ...

mack::options::test_option_type::test_option_type()
  : _class_names()
{
  _class_names.push_back("tests::mack::options::recursive_class");
  _class_names.push_back("tests::mack::options::ambiguous_class");
  _class_names.push_back("tests::mack::options::empty_class");
  _class_names.push_back("tests::mack::ambiguous_class");
}

mack::options::test_option_type::~test_option_type()
{
}

std::vector<std::string> const&
mack::options::test_option_type::get_class_names() const
{
  return _class_names;
}

mack::options::options*
mack::options::test_option_type::get_options(
    std::string const& class_name) const // throws if no such value
{
  std::vector<option*> opts;
  std::string brief;
  std::string details;
  if (class_name.compare("tests::mack::options::recursive_class") == 0)
  {
    brief = "Brief: Recursive Class.";
    details = "Details: tests::mack::options::recursive_class";
    {
      opts.push_back(new type_option(
            "s",
            "standard",
            "A brief description: standard type.",
            "A detailed description: standard type.",
            false,
            new test_option_type()
            ));
    }
    {
      opts.push_back(new type_option(
            "d",
            "default",
            "A brief description: default type.",
            "A detailed description: default type.",
            false,
            new test_option_type(),
            "Tests::Mack::Options::Ambiguous_Class"
            ));
    }
    {
      opts.push_back(new type_option(
            "i",
            "initial",
            "A brief description: initial type.",
            "A detailed description: initial type.",
            false,
            new test_option_type(),
            "Recursive_Class",
            "Tests::Mack::Options::Ambiguous_Class"
            ));
    }
  }
  else if (class_name.compare("tests::mack::options::ambiguous_class") == 0)
  {
    brief = "Brief: Ambiguous Class.";
    details = "Details: tests::mack::options::ambiguous_class";
    {
      std::vector<std::string> values;
      values.push_back("option 1");
      values.push_back("option 2");
      values.push_back("option 3");
      opts.push_back(new selection_option(
            "s",
            "standard",
            "A brief description: standard selection.",
            "A detailed description: standard selection.",
            values
            ));
    }
    {
      std::vector<std::string> values;
      values.push_back("option 1");
      values.push_back("option 2");
      values.push_back("option 3");
      opts.push_back(new selection_option(
            "d",
            "default",
            "A brief description: default selection.",
            "A detailed description: default selection.",
            values,
            "option 1"
            ));
    }
    {
      std::vector<std::string> values;
      values.push_back("option 1");
      values.push_back("option 2");
      values.push_back("option 3");
      opts.push_back(new selection_option(
            "i",
            "initial",
            "A brief description: initial selection.",
            "A detailed description: initial selection.",
            values,
            "option 2",
            "option 1"
            ));
    }
    {
      opts.push_back(new option_switch(
            "b",
            "boolean",
            "A brief description: switch.",
            "A detailed description: switch."
            ));
    }
  }
  else if (class_name.compare("tests::mack::options::empty_class") == 0)
  {
    brief = "Brief: Empty Class.";
    details = "Details: tests::mack::options::empty_class";
    {
    }
  }
  else if (class_name.compare("tests::mack::ambiguous_class") == 0)
  {
    brief = "Brief: Ambiguous Class.";
    details = "Details: tests::mack::ambiguous_class";
    {
      opts.push_back(new option(
            "s",
            "standard",
            "A brief description: standard.",
            "A detailed description: standard."
            ));
      opts.push_back(new option(
            "d",
            "default",
            "A brief description: default.",
            "A detailed description: default.",
            "default value"
            ));
      opts.push_back(new option(
            "i",
            "initial",
            "A brief description: initial.",
            "A detailed description: initial.",
            "default value",
            "initial value"
            ));
    }
  }
  else
  {
    BOOST_THROW_EXCEPTION(invalid_value_error()
        << errinfo_option_value(class_name));
  }
  return new options(brief, details, opts);
}

mack::options::value*
mack::options::test_option_type::create_value(
    std::string const& class_name,
    mack::options::values const* values,
    const bool is_template_value) const
{
  if (class_name.compare("tests::mack::options::recursive_class") == 0)
  {
    return new value((tests::mack::base_class*)
        new tests::mack::options::recursive_class(
          values),
        "tests::mack::options::recursive_class");
  }
  else if (class_name.compare("tests::mack::options::ambiguous_class") == 0)
  {
    return new value((tests::mack::base_class*)
        new tests::mack::options::ambiguous_class(
          values),
        "tests::mack::options::ambiguous_class");
  }
  else if (class_name.compare("tests::mack::options::empty_class") == 0)
  {
    return new value((tests::mack::base_class*)
        new tests::mack::options::empty_class(),
        "tests::mack::options::empty_class");
  }
  else if (class_name.compare("tests::mack::ambiguous_class") == 0)
  {
    return new value((tests::mack::base_class*)
        new tests::mack::ambiguous_class(
          values),
        "tests::mack::ambiguous_class");
  }
  else
  {
    BOOST_THROW_EXCEPTION(invalid_value_error()
        << errinfo_option_value(class_name));
  }
}

