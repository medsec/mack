#ifndef __TESTS_MACK_OPTIONS_TYPES_TEST_OPTION_TYPE_HPP__
#define __TESTS_MACK_OPTIONS_TYPES_TEST_OPTION_TYPE_HPP__

#include <mack/options/option_type.hpp>
#include <mack/options/value.hpp>
#include <mack/options/values.hpp>
#include <mack/options/options.hpp>

#include <string>

namespace mack {
namespace options {

class test_option_type : public option_type
{

  public:

    test_option_type();

    virtual
    ~test_option_type();

    virtual
    std::vector<std::string> const&
    get_class_names() const;

    virtual
    options*
    get_options(
        std::string const& class_name) const;

    virtual
    value*
    create_value(
        std::string const& class_name,
        values const* values,
        const bool is_template_value) const;

  private:

    std::vector<std::string> _class_names;

}; // class test_option_type

} // namespace options
} // namespace mack

namespace tests {
namespace mack {

class base_class {
  public:
    base_class()
    {
    }
};

namespace options {

class recursive_class : public base_class {
  public:
    recursive_class(
        ::mack::options::values const* values)
    {
      delete values->get<base_class>("standard");
      delete values->get<base_class>("default");
      delete values->get<base_class>("initial");
    }
};

class ambiguous_class : public base_class {
  public:
    ambiguous_class(
        ::mack::options::values const* values)
    {
      values->get("standard");
      values->get("default");
      values->get("initial");
      values->get_boolean("boolean");
    }
};

class empty_class : public base_class {
  public:
    empty_class()
    {
    }
};

}

class ambiguous_class : public base_class {
  public:
    ambiguous_class(
        ::mack::options::values const* values)
    {
      values->get("standard");
      values->get("default");
      values->get("initial");
    }
};

}
}

#endif /* __TESTS_MACK_OPTIONS_TYPES_TEST_OPTION_TYPE_HPP__ */

