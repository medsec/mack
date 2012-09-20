#include "location_parser.hpp"

mack::options::doc_parser::location_parser::location_parser()
  : _target_class_id(),
    _stack(),
    _location()
{
}

mack::options::doc_parser::location_parser::~location_parser()
{
}

void
mack::options::doc_parser::location_parser::set_target_class_id(
    std::string const& target_class_id)
{
  _target_class_id = target_class_id;
}

void
mack::options::doc_parser::location_parser::clear()
{
  _location.erase();
  _stack.clear();
}

std::string const&
mack::options::doc_parser::location_parser::get_location() const
{
  return _location;
}

void
mack::options::doc_parser::location_parser::start_element(
    std::string const& name,
    mack::core::attributes_type const& attributes)
{
  if (_stack.empty())
  {
    if (name == "compounddef"
      && attributes.find("id")->second == _target_class_id)
    {
      _stack.push_back(name);
    }
  }
  else
  {
    if (_stack.size() == 1 && name == "location")
    {
      _location = attributes.find("file")->second;
    }
    _stack.push_back(name);
  }
}

void
mack::options::doc_parser::location_parser::end_element(
    std::string const& name)
{
  if (!_stack.empty())
  {
    _stack.pop_back();
  }
}

void
mack::options::doc_parser::location_parser::characters(
    std::string const& text)
{
}

