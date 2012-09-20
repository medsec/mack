namespace my_namespace {
  /**
   * ...
   * @is_of_type{my_class_type}
   * This class is of type *my_class_type*; therefore, it
   * has to extend the type class of this type.
   */
  class my_extended_class : public my_class_type_class
  {
    ...
  };

  /**
   * ...
   * @is_of_type{my_no_class_type}
   * *my_no_class_type* does not have a type class.
   */
  class my_other_class
  {
    ...
  };
} // namespace my_namespace
