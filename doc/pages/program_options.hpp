/**
 * \page page_program_options Program Options
 * Mack the Knife features a documentation-based system for the
 * definition of program options. Parameters for classes or
 * programs can be defined within the
 * [doxygen](http://www.stack.nl/~dimitri/doxygen/index.html)
 * documentation block of the class or the program. These
 * options are then automatically available in the command line
 * interface (or any other interface).
 *
 * \tableofcontents
 *
 * \section page_program_options_overview Overview
 * Mack the Knife aims at enabling it's users to easily adjust
 * the methods for cracking (\ref crackers) to the task at hand.
 * We believe that class inheritance is the right way to
 * achieve this flexibility. Furthermore, we decided to offer
 * programmers a simple way to integrate this flexibility into
 * their own classes and programs. This allows the classes to be
 * easily interchanged and recombined with other classes just by
 * adjusting command line parameters.
 *
 * \section page_program_options_flags Setting Program Options
 * Program options can be specified via the command line
 * parameters of Mack the Knife programs. The same method is
 * employed for the main program (\ref mack_the_knife) as well
 * as for other programs within the framework.
 *
 * Options are addressed by using **flags**. These flags are
 * single letters (*short flags*) and words (*long flags*)
 * which are specified for each option. By using the flag,
 * a *value* can be assigned to the corresponding option.
 *
 * In the Mack the Knife framework, each option can have exactly
 * one **value** which is assigned to it. However, for some options
 * this value is interpreted as the name of the class to be
 * used. For example, the method for cracking employed be the
 * \ref mack_the_knife program can be adjusted by setting
 * the <tt>cracker</tt> parameter (e.g., to %brute_cracker).
 * Most %crackers, however, have their own parameters,
 * which are set similiarly. In order to avoid ambiguity,
 * it is recommended to specify the *parent* option as well
 * (e.g., <tt>cracker.keytable</tt> for specifying the
 * keytable to be used by the %brute_cracker).
 * Values can be assigned to options from the command line by
 * using one of the following patterns:
 *     * <tt>-<i>flag</i> <i>value</i></tt>
 *     * <tt>-<i>flag</i>=<i>value</i></tt>
 *     * <tt>-\-<i>flag</i> <i>value</i></tt>
 *     * <tt>-\-<i>flag</i>=<i>value</i></tt>
 *
 * Some options have a **default** and an **initial** value, which is
 * often the same. The default value can be assigned to an option
 * if no value is specified:
 *     * <tt>-<i>flag</i></tt>
 *     * <tt>-\-<i>flag</i></tt>
 *
 * An important type of options where the default value is not
 * identical to the initial value is the *option switch*.
 * Switches are used for parameters which can only be *true* or
 * *false*. Initially, switches are set to *false*. The default
 * value, however, is *true*. Thus, it is possible to set a
 * switch to *true* by only specifying the flag. To revert a
 * switch to the initial setting, <tt>-<i>flag</i>=false</tt> is
 * required.
 *
 * For convenience, each Mack the Knife program supports the
 * <tt>config</tt> option which allows the user to specify a
 * **configuration file**. In configuration files, values can be
 * specified for options with one assignment per line:
 *     * <tt><i>flag</i></tt>
 *     * <tt><i>flag</i> = <i>value</i></tt>
 *
 * This is useful for saving often used configurations. Additionally,
 * these files can be commented. Starting from a <tt>#</tt> character,
 * everything to the end of the line is ignored. Configuration files
 * are handled by replacing the <tt>-\-config=<i>file</i></tt> assignment
 * with the assignments found in the configuration file.
 *
 * If an error has been made in specifying the values, the program
 * stops and an error message is displayed. It is, however, also possible
 * to specify an *option handler* via the <tt>interactive</tt> option
 * (which is also provided for each program). These handlers allow the
 * user to adjust the current configuration in a more convenient way.
 * The available handlers are listed here: \ref option_handlers.
 *
 * \section page_program_options_own Specifying Own Classes
 * One strength of Mack the Knife is its extensibility. If a programmer
 * wants to implement an own cracker, algorithm, etc., the following
 * steps have to be taken:
 *     * Implement the class as usual, possibly extending a specific
 *       base class
 *     * Add the <tt>\@is_of_type{<i>typename</i>}</tt> annotation to
 *       the documentation block of the class
 *     * Specify the options for the class in the documentation block
 *     * Provide a constructor which takes a 
 *       <tt>%mack::options::values const*</tt> as only parameter
 *       (see \ref mack::options::values)
 *
 * We refer to cracker, algorithm, etc. as **types**. For each type,
 * a set of linked classes are specified. Some options do not take
 * simple values (strings, integers, etc.), but require more
 * complicated values (e.g., a method for cracking passwords).
 * All classes linked to the corresponding type can be assigned to
 * such an option. Furthermore, they are all listed up on the
 * corresponding page of \ref option_types.
 *
 * The above listed steps will now be explained in detail.
 *
 * \subsection page_program_options_own_implementation Implementation
 * Most types specify a **type class**, which all linked classes have
 * to extend. For example, all classes of type \ref crackers have to
 * extend \ref mack::core::Cracker. Check the documentation of the
 * type to make sure which class has to be extended.
 *
 * We will explain the single possibilities for specifying options by
 * using the (hypothetical) example of the *Cheesy Cracker*.
 *
 * \subsection page_program_options_own_documentation Documentation
 * In order to specify the type of the class, the documentation has
 * to contain the <tt>\@is_of_type{<i>type name</i>}</tt> doxygen command.
 *
 * An example is shown below:
 * \verbinclude is_type_for.hpp
 *
 * If the class has own parameters, they have to be specified in the
 * documentation block. **Option switches** (see above) can be specified
 * by using the form:
 * <tt>\@%option_switch{<i>short flag</i>,<i>long flag</i>} <i>description</i></tt>
 * \verbinclude option_switch.hpp
 * This will allow the user to set the cracker to a *fast mode* by specifying
 * <tt>-\-fast</tt> or <tt>-f</tt> as command line arguments when starting the
 * program. If, however, the program is started with the predefined
 * <tt>-\-help</tt> (or <tt>-h</tt>) switch, then the description of the option
 * will be shown to the user.
 * Additionally, this will create output for the doxygen documentation which
 * looks like this:
 * <blockquote>
 * @option_switch{f,fast} If this option is used, the cracker
 * will run in a faster but more inaccurate mode.
 * </blockquote>
 *
 * **Options** which take string literals as values
 * are specified by using the <tt>\@option{<i>short flag</i>,<i>long
 * flag</i>,<i>default value</i>} <i>description</i></tt> command. This will
 * create an option that can be specified by the user, by does not have to.
 * If it is not specified, the *default value* will be used instead.
 * Alternatively, the default value can be omitted. In this case, the option
 * will be mandatory; therefore, it will result in an error if no value is
 * provided by the user.
 *
 * For example, consider the cheesy cracker runs multiple rounds (whatever
 * it does in these rounds...), which can be adjusted by the user. Also,
 * special rules for cracking are read from a file which is called a
 * *cheese file*. The cheesy cracker absolutely requires such a file for
 * cracking. Therefore, the code or documentation could be further enhanced like
 * this:
 * \verbinclude option.hpp
 * Which will result in this doxygen documentation:
 * <blockquote>
 * @option{r,rounds,42} This option can be employed in order to
 * specify the maximum number of rounds the cracker tries to
 * find a key.
 * @option{c,cheese_file} This option specifies the cheese file
 * which the cracker should use.
 * </blockquote>
 *
 * **Type options** can be specified in a similar manner as normal options.
 * Of course, the type of possible classes has to specified additionally
 * as a further parameter:
 * \verbinclude type_option.hpp
 * Similar to above, the option will become mandatory if no default value
 * (which is the fourth parameter in this case) is provided.
 * The documentation generated by doxygen will then look like this:
 * <blockquote>
 * @type_option{a,algorithm,algorithms,MD5_Algorithm}
 * This option can be employed in order to specify the algorithm
 * which is used by the cracker.
 * </blockquote>
 *
 * If, however, the option should be used in order to specify a type which is
 * a template parameter of the class, it is not suitable to use \ref mack::options::values in
 * a constructor, since the template parameter has to be set in advance.
 * In this case, a variant of the type option which is called **template option**
 * can be employed instead. It is specified in the same way as a type option,
 * but additionally takes the template parameter as its *first* argument:
 * \verbinclude template_option.hpp
 * This will not change the documentation produced by doxygen, but only signal
 * what parameter is to be set. Again, a variant of the command which only
 * takes 4 parameters but sets the option to be mandatory exists also.
 *
 * Note that there is currently a restriction on the usage of templates in
 * type options. Namely, template types can not be assigned to template
 * options. Thus, it is currently not possible to create an object with a
 * templated template parameter (e.g.,
 * <tt>first_template_class&lt;second_template_class&lt;some_class&gt;
 * &gt;</tt>).
 *
 * Also note that flags have to have a certain format. They may only contain
 * letters, numbers, dashes and underscores. The first character, however,
 * may only be a letter or a number.
 *
 * If you ever need to declare your **own types**, you can do so by using the
 * <tt>\@%option_type{<i>tag</i>,<i>title</i>} <i>type_description</i></tt>
 * command. The <i>tag</i> is the string that is used to reference the type
 * (e.g., <tt>crackers</tt>). The <i>title</i> and the <i>type_description</i>
 * will be shown in the doxygen documentation. Please note, that the type
 * description can also contain further doxygen annotations like \@brief,
 * \@details or \@author and does not end after a blank comment line.
 * For each type, an entry is created on the \ref option_types page.
 *
 * Finally, you can also employ the <tt>\@%option_type_class{<i>class</i>}</tt>
 * command within the type description of a type. In this case, each member
 * of this type has to extend the specified class. On the other hand, this
 * enables you to use this class as an abstract class in your code.
 * Generally, types without such a specified type class can only be used
 * within *template options*.
 *
 * \subsection page_program_options_own_constructor Constructor
 * If the class has no options specified for it, the default constructor (no
 * parameters) is employed for creating instances of this class. If, however,
 * options are specified, a constructor which takes a
 * <tt>%mack::options::values const*</tt> as only parameter is required.
 *
 * The values, which are set by the user, can then be queried by using the
 * **get** methods of the \ref mack::options::values class. For *switches*,
 * <tt>get_boolean(<i>flag</i>)</tt> can be employed, where <i>flag</i>
 * is the long flag of the option. For string *options*,
 * <tt>get(<i>flag</i>)</tt> gives the desired value.
 * If the string has to be converted,
 * <tt>cast\<<i>type</i>\>(<i>flag</i>)</tt> can be used with <i>type</i> being
 * the desired simple type (e.g, *int*, *char* or *float*).
 * For **type options**, the <tt>get\<<i>type</i>\>(<i>flag</i>)</tt> method is
 * available. This method returns a pointer of the specified complex type.
 * For example, <tt>get\<mack::core::Cracker\>("cracker")</tt> returns a pointer
 * to the cracker (flag "cracker") as it was configured by the user. If, however,
 * the corresponding option is a **template option**, the concrete class has to
 * be specified (e.g.,
 * <tt>ALGORITHM* algorithm = values-\>get\<ALGORITHM\>("algorithm")</tt>).
 * In the case of type options (including template options), the class with the
 * option has to take care of deleting the instance.
 *
 * \section page_program_options_programs Own Programs
 * Information on how to create own programs can be found in \ref programs_group and in the
 * documentation of \ref mack::options::parser.
 *
 * \section page_program_options_quickstart Code Examples
 * \subsection page_program_options_quickstart_types_and_options Defining Types and Options
 * Defining types:
 * \verbinclude quickstart_types.hpp
 * Declaring some classes:
 * \verbinclude quickstart_classes.hpp
 * An usage example:
 * \verbinclude quickstart.hpp
 * More information on the usage of \ref mack::options::values can be found
 * there.
 * \subsection page_program_options_quickstart_program A *Mack the Knife* Program
 * \verbinclude quickstart_program.hpp
 * Finally, some examples for the usage of *my_program* on the command line
 * <blockquote>
 * <tt>\> bin/my_program -\-help</tt><br/>
 * Displays help on the program.
 *
 * <tt>\> bin/my_program -C "my_namespace::my_class" -\-help</tt><br/>
 * Displays help on the program and on parameters of
 * <tt>my_namespace::my_class</tt>.
 *
 * <tt>\> bin/my_program -C "my_namespace::my_class" -C.s -\-class.m "value"
 * -\-C.mandatory_type_option "my_extended_class" -\-class.mandatory_template="my_other_class"</tt><br/>
 * </blockquote>
 *
 * \section page_program_options_implementation Implementation
 * <b>Note:</b><br />
 * This section is provides information on the implementation of the program
 * options framework. It is only useful if you want to gain a deeper
 * understanding on how things work, for example in order to extend this
 * functionality. It is not necessary if you want to simply use the programs or
 * write new types.
 *
 * The program options are parsed from the documentation in a first step (\ref page_program_options_implementation_doc_parser)
 * and then assigned based on the command line parameters in a second step
 * (\ref page_program_options_implementation_options).
 *
 * \subsection page_program_options_implementation_doc_parser Documentation Parser
 * The options are generated in three steps. First, the annotations are parsed
 * from the source code using *Doxygen*. This creates the documentation as well
 * as an XML enriched output of the parsed code. Second, this XML output is
 * parsed again using a variety of XML parsers. The result of this step are
 * instances of structs which contain the necessary information about types,
 * classes, options and programs. Third, the type data (which contains class
 * and option data) and the programs data are written to source files.
 *
 * These steps are executed on the invocation of <tt>make update</tt>. The
 * required source code resides (mainly) in the <tt>configuration/src</tt>
 * directory and is compiled up front, if necessary. Note that if <tt>make
 * update</tt> is run repeatedly, the generated files will only be overwritten
 * if their content has changed. This avoids recompilation of the sources.
 *
 * For each found type, a new pair of files is created in the
 * <tt>src/mack/options/types</tt> directory. First, a header file
 * (<tt>.hpp</tt>) with the name of the type (e.g., %crackers.hpp). In it,
 * a class is declared (with the same name, e.g., %crackers) which extends
 * \ref mack::options::option_type. The implementation is provided in the
 * generated source file (either <tt>.cu</tt> or <tt>.cpp</tt> depending on
 * if CUDA is required by the included classes or not). The sources will
 * be compiled automatically.
 *
 * For each program, however, an entry is created within the
 * <tt>src/mack/options/programs.cpp</tt> source file. If a program is
 * requested, the corresponding \ref mack::options::program_options are
 * created and returned.
 *
 * \image html doc_parser.png "Parsing program options"
 * \image latex doc_parser.eps "Parsing program options" width=.9\textwidth
 *
 * \subsection page_program_options_implementation_options Options and Values
 * The parsed program options can then be used in conjunction with command line
 * parameters and further user input. To simplify the usage, the
 * \ref mack::options::parser class is provided as an interface. It takes the
 * command line parameters and the program name and takes care of assigning
 * values to the program options (see
 * \ref page_program_options_quickstart_program for an usage example).
 *
 * The command line parameters are used to set the program options. Optionally,
 * an \ref mack::options::handlers::handler can be used to provide further
 * settings. If type options (defined by the corresponding class in
 * <tt>src/mack/options/types/</tt>) are set, new options are created which
 * can then again be set. Finally, the assignments to the options are
 * transformed into values (see \ref mack::options::values) which can then be
 * used within the program.
 *
 * In the same way, the logging facility is configured using default program
 * options (see \ref page_logging).
 *
 * Note that, depending on the option handler, multiple runs per program are
 * possible.
 *
 * \image html options.png "Setting program options"
 * \image latex options.eps "Setting program options" width=.9\textwidth
 *
 */

