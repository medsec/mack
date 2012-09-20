/**
 * \page page_crackers_and_algorithms Mack the Knife Concepts
 * The *Mack the Knife* framework aims at providing methods for cracking
 * passwords or deciphering ciphertext. Messages can be encrypted using
 * ciphers (which leads to the ciphertext) or hashed using hash %algorithms
 * (which leads to a hash) in order to hide them from unauthorized views.
 * However, there are always ways to gain this unauthorized access.
 * Often, these ways are not feasible. To test if they are is the task
 * *Mack the Knife* is developed for.
 *
 * In our terminology, we refer to ciphers and hash %algorithms both as
 * **algorithms**. The ciphertext or hash is called **target**, since we
 * assume this is what the attacker has.
 * A core concept of *Mack the Knife* is the separation of implementations of
 * %algorithms and general methods of cracking these (although it is possible to
 * implement methods specific for certain %algorithms). These methods are called
 * **crackers**. For example, the brute_cracker implements a complete (brute
 * force) search over the space of possible input messages.
 *
 * The main program of the framework (\ref mack_the_knife) therefore is build
 * on these ideas. Targets can be loaded from different sources by using the
 * appropriate **target loader** (see \ref targetloaders). A %cracker (see
 * \ref crackers) is chosen in order to specify the strategy for cracking the
 * targets. In most cases, such a %cracker tries to use the specified
 * %algorithm (see \ref algorithms) to reproduce the target with chosen
 * messages. If a target was reproduced, it is given to a callback (see
 * \ref callbacks) which handles it appropriately (e.g., prints the original
 * message).
 *
 * \image html mack.png "Classes for cracking targets"
 * \image latex mack.eps "Classes for cracking targets" width=.9\textwidth
 */
