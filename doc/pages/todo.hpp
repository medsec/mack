/**
 * \page page_todo ToDo List
 *
 *	Mack the Knife in its current version runs very fine and without any known bugs. Nevertheless there are some features to implement.
 *
 *	\section known_bugs Known Bugs
 *	* no known bugs
 *
 *  \section additional_algorithms Additional Algorithms
 *  Some standard hash algorithms should be added to Mack the Knife, for example:
 *  <p>
 *  * MD4
 *  * SHA1
 *  * SHA512
 *  * ...
 *  </p>
 *
 *	\section featurelist Additional Features
 *  <p>
 *
 *  * <b>Logging everywhere</b>
 *  <div style="padding-left:50px">Because of the last minute feature "Logging", we are able to handle loggs using mack. This should be included into all classes.</div>
 *	</p>
 *
 *  * <b>Rainbow Cracker Tests</b>
 *  <div style="padding-left:50px">to test the correctness of the Rainbow Cracker some UnitTest tests are needed</div>
 *
 *	* <b>segmentation of the targets</b>
 *	<div style="padding-left:50px">to handle more targets than the graphics cards memory can hold</div>
 *
 *	* <b>Configurable algorithms within crackers</b>
 *	<div style="padding-left:50px">currently, crackers use not the algorithm which is created from the options, but use a default constructed algorithm instead.
 *	At the moment, this is no problem since algorithms don't have any options. However, this may change</div>
 *
 *	* <b>Use Logging</b>
 *	<div style="padding-left:50px">currently, the logging facility (see \ref page_logging) is not used within crackers,
 *	but messages are printed to standard streams</div>
 *
 *	* <b>Hybrid Cracker</b>
 *	<div style="padding-left:50px">uses a pattern and a word of the dictionary as input: allow the user to input a pattern how the message of the hash can look like</div>
 *
 *	* <b>Markov Chains</b>
 *	<div style="padding-left:50px">implementation of a cracker using Markov-chains to improve the performance of a brute force cracker</div>
 *
 *	* <b>Logger</b>
 *  <div style="padding-left:50px">first elements of a logger class are preimplemented, it must be finished</div>
 *
 *  * <b>Usage of more than one graphics cards</b>
 *  <div style="padding-left:50px">if the system provides multiple graphics cards, mack the knife should use them</div>
 *
 *  * <b>GUI</b>
 *  <div style="padding-left:50px">develop a GUI front-end for Mack the Knife to help users with less command line knowledge</div>
 *
 *  * <b>Windows port</b>
 *  <div style="padding-left:50px">port Mack the Knife to Windows, this may will be the most difficult part, because we use some Linux only techniques like mmap</div>
 *	</p>
 *
 *	* <b>Clean Things Up</b>
 *	<div style="padding-left:50px">Should class names start with an upper case letter? Is the directory structure good?
 *	Should Callbacks/Algorithms/Target Loaders be options of the cracker or the program?</div>
 *
 */
