/**
 * \page page_quickstart_user Quickstart User Guide
 *
 *	This page contains some examples for using the various cracker of Mack. Keep in mind that you always have to specify
 *	a target <b>-t target_file</b> and an algorithm <b>-a algo</b>.
 *
 *	\section quick_brute Brute Force Cracker
 *
 *  This cracker tries every possible combination of words to crack its targets. It is limited by the length of the word and
 *  the possible values for each character. Try to use short word length and small character tables at first. Long words
 *  with every possible character will take an eternity, even on the GPU.
 *
 *	What you need:
 *	<p>
 *	The Brute Force Cracker is specified with <b>-C Brute_Cracker</b> and always needs a word length given by <b>-\-cracker.length</b>.
 *	You also need to specify a char table which contains all possible values that a character in your word can have.
 *	There are already some char tables provided for you in <i>resources/char_tables</i>, but you can also create you own
 *	by using our little tool: <b>Keymap_Reader</b>.
 *	<p>
 *	* -C Brute_Cracker
 *	* -\-cracker.length \<word length\>
 *  * -cracker.keytable \<path to key table\>
 *
 *	Example:
 	\verbatim
 	./mack -C Brute_Cracker --cracker.length 3 --cracker.keytable ../../src/keys/char_tables/visible_ascii.txt -a md5 -t targets/md5_3_hash
 	\endverbatim

 *
 *  \section quick_dict Dictionary Cracker
 *
 *	This cracker tries to find passwords by hashing through a dictionary. It also has features to increase the performance by using more
 *	memory and can make your dictionary more useful by alternating Characters and/or appending Numbers.
 *
 *	What you need:
 *	<p>
 *	The Dictionary Cracker is specified with <b>-C Dictionary_Cracker</b> and always needs a dictionary given by <b>-D my_dict</b>
 *	<p>
 *	* -C Dictionary_Cracker
 *	* -D my_dict
 *
 *	Additional Features:
 *	<p>
 *	Some parameters can increase your success rate in cracking a password. To alternate the first character of each word use <b>-x</b>.
 *	To let Mack use more or less of your RAM use <b>-m</b> (Standard 2048 MB). To append a n-digit long number use <b>-n</b>.
 *	<p>
 *	* -n \<digits to append after each Word\>
 *	* -m \<max memory used in MB\>
 *	* -x
 *
 *	Example:
 *
 	\verbatim
 	mack -C Dictionary_Cracker -D dict/de.dict -n 2 -a md5 -t=Default_Loader -cracker.target-file tests/resources/targets/dict_rand_hash -o=container_callback
 	\endverbatim

 *
 *  \section quick_Rainbow Rainbow Table Cracker
 *
 *	It could be useful to inform yourself about rainbow tables before using this cracker \a http://en.wikipedia.org/wiki/Rainbow_table.
 *	This cracker creates and uses rainbow tables to crack its targets. If no rainbow table is available for the given parameters,
 *	the cracker will ask you to create one. Remember creating such a table can take a long time and could use lots of memory, so be warned.
 *
 * 	What you need:
 *	<p>
 *	The Rainbow Cracker is specified with <b>-C Rainbow_Cracker</b> and always needs a word length given by <b>-\-cracker.length</b>.
 *	Similar to the Brute Force Cracker you need to specify a char table which contains all possible values that a character in your word can have.
 *	There are already some char tables provided for you in <i>resources/char_tables</i>, but you can also create you own
 *	by using our little tool: <b>Keymap_Reader</b>.
 *	<p>
 *	* -C Rainbow_Cracker
 *	* -\-cracker.length \<word length\>
 *  * -cracker.keytable \<path to key table\>
 *
 * 	Additional Features:
 *	<p>
 *	You can specify the chain length of the chains in the table to adjust the time-memory tradeoff (Standard 32).
 *	<p>
 *	* -\-cracker.chainlength \<length\>
 *
 *  Example:
 	\verbatim
 	mack -c Rainbow_Cracker --cracker.keytable visible_german --cracker.length 7 --cracker.chainlength 1024 -a md5 -t ../targets/targets
 	\endverbatim

 *
 */
