/**
 * \mainpage Mack the Knife Documentation
 *
 *	\section what_is What is Mack the Knife?
 *	<p>
 *	Mack the Knife is a modular password cracking framework which relies on heavy parallel computing on the GPU
 *	using Nvidia's CUDA technology. The name <b>Mack the Knife</b> comes from the protagonist from Bertolt Brecht's "The Threepenny Opera"
 *	and is supposed to be a reference to the well known password cracker John the Ripper.
 *	</p>
 *
 *	\subsection why_another Why another password cracker?
 *	Using the GPU for massive parallel computation of hash algorithms is alot faster than using the CPU (even with multiple cores).
 *
 *  <p>
 *  Existing Frameworks:
 *  </p>
 *  	* have limited support for GPU computing (John the Ripper)
 *  	* are not Open Source (Hashcat)
 *  	* have a rigid and/or old structure (John the Ripper)
 *
 *	<p>
 *	Mack is written in C++ and tries to provide an Object Oriented Framwork which can easily expanded with
 *	new Cracking Methods and cryptographic algorithms. If someone implements a new algorithm for Mack, it can instantly
 *	be used by all Cracking Methods provided, for instance creating rainbow tables, Dictionary or Bruteforce attacks.
 *	The other way around: if a new cracking Method is implemented it will work with every cryptographic algorithm available in
 *	Mack.
 *  </p>
 *
 *  Keep in mind that Mack is still in its infancy and still does not have all the features other password crackers provide.
 *  But maybe you are willing to implement some of them and help Mack the Knife mature.
 *
 *  \section requirements Requirements
 *
 * 		* Linux
 * 		* Cuda 5 (currently beta)
 * 		* gcc-4.6 / g++-4.6
 * 		* Nvidia Cuda capable graphics card with compute capability >=2.0
 *
 *	Libraries:
 *		*	expat
 *		* boost (algorithms, filesystem, lexical_cast, any, exception)
 *		*	Doxygen (>= 1.8.1.1)
 *		*	Unittest++ (for Testing)
 *		*	Python (optional for generating new test targets)
 *
 *  \section where_start Where to start?
 *
 *  \subsection sub_using Using Mack the Knife
 *  <p>\ref page_crackers_and_algorithms</p>
 *  <p>\ref page_quickstart_user</p>
 *
 *  \subsection sub_develop Develop for Mack the Knife
 *	<p>\ref page_todo</p>
 *	<p>\ref page_program_options</p>
 *	<p>\ref page_logging</p>
 */
