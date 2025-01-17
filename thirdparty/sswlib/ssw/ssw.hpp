/**
 ** DRAGEN Open Source Software
 ** Copyright (c) 2019-2020 Illumina, Inc.
 ** All rights reserved.
 **
 ** Based on SSW implementation
 ** https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library
 ** Version 0.1.4
 ** Last revision by Mengyao Zhao on 07/19/16 <zhangmp@bc.edu>
 **
 ** License: MIT
 ** Copyright (c) 2012-2015 Boston College
 ** Copyright (c) 2020-2021 Illumina
 **
 ** Permission is hereby granted, free of charge, to any person obtaining a copy of this
 ** software and associated documentation files (the "Software"), to deal in the Software
 ** without restriction, including without limitation the rights to use, copy, modify,
 ** merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 ** permit persons to whom the Software is furnished to do so, subject to the following
 ** conditions:
 ** The above copyright notice and this permission notice shall be included in all copies
 ** or substantial portions of the Software.
 ** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 ** INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 ** PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 ** HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 ** OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 ** SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 **/

#ifndef SSW_H
#define SSW_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>


#define MAPSTR "MIDNSHP=X"
#define BAM_CIGAR_SHIFT 4u


struct alignment_end{
  uint16_t score;
  int32_t ref;   //0-based position
  int32_t read;    //alignment ending position on read, 0-based
};


/*!	@typedef	structure of the alignment result
	@field	score1	the best alignment score
	@field	score2	sub-optimal alignment score
	@field	ref_begin1	0-based best alignment beginning position on reference;	ref_begin1 = -1 when the best alignment beginning
						position is not available
	@field	ref_end1	0-based best alignment ending position on reference
	@field	read_begin1	0-based best alignment beginning position on read; read_begin1 = -1 when the best alignment beginning
						position is not available
	@field	read_end1	0-based best alignment ending position on read
	@field	read_end2	0-based sub-optimal alignment ending position on read
	@field	cigar	best alignment cigar; stored the same as that in BAM format, high 28 bits: length, low 4 bits: M/I/D (0/1/2);
					cigar = 0 when the best alignment path is not available
	@field	cigarLen	length of the cigar string; cigarLen = 0 when the best alignment path is not available
*/
struct s_align{
	~s_align();
	uint16_t score1;
	uint16_t score2;
	int32_t ref_begin1;
	int32_t ref_end1;
	int32_t	read_begin1;
	int32_t read_end1;
	int32_t ref_end2;
	uint32_t* cigar;
	int32_t cigarLen;
};

class s_profile{
    protected:
		s_profile(const int8_t* _read, const int32_t _readLen, const int8_t* _mat, const int32_t _n, const int32_t _bias);
		void init (const int8_t score_size);
    protected:
		virtual int8_t* profile_byte_init(const int8_t* read_num);
		virtual int16_t* profile_word_init(const int8_t* read_num);
		virtual int8_t* profile_byte_rev(const int8_t* read_num, const int32_t readLen);
		virtual int16_t* profile_word_rev(const int8_t* read_num, const int32_t readLen);

    protected:
		 /* Striped Smith-Waterman
		   Record the highest score of each reference position.
		   Return the alignment score and ending position of the best alignment, 2nd best alignment, etc.
		   Gap begin and gap extension are different.
		   wight_match > 0, all other weights < 0.
		   The returned positions are 0-based.
		 */
		virtual alignment_end* ssw_byte (const int8_t* ref,
        int8_t ref_dir,	// 0: forward ref; 1: reverse ref
        int32_t refLen,
        int32_t readLen,
        const uint8_t weight_gapO, /* will be used as - */
        const uint8_t weight_gapE, /* will be used as - */
		const int8_t* profile,
        uint8_t terminate,	/* the best alignment score: used to terminate
												       the matrix calculation when locating the
												       alignment beginning point. If this score
												       is set to 0, it will not be used */
        int32_t maskLen) = 0;

    virtual alignment_end* ssw_word(const int8_t* ref,
        int8_t ref_dir,	// 0: forward ref; 1: reverse ref
        int32_t refLen,
        int32_t readLen,
        const uint8_t weight_gapO, /* will be used as - */
        const uint8_t weight_gapE, /* will be used as - */
		const int16_t* profile,
        uint16_t terminate,	/* the best alignment score: used to terminate
												       the matrix calculation when locating the
												       alignment beginning point. If this score
												       is set to 0, it will not be used */
        int32_t maskLen) = 0;

      int8_t* _profile_byte;  // 0: none
      int16_t* _profile_word;  // 0: none
      const int8_t* _read;
      const int8_t* _mat;
      int32_t _readLen;
      int32_t _n;
      uint8_t _bias;

	  s_profile* _baseline;
  public:
/*!	@function	Create the query profile using the query sequence.
	@param	read	pointer to the query sequence; the query sequence needs to be numbers
	@param	readLen	length of the query sequence
	@param	mat	pointer to the substitution matrix; mat needs to be corresponding to the read sequence
	@param	n	the square root of the number of elements in mat (mat has n*n elements)
	@param	score_size	estimated Smith-Waterman score; if your estimated best alignment score is surely < 255 please set 0; if
						your estimated best alignment score >= 255, please set 1; if you don't know, please set 2
	@return	pointer to the query profile structure
	@note	example for parameter read and mat:
			If the query sequence is: ACGTATC, the sequence that read points to can be: 1234142
			Then if the penalty for match is 2 and for mismatch is -2, the substitution matrix of parameter mat will be:
			//A  C  G  T
			  2 -2 -2 -2 //A
			 -2  2 -2 -2 //C
			 -2 -2  2 -2 //G
			 -2 -2 -2  2 //T
			mat is the pointer to the array {2, -2, -2, -2, -2, 2, -2, -2, -2, -2, 2, -2, -2, -2, -2, 2}
*/
	static s_profile* create(const int8_t* _read, const int32_t _readLen,
    const int8_t* _mat, const int32_t _n, const int32_t _bias, const int8_t score_size);
	virtual ~s_profile();
// @function	ssw alignment.
/*!	@function	Do Striped Smith-Waterman alignment.
	@param	prof	pointer to the query profile structure
	@param	ref	pointer to the target sequence; the target sequence needs to be numbers and corresponding to the mat parameter of
				function ssw_init
	@param	refLen	length of the target sequence
	@param	weight_gapO	the absolute value of gap open penalty
	@param	weight_gapE	the absolute value of gap extension penalty
	@param	flag	bitwise FLAG; (from high to low) bit 5: when setted as 1, function ssw_align will return the best alignment
					beginning position; bit 6: when setted as 1, if (ref_end1 - ref_begin1 < filterd && read_end1 - read_begin1
					< filterd), (whatever bit 5 is setted) the function will return the best alignment beginning position and
					cigar; bit 7: when setted as 1, if the best alignment score >= filters, (whatever bit 5 is setted) the function
  					will return the best alignment beginning position and cigar; bit 8: when setted as 1, (whatever bit 5, 6 or 7 is
 					setted) the function will always return the best alignment beginning position and cigar. When flag == 0, only
					the optimal and sub-optimal scores and the optimal alignment ending position will be returned.
	@param	filters	score filter: when bit 7 of flag is setted as 1 and bit 8 is setted as 0, filters will be used (Please check the
 					decription of the flag parameter for detailed usage.)
	@param	filterd	distance filter: when bit 6 of flag is setted as 1 and bit 8 is setted as 0, filterd will be used (Please check
					the decription of the flag parameter for detailed usage.)
	@param	maskLen	The distance between the optimal and suboptimal alignment ending position >= maskLen. We suggest to use
					readLen/2, if you don't have special concerns. Note: maskLen has to be >= 15, otherwise this function will NOT
					return the suboptimal alignment information. Detailed description of maskLen: After locating the optimal
					alignment ending position, the suboptimal alignment score can be heuristically found by checking the second
					largest score in the array that contains the maximal score of each column of the SW matrix. In order to avoid
					picking the scores that belong to the alignments sharing the partial best alignment, SSW C library masks the
					reference loci nearby (mask length = maskLen) the best alignment ending position and locates the second largest
					score from the unmasked elements.
	@return	pointer to the alignment result structure
	@note	Whatever the parameter flag is setted, this function will at least return the optimal and sub-optimal alignment score,
			and the optimal alignment ending positions on target and query sequences. If both bit 6 and 7 of the flag are setted
			while bit 8 is not, the function will return cigar only when both criteria are fulfilled. All returned positions are
			0-based coordinate.
*/
s_align* align (
    const int8_t* ref,
    int32_t refLen,
    const uint8_t weight_gapO,
    const uint8_t weight_gapE,
    const uint8_t flag,
    const uint16_t filters,
    const int32_t filterd,
    const int32_t maskLen);
};

class s_profile_baseline : public s_profile
{
	friend class s_profile;

	protected:
	s_profile_baseline(const int8_t* _read, const int32_t _readLen,
		const int8_t* _mat, const int32_t _n, const int32_t _bias);

	protected:
	alignment_end* ssw_byte (const int8_t* ref,
		int8_t ref_dir,
		int32_t refLen,
		int32_t readLen,
		const uint8_t weight_gapO, 
		const uint8_t weight_gapE,
		const int8_t* profile,
		uint8_t terminate,
		int32_t maskLen);

	alignment_end* ssw_word(const int8_t* ref,
		int8_t ref_dir,
		int32_t refLen,
		int32_t readLen,
		const uint8_t weight_gapO,
		const uint8_t weight_gapE,
		const int16_t* profile,
		uint16_t terminate,
		int32_t maskLen);
};

class s_profile_sse2 : public s_profile
{
	friend class s_profile;

	protected:
	s_profile_sse2(const int8_t* _read, const int32_t _readLen,
		const int8_t* _mat, const int32_t _n, const int32_t _bias);

	protected:
	alignment_end* ssw_byte (const int8_t* ref,
		int8_t ref_dir,
		int32_t refLen,
		int32_t readLen,
		const uint8_t weight_gapO, 
		const uint8_t weight_gapE,
		const int8_t* profile,
		uint8_t terminate,
		int32_t maskLen);

	alignment_end* ssw_word(const int8_t* ref,
		int8_t ref_dir,
		int32_t refLen,
		int32_t readLen,
		const uint8_t weight_gapO,
		const uint8_t weight_gapE,
		const int16_t* profile,
		uint16_t terminate,
		int32_t maskLen);
};



int32_t ssw_get_bias (
    const int8_t* mat, const int32_t n);

/*! @function:
     1. Calculate the number of mismatches.
     2. Modify the cigar string:
         differentiate matches (=), mismatches(X), and softclip(S).
	@param	ref_begin1	0-based best alignment beginning position on the reference sequence
	@param	read_begin1	0-based best alignment beginning position on the read sequence
	@param	read_end1	0-based best alignment ending position on the read sequence
	@param	ref	pointer to the reference sequence
	@param	read	pointer to the read sequence
	@param	readLen	length of the read
	@param	cigar	best alignment cigar; stored the same as that in BAM format, high 28 bits: length, low 4 bits: M/I/D (0/1/2)
	@param	cigarLen	length of the cigar string
 	@return:
     The number of mismatches.
	 The cigar and cigarLen are modified.
*/
int32_t mark_mismatch (int32_t ref_begin1,
					   int32_t read_begin1,
					   int32_t read_end1,
					   const char* ref,
					   const char* read,
					   int32_t readLen,
					   uint32_t** cigar, 
					   int32_t* cigarLen);

/*!	@function		Produce CIGAR 32-bit unsigned integer from CIGAR operation and CIGAR length
	@param	length		length of CIGAR
	@param	op_letter	CIGAR operation character ('M', 'I', etc)
	@return			32-bit unsigned integer, representing encoded CIGAR operation and length
*/
uint32_t to_cigar_int (uint32_t length, char op_letter);

/*!	@function		Extract CIGAR operation character from CIGAR 32-bit unsigned integer
	@param	cigar_int	32-bit unsigned integer, representing encoded CIGAR operation and length
	@return			CIGAR operation character ('M', 'I', etc)
*/
//char cigar_int_to_op (uint32_t cigar_int);
static inline char cigar_int_to_op(uint32_t cigar_int) 
{
	return (cigar_int & 0xfU) > 8 ? 'M': MAPSTR[cigar_int & 0xfU];
}


/*!	@function		Extract length of a CIGAR operation from CIGAR 32-bit unsigned integer
	@param	cigar_int	32-bit unsigned integer, representing encoded CIGAR operation and length
	@return			length of CIGAR operation
*/
//uint32_t cigar_int_to_len (uint32_t cigar_int);
static inline uint32_t cigar_int_to_len (uint32_t cigar_int)
{
	return cigar_int >> BAM_CIGAR_SHIFT;
}

typedef struct {
  uint32_t* seq;
  int32_t length;
} cigar;

/*! @function   Compute banded SMW with a traceback and return CIGAR representation.
*/
cigar* banded_sw (
    const int8_t* ref,
    const int8_t* read,
    int32_t refLen,
    int32_t readLen,
    int32_t score,
    const uint32_t weight_gapO,  /* will be used as - */
    const uint32_t weight_gapE,  /* will be used as - */
    int32_t band_width,
    const int8_t* mat, /* pointer to the weight matrix */
    int32_t n,
    int32_t read_begin,
    int32_t unclippedReadLen
);

#endif	// SSW_H
