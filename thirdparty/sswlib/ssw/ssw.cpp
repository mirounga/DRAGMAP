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

// shared and sse2-specific implementations

#ifdef NDEBUG
#define NDEBUG_DEFINED
#undef NDEBUG
#endif

#include <cassert>

#ifdef NDEBUG_DEFINED
#define NDEBUG
#endif

#include <emmintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ssw.hpp"
#include "ssw_internal.hpp"

/* Generate query profile rearrange query sequence & calculate the weight of match/mismatch. */
int8_t* s_profile::profile_byte_init(const int8_t* read_num) {
  constexpr int32_t byteElems = 16;

  int32_t const segLen = (_readLen + byteElems - 1) / byteElems; /* Split the register into byteElems pieces.
								     Each piece is 8 bit.
   */

  int8_t* const profile = (int8_t*)_mm_malloc(_n * segLen * byteElems, 64);
  int8_t* t = profile;

  /* Generate query profile rearrange query sequence & calculate the weight of match/mismatch */
  for (int32_t nt = 0; LIKELY(nt < _n); nt ++) {
    for (int32_t i = 0; i < segLen; i ++) {
      int32_t j = i;
      for (int32_t segNum = 0; LIKELY(segNum < byteElems) ; segNum ++) {
        int8_t bonus = (j==0 || j == _readLen-1) ? UNCLIP_BONUS : 0;

        *t++ = (j >= _readLen) ? _bias : _mat[nt * _n + read_num[j]] + _bias + bonus;

        j += segLen;
      }
    }
  }

  return profile;
}

int16_t* s_profile::profile_word_init(const int8_t* read_num) { 
  constexpr int32_t wordElems = 8;

  int32_t const segLen = (_readLen + wordElems - 1) / wordElems;

  int16_t* const profile = (int16_t*)_mm_malloc(_n * segLen * 2 * wordElems, 64);
  int16_t* t = profile;

  /* Generate query profile rearrange query sequence & calculate the weight of match/mismatch */
  for (int32_t nt = 0; LIKELY(nt < _n); nt ++) {
    for (int32_t i = 0; i < segLen; i ++) {
      int32_t j = i;
      for (int32_t segNum = 0; LIKELY(segNum < wordElems) ; segNum ++) {
        int16_t bonus = (j==0 || j == _readLen-1) ? UNCLIP_BONUS : 0;

        *t++ = (j >= _readLen) ? 0 : _mat[nt * _n + read_num[j]] + bonus;
        j += segLen;
      }
    }
  }

  return profile;
}


int8_t* s_profile::profile_byte_rev(const int8_t* read_num, const int32_t readLen) {
  constexpr int32_t byteElems = 16;

  int32_t const segLen = (readLen + byteElems - 1) / byteElems;

  int8_t* const profile = (int8_t*)_mm_malloc(_n * segLen * byteElems, 64);
  int8_t* t = profile;

  for (int32_t nt = 0; LIKELY(nt < _n); nt ++) {
    for (int32_t i = 0; i < segLen; i ++) {
      int32_t j = i;
      for (int32_t segNum = 0; LIKELY(segNum < byteElems) ; segNum++) {
        int8_t bonus = (( j == readLen-1) || ( (readLen == _readLen) && (j==0))) ? UNCLIP_BONUS : 0;

        *t++ = (j >= readLen) ? _bias : _mat[nt * _n + read_num[j]] + _bias + bonus;

        j += segLen;
      }
    }
  }

  return profile;
}


int16_t* s_profile::profile_word_rev(const int8_t* read_num, const int32_t readLen) {
  constexpr int32_t wordElems = 8;

  int32_t const segLen = (readLen + wordElems - 1) / wordElems;

  int16_t* const profile = (int16_t*)_mm_malloc(_n * segLen * 2 * wordElems, 64);
  int16_t* t = profile;

  for (int32_t nt = 0; LIKELY(nt < _n); nt ++) {
    for (int32_t i = 0; i < segLen; i ++) {
      int32_t j = i;
      for (int32_t segNum = 0; LIKELY(segNum < wordElems) ; segNum ++) {
        int16_t bonus = (( j == readLen-1) || ( (readLen == _readLen) && (j==0))) ? UNCLIP_BONUS : 0;

        *t++ = (j >= readLen) ? 0 : _mat[nt * _n + read_num[j]] + bonus;

        j += segLen;
      }
    }
  }

  return profile;
}

static int8_t* seq_reverse(const int8_t* seq, int32_t end)	/* end is 0-based alignment ending position */
{
  int8_t* reverse = (int8_t*)calloc(end + 1, sizeof(int8_t));
  int32_t start = 0;
  while (LIKELY(start <= end)) {
    reverse[start] = seq[end];
    reverse[end] = seq[start];
    ++ start;
    -- end;
  }
  return reverse;
}

int32_t ssw_get_bias(const int8_t* mat, const int32_t n) {
  int32_t bias = 0, i;
  for (i = 0; i < n*n; i++) {
    if (mat[i] < bias) {
      bias = mat[i];
    }
  }
  bias = abs(bias);
  return bias;
}

/*!     @function               Produce CIGAR 32-bit unsigned integer from CIGAR operation and CIGAR length
        @param  length          length of CIGAR
        @param  op_letter       CIGAR operation character ('M', 'I', etc)
        @return                 32-bit unsigned integer, representing encoded CIGAR operation and length
 */
uint32_t to_cigar_int (uint32_t length, char op_letter)
{
  switch (op_letter) {
  case 'M': /* alignment match (can be a sequence match or mismatch */
  default:
    return length << BAM_CIGAR_SHIFT;
  case 'S': /* soft clipping (clipped sequences present in SEQ) */
    return (length << BAM_CIGAR_SHIFT) | (4u);
  case 'D': /* deletion from the reference */
    return (length << BAM_CIGAR_SHIFT) | (2u);
  case 'I': /* insertion to the reference */
    return (length << BAM_CIGAR_SHIFT) | (1u);
  case 'H': /* hard clipping (clipped sequences NOT present in SEQ) */
    return (length << BAM_CIGAR_SHIFT) | (5u);
  case 'N': /* skipped region from the reference */
    return (length << BAM_CIGAR_SHIFT) | (3u);
  case 'P': /* padding (silent deletion from padded reference) */
    return (length << BAM_CIGAR_SHIFT) | (6u);
  case '=': /* sequence match */
    return (length << BAM_CIGAR_SHIFT) | (7u);
  case 'X': /* sequence mismatch */
    return (length << BAM_CIGAR_SHIFT) | (8u);
  }
  return (uint32_t)-1; // This never happens
}

cigar* banded_sw (const int8_t* ref,
    const int8_t* read,
    int32_t refLen,
    int32_t readLen,
    int32_t score,
    const uint32_t weight_gapO,  /* will be used as - */
    const uint32_t weight_gapE,  /* will be used as - */
    int32_t band_width,
    const int8_t* mat,	/* pointer to the weight matrix */
    int32_t n,
    int32_t read_begin,
    int32_t unclippedReadLen
) {

  //printf("banded_sw read %p, readLen %i begin %i fulllen %i \n",read,readLen,read_begin,unclippedReadLen);

  uint32_t *c = (uint32_t*)malloc(16 * sizeof(uint32_t)), *c1;
  int32_t i, j, e, f, temp1, temp2, s = 16, s1 = 8, l, max = 0;
  int64_t s2 = 1024;
  char op, prev_op;
  int32_t width, width_d, *h_b, *e_b, *h_c;
  int8_t *direction, *direction_line;
  cigar* result = (cigar*)malloc(sizeof(cigar));
  h_b = (int32_t*)malloc(s1 * sizeof(int32_t));
  e_b = (int32_t*)malloc(s1 * sizeof(int32_t));
  h_c = (int32_t*)malloc(s1 * sizeof(int32_t));
  direction = (int8_t*)malloc(s2 * sizeof(int8_t));

  do {
    width = band_width * 2 + 3, width_d = band_width * 2 + 1;
    while (width >= s1) {
      ++s1;
      kroundup32(s1);
      h_b = (int32_t*)realloc(h_b, s1 * sizeof(int32_t));
      e_b = (int32_t*)realloc(e_b, s1 * sizeof(int32_t));
      h_c = (int32_t*)realloc(h_c, s1 * sizeof(int32_t));
    }
    while (width_d * readLen * 3 >= s2) {
      ++s2;
      kroundup32(s2);
      if (s2 < 0) {
        fprintf(stderr, "Alignment score and position are not consensus.\n");
        exit(1);
      }
      direction = (int8_t*)realloc(direction, s2 * sizeof(int8_t));
    }
    direction_line = direction;
    for (j = 1; LIKELY(j < width - 1); j ++) h_b[j] = 0;
    for (i = 0; LIKELY(i < readLen); i ++) {
      int32_t beg = 0, end = refLen - 1, u = 0, edge;
      j = i - band_width;	beg = beg > j ? beg : j; // band start
      j = i + band_width; end = end < j ? end : j; // band end
      edge = end + 1 < width - 1 ? end + 1 : width - 1;
      f = h_b[0] = e_b[0] = h_b[edge] = e_b[edge] = h_c[0] = 0;
      direction_line = direction + width_d * i * 3;

      int8_t bonus = 0;
      for (j = beg; LIKELY(j <= end); j ++) {
        int32_t b, e1, f1, d, de, df, dh;
        set_u(u, band_width, i, j);	set_u(e, band_width, i - 1, j);
        set_u(b, band_width, i, j - 1); set_u(d, band_width, i - 1, j - 1);
        set_d(de, band_width, i, j, 0);
        set_d(df, band_width, i, j, 1);
        set_d(dh, band_width, i, j, 2);

        temp1 = i == 0 ? 0-weight_gapO : h_b[e] - weight_gapO;
        temp2 = i == 0 ? 0-weight_gapE : e_b[e] - weight_gapE;
        e_b[u] = temp1 > temp2 ? temp1 : temp2;
        direction_line[de] = temp1 > temp2 ? 3 : 2;

        temp1 = h_c[b] - weight_gapO;
        temp2 = f - weight_gapE;
        f = temp1 > temp2 ? temp1 : temp2;
        direction_line[df] = temp1 > temp2 ? 5 : 4;

        e1 = e_b[u] > 0 ? e_b[u] : 0;
        f1 = f > 0 ? f : 0;
        temp1 = e1 > f1 ? e1 : f1;
        bonus=0;
        if( (i+read_begin) ==0 || (i+read_begin)== (unclippedReadLen -1))
        {
          bonus = UNCLIP_BONUS;
          //printf("traceback UB  readLen %i unclippedReadLen %i read_begin %i i  %i  read_num[%i] = %i  bonus %i\n",readLen,unclippedReadLen,read_begin,i,i,read[i],bonus);

        }
        temp2 = h_b[d] + mat[ref[j] * n + read[i]] + bonus;
        h_c[u] = temp1 > temp2 ? temp1 : temp2;

        if (h_c[u] > max) max = h_c[u];

        if (temp1 <= temp2) direction_line[dh] = 1;
        else direction_line[dh] = e1 > f1 ? direction_line[de] : direction_line[df];
      }
      for (j = 1; j <= u; j ++) h_b[j] = h_c[j];
    }
    band_width *= 2;
  } while (LIKELY(max < score));
  band_width /= 2;

  // trace back
  i = readLen - 1;
  j = refLen - 1;
  e = 0;	// Count the number of M, D or I.
  l = 0;	// record length of current cigar
  op = prev_op = 'M';
  temp2 = 2;	// h
  while (LIKELY(i > 0)) {
    set_d(temp1, band_width, i, j, temp2);
    switch (direction_line[temp1]) {
    case 1:
      --i;
      --j;
      temp2 = 2;
      direction_line -= width_d * 3;
      op = 'M';
      break;
    case 2:
      --i;
      temp2 = 0;	// e
      direction_line -= width_d * 3;
      op = 'I';
      break;
    case 3:
      --i;
      temp2 = 2;
      direction_line -= width_d * 3;
      op = 'I';
      break;
    case 4:
      --j;
      temp2 = 1;
      op = 'D';
      break;
    case 5:
      --j;
      temp2 = 2;
      op = 'D';
      break;
    default:
      fprintf(stderr, "Trace back error: %d.\n", direction_line[temp1 - 1]);
      free(direction);
      free(h_c);
      free(e_b);
      free(h_b);
      free(c);
      free(result);
      return 0;
    }
    if (op == prev_op) ++e;
    else {
      ++l;
      while (l >= s) {
        ++s;
        kroundup32(s);
        c = (uint32_t*)realloc(c, s * sizeof(uint32_t));
      }
      c[l - 1] = to_cigar_int(e, prev_op);
      prev_op = op;
      e = 1;
    }
  }
  if (op == 'M') {
    ++l;
    while (l >= s) {
      ++s;
      kroundup32(s);
      c = (uint32_t*)realloc(c, s * sizeof(uint32_t));
    }
    c[l - 1] = to_cigar_int(e + 1, op);
  }else {
    l += 2;
    while (l >= s) {
      ++s;
      kroundup32(s);
      c = (uint32_t*)realloc(c, s * sizeof(uint32_t));
    }
    c[l - 2] = to_cigar_int(e, op);
    c[l - 1] = to_cigar_int(1, 'M');
  }

  // reverse cigar
  c1 = (uint32_t*)malloc(l * sizeof(uint32_t));
  s = 0;
  e = l - 1;
  while (LIKELY(s <= e)) {
    c1[s] = c[e];
    c1[e] = c[s];
    ++ s;
    -- e;
  }
  result->seq = c1;
  result->length = l;

  free(direction);
  free(h_c);
  free(e_b);
  free(h_b);
  free(c);
  return result;
}

s_profile* s_profile::create(const int8_t* _read, const int32_t _readLen,
    const int8_t* _mat, const int32_t _n, const int32_t _bias, const int8_t score_size) {

s_profile* profile = nullptr;

profile = new s_profile_sse2(_read, _readLen, _mat, _n, _bias);

profile->_baseline = new s_profile_baseline(_read, _readLen, _mat, _n, _bias);

profile->init(score_size);

return profile;
}

s_profile::s_profile (
    const int8_t* read, const int32_t readLen,
    const int8_t* mat, const int32_t n, const int32_t bias) {
  _profile_byte = nullptr;
  _profile_word = nullptr;
  _bias = bias;

  _read = read;
  _mat = mat;
  _readLen = readLen;
  _n = n;

  _baseline = nullptr;
}


void s_profile::init(const int8_t score_size) {
  if (score_size == 0 || score_size == 2) {
    _profile_byte = profile_byte_init (_read);
  }

  if (score_size == 1 || score_size == 2) {
      _profile_word = profile_word_init (_read);
      }

  if (_baseline != nullptr) {
      _baseline->init(score_size);
  }
}

s_profile::~s_profile() {
  if (_profile_byte != nullptr) {
    _mm_free(_profile_byte);
  }

  if (_profile_word != nullptr) {
    _mm_free(_profile_word);
  }

  if (_baseline != nullptr) {
    delete _baseline;
  }
}

s_align* s_profile::align (const int8_t* ref,
    int32_t refLen,
    const uint8_t weight_gapO,
    const uint8_t weight_gapE,
    const uint8_t flag,	//  (from high to low) bit 5: return the best alignment beginning position; 6: if (ref_end1 - ref_begin1 <= filterd) && (read_end1 - read_begin1 <= filterd), return cigar; 7: if max score >= filters, return cigar; 8: always return cigar; if 6 & 7 are both setted, only return cigar when both filter fulfilled
    const uint16_t filters,
    const int32_t filterd,
    const int32_t maskLen) {
    s_align* r_baseline = nullptr;
    if (_baseline != nullptr) {
    r_baseline = _baseline->align(ref, refLen, weight_gapO, weight_gapE, flag, filters, filterd, maskLen);
    }

  //printf("----- ssw_align readLen %i   read %p ------\n",prof->readLen,prof->read);
  alignment_end* bests = 0, *bests_reverse = 0;
  __m128i* vP = 0;
  int32_t word = 0, band_width = 0, readLen = _readLen;
  int8_t* read_reverse = 0;
  cigar* path;
  s_align* r = new s_align;
  r->ref_begin1 = -1;
  r->read_begin1 = -1;
  r->cigar = nullptr;
  r->cigarLen = 0;
  if (maskLen < 15) {
    //fprintf(stderr, "When maskLen < 15, the function ssw_align doesn't return 2nd best alignment information.\n");
  }

  // Find the alignment scores and ending positions
  if (_profile_byte != nullptr) {
    bests = this->ssw_byte(ref, 0, refLen, readLen, weight_gapO, weight_gapE, _profile_byte, -1, maskLen);
    if ((_profile_word != nullptr) && (bests[0].score == 255)) {
      free(bests);
      bests = this->ssw_word(ref, 0, refLen, readLen, weight_gapO, weight_gapE, _profile_word, -1, maskLen);
      word = 1;
    } else if (bests[0].score == 255) {
      fprintf(stderr, "Please set 2 to the score_size parameter of the function ssw_init, otherwise the alignment results will be incorrect.\n");
      free(r);
      return NULL;
    }
  }else if (_profile_word != nullptr) {
    bests = this->ssw_word(ref, 0, refLen, readLen, weight_gapO, weight_gapE, _profile_word, -1, maskLen);
    word = 1;
  }else {
    fprintf(stderr, "Please call the function ssw_init before ssw_align.\n");
    free(r);
    return NULL;
  }
  r->score1 = bests[0].score;
  r->ref_end1 = bests[0].ref;
  r->read_end1 = bests[0].read;
  if (maskLen >= 15) {
    r->score2 = bests[1].score;
    r->ref_end2 = bests[1].ref;
  } else {
    r->score2 = 0;
    r->ref_end2 = -1;
  }
  free(bests);
 if (flag == 0 || (flag == 2 && r->score1 < filters)) goto end;

  // Find the beginning position of the best alignment.
  read_reverse = seq_reverse(_read, r->read_end1);
  if (word == 0) {
    int8_t* profile_rev = profile_byte_rev(read_reverse,r->read_end1 + 1);
    bests_reverse = this->ssw_byte(ref, 1, r->ref_end1 + 1, r->read_end1 + 1, weight_gapO, weight_gapE, profile_rev, r->score1, maskLen);
   _mm_free(profile_rev);
  } else {
    int16_t* profile_rev = profile_word_rev(read_reverse,r->read_end1 + 1);
    bests_reverse = this->ssw_word(ref, 1, r->ref_end1 + 1, r->read_end1 + 1, weight_gapO, weight_gapE, profile_rev, r->score1, maskLen);
   _mm_free(profile_rev);
 }
  free(read_reverse);
  r->ref_begin1 = bests_reverse[0].ref;
  r->read_begin1 = r->read_end1 - bests_reverse[0].read;
  free(bests_reverse);
  if ((7&flag) == 0 || ((2&flag) != 0 && r->score1 < filters) || ((4&flag) != 0 && (r->ref_end1 - r->ref_begin1 > filterd || r->read_end1 - r->read_begin1 > filterd))) goto end;

  if(r != nullptr && r_baseline != nullptr) {
    assert(r->read_begin1 == r_baseline->read_begin1);
    assert(r->ref_begin1 == r_baseline->ref_begin1);
    assert(r->read_end1 == r_baseline->read_end1);
    assert(r->ref_end1== r_baseline->ref_end1);
    assert(r->ref_end2 == r_baseline->ref_end2);
    assert(r->score1 == r_baseline->score1);
    assert(r->score2 == r_baseline->score2);
  }

  // Generate cigar.
  refLen = r->ref_end1 - r->ref_begin1 + 1;
  readLen = r->read_end1 - r->read_begin1 + 1;
  band_width = abs(refLen - readLen) + 1;

  //printf("begin end %i %i \n",r->read_begin1 ,r->read_end1);

  path = banded_sw(ref + r->ref_begin1, _read + r->read_begin1, refLen, readLen, r->score1, weight_gapO, weight_gapE, band_width, _mat, _n,  r->read_begin1 , _readLen);
  if (path == 0) {
    free(r);
    r = NULL;
  }
  else {
    r->cigar = path->seq;
    r->cigarLen = path->length;
    free(path);
  }

end:
   if(r != nullptr && r_baseline != nullptr) {
    assert(r->read_begin1 == r_baseline->read_begin1);
    assert(r->read_end1 == r_baseline->read_end1);
    assert(r->ref_begin1 == r_baseline->ref_begin1);
    assert(r->ref_end1== r_baseline->ref_end1);
    assert(r->ref_end2 == r_baseline->ref_end2);
    assert(r->score1 == r_baseline->score1);
    assert(r->score2 == r_baseline->score2);
    assert(r->cigarLen == r_baseline->cigarLen);
    assert(strncmp((char*)r->cigar, (char*)r_baseline->cigar, r->cigarLen*4) == 0);

    delete r_baseline;
  }
  return r;
}

s_align::~s_align() {
  free(cigar);
}

uint32_t* add_cigar (uint32_t* new_cigar, int32_t* p, int32_t* s, uint32_t length, char op) {
  if ((*p) >= (*s)) {
    ++(*s);
    kroundup32(*s);
    new_cigar = (uint32_t*)realloc(new_cigar, (*s)*sizeof(uint32_t));
  }
  new_cigar[(*p) ++] = to_cigar_int(length, op);
  return new_cigar;
}

uint32_t* store_previous_m (int8_t choice,	// 0: current not M, 1: current match, 2: current mismatch
    uint32_t* length_m,
    uint32_t* length_x,
    int32_t* p,
    int32_t* s,
    uint32_t* new_cigar) {

  if ((*length_m) && (choice == 2 || !choice)) {
    new_cigar = add_cigar (new_cigar, p, s, (*length_m), '=');
    (*length_m) = 0;
  } else if ((*length_x) && (choice == 1 || !choice)) {
    new_cigar = add_cigar (new_cigar, p, s, (*length_x), 'X');
    (*length_x) = 0;
  }
  return new_cigar;
}				

/*! @function:
     1. Calculate the number of mismatches.
     2. Modify the cigar string:
         differentiate matches (=) and mismatches(X); add softclip(S) at the beginning and ending of the original cigar.
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
    int32_t* cigarLen) {

  int32_t mismatch_length = 0, p = 0, i, length, j, s = *cigarLen + 2;
  uint32_t *new_cigar = (uint32_t*)malloc(s*sizeof(uint32_t)), length_m = 0,  length_x = 0;
  char op;

  ref += ref_begin1;
  read += read_begin1;
  if (read_begin1 > 0) new_cigar[p ++] = to_cigar_int(read_begin1, 'S');
  for (i = 0; i < (*cigarLen); ++i) {
    op = cigar_int_to_op((*cigar)[i]);
    length = cigar_int_to_len((*cigar)[i]);
    if (op == 'M') {
      for (j = 0; j < length; ++j) {
        fprintf(stderr, "ref[%d]: %c\tread[%d]: %c\n", j, *ref, j, *read);
        if (*ref != *read) {
          ++ mismatch_length;
          fprintf(stderr, "length_m: %d\n", length_m);
          // the previous is match; however the current one is mismatche
          new_cigar = store_previous_m (2, &length_m, &length_x, &p, &s, new_cigar);
          ++ length_x;
        } else {
          // the previous is mismatch; however the current one is matche
          new_cigar = store_previous_m (1, &length_m, &length_x, &p, &s, new_cigar);
          ++ length_m;
        }
        ++ ref;
        ++ read;
      }
    }else if (op == 'I') {
      read += length;
      mismatch_length += length;
      new_cigar = store_previous_m (0, &length_m, &length_x, &p, &s, new_cigar);
      new_cigar = add_cigar (new_cigar, &p, &s, length, 'I');
    }else if (op == 'D') {
      ref += length;
      mismatch_length += length;
      new_cigar = store_previous_m (0, &length_m, &length_x, &p, &s, new_cigar);
      new_cigar = add_cigar (new_cigar, &p, &s, length, 'D');
    }
  }
  new_cigar = store_previous_m (0, &length_m, &length_x, &p, &s, new_cigar);

  length = readLen - read_end1 - 1;
  if (length > 0) new_cigar = add_cigar(new_cigar, &p, &s, length, 'S');

  (*cigarLen) = p;
  free(*cigar);
  (*cigar) = new_cigar;
  return mismatch_length;
}

