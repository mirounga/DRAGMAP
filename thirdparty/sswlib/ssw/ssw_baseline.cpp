#include <iostream>
#include <fstream>
#include <cstdint>
#include <math.h>

#include <emmintrin.h>

#include "ssw.hpp"
#include "ssw_internal.hpp"

	s_profile_baseline::s_profile_baseline(const int8_t* _read, const int32_t _readLen,
		const int8_t* _mat, const int32_t _n, const int32_t _bias)
        : s_profile(_read, _readLen, _mat,_n, _bias)
    {}

 alignment_end* s_profile_baseline::ssw_byte (const int8_t* ref,
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
    int32_t maskLen) {

#define max16(m, vm) (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 8)); \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 4)); \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 2)); \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 1)); \
    (m) = _mm_extract_epi16((vm), 0)

  uint8_t max = 0;		                     /* the max alignment score */
  int32_t end_read = readLen - 1;
  int32_t end_ref = -1; /* 0_based best alignment ending point; Initialized as isn't aligned -1. */
  int32_t segLen = (readLen + 15) / 16; /* number of segment */

  /* array to record the largest score of each reference position */
  uint8_t* maxColumn = (uint8_t*) calloc(refLen, 1);

  /* array to record the alignment read ending position of the largest score of each reference position */
  int32_t* end_read_column = (int32_t*) calloc(refLen, sizeof(int32_t));

  /* Define 16 byte 0 vector. */
  __m128i vZero = _mm_set1_epi32(0);

  __m128i* pvHStore = (__m128i*) calloc(segLen, sizeof(__m128i));
  __m128i* pvHLoad = (__m128i*) calloc(segLen, sizeof(__m128i));
  __m128i* pvE = (__m128i*) calloc(segLen, sizeof(__m128i));
  __m128i* pvHmax = (__m128i*) calloc(segLen, sizeof(__m128i));

  int32_t i, j;
  /* 16 byte insertion begin vector */
  __m128i vGapO = _mm_set1_epi8(weight_gapO);

  /* 16 byte insertion extension vector */
  __m128i vGapE = _mm_set1_epi8(weight_gapE);

  /* 16 byte bias vector */
  __m128i vBias = _mm_set1_epi8(_bias);

  __m128i vMaxScore = vZero; /* Trace the highest score of the whole SW matrix. */
  __m128i vMaxMark = vZero; /* Trace the highest score till the previous column. */
  __m128i vTemp;
  int32_t edge, begin = 0, end = refLen, step = 1;

  /* outer loop to process the reference sequence */
  if (ref_dir == 1) {
    begin = refLen - 1;
    end = -1;
    step = -1;
  }
  //  std::ofstream dump;
  //dump.open("ssw_base.csv");

for (i = begin; LIKELY(i != end); i += step) {
    int32_t cmp;
    __m128i e, vF = vZero, vMaxColumn = vZero; /* Initialize F value to 0.
							   Any errors to vH values will be corrected in the Lazy_F loop.
     */

    __m128i vH = pvHStore[segLen - 1];
    vH = _mm_slli_si128 (vH, 1); /* Shift the 128-bit value in vH left by 1 byte. */
    const __m128i* pvP = reinterpret_cast<const __m128i*>(profile) + ref[i] * segLen; /* Right part of the vProfile */

    /* Swap the 2 H buffers. */
    __m128i* pv = pvHLoad;
    pvHLoad = pvHStore;
    pvHStore = pv;

    /* inner loop to process the query sequence */
    for (j = 0; LIKELY(j < segLen); ++j) {
      __m128i p = _mm_load_si128(pvP + j);

      vH = _mm_adds_epu8(vH, p);
      vH = _mm_subs_epu8(vH, vBias); /* vH will be always > 0 */

      /* Get max from vH, vE and vF. */
      e = _mm_load_si128(pvE + j);

      vH = _mm_max_epu8(vH, e);
      vH = _mm_max_epu8(vH, vF);

      vMaxColumn = _mm_max_epu8(vMaxColumn, vH);

      /* Save vH values. */
      _mm_store_si128(pvHStore + j, vH);

      /* Update vE value. */
      vH = _mm_subs_epu8(vH, vGapO); /* saturation arithmetic, result >= 0 */

      e = _mm_subs_epu8(e, vGapE);
      e = _mm_max_epu8(e, vH);
      _mm_store_si128(pvE + j, e);

      /* Update vF value. */
      vF = _mm_subs_epu8(vF, vGapE);
      vF = _mm_max_epu8(vF, vH);

      /* Load the next vH. */
      vH = _mm_load_si128(pvHLoad + j);
    }

    //dump << i << ",E,";
    //for (int di = 0; di < 16; di++) {
    //  for(int ds = 0; ds <segLen; ds++)
    //    dump << int(pvE[ds].m128i_u8[di]) << ",";
    //}
    //dump << std::endl;

    //dump << i << ",H0,";
    //for (int di = 0; di < 16; di++) {
    //  for(int ds = 0; ds <segLen; ds++)
    //    dump << int(pvHStore[ds].m128i_u8[di]) << ",";
    //}
    //dump << std::endl;

    /* Lazy_F loop: has been revised to disallow adjacent insertion and then deletion, so don't update E(i, j), learn from SWPS3 */
    /* reset pointers to the start of the saved data */
    j = 0;
    vH = _mm_load_si128 (pvHStore + j);

    /*  the computed vF value is for the given column.  since */
    /*  we are at the end, we need to shift the vF value over */
    /*  to the next column. */
    vF = _mm_slli_si128 (vF, 1);
    vTemp = _mm_subs_epu8 (vH, vGapO);
    vTemp = _mm_subs_epu8 (vF, vTemp);
    vTemp = _mm_cmpeq_epi8 (vTemp, vZero);
    cmp  = _mm_movemask_epi8 (vTemp);

    while (cmp != 0xffff)
    {
      vH = _mm_max_epu8 (vH, vF);
      vMaxColumn = _mm_max_epu8(vMaxColumn, vH);
      _mm_store_si128 (pvHStore + j, vH);
      vF = _mm_subs_epu8 (vF, vGapE);
      j++;
      if (j >= segLen)
      {
        j = 0;
        vF = _mm_slli_si128 (vF, 1);
      }
      vH = _mm_load_si128 (pvHStore + j);

      vTemp = _mm_subs_epu8 (vH, vGapO);
      vTemp = _mm_subs_epu8 (vF, vTemp);
      vTemp = _mm_cmpeq_epi8 (vTemp, vZero);
      cmp  = _mm_movemask_epi8 (vTemp);
    }

    //dump << i << ",H1,";
    //for (int di = 0; di < 16; di++) {
    //  for(int ds = 0; ds <segLen; ds++)
    //    dump << int(pvHStore[ds].m128i_u8[di]) << ",";
    //}
    //dump << std::endl;

    vMaxScore = _mm_max_epu8(vMaxScore, vMaxColumn);
    vTemp = _mm_cmpeq_epi8(vMaxMark, vMaxScore);
    cmp = _mm_movemask_epi8(vTemp);
    if (cmp != 0xffff) {
      uint8_t temp;
      vMaxMark = vMaxScore;
      max16(temp, vMaxScore);
      vMaxScore = vMaxMark;

      if (LIKELY(temp > max)) {
        max = temp;
        if (max + _bias >= 255) break;	//overflow
        end_ref = i;

        /* Store the column with the highest alignment score in order to trace the alignment ending position on read. */
        for (j = 0; LIKELY(j < segLen); ++j) pvHmax[j] = pvHStore[j];
      }
    }

    /* Record the max score of current column. */
    max16(maxColumn[i], vMaxColumn);
    if (maxColumn[i] == terminate) break;
  }

  /* Trace the alignment ending position on read. */
  uint8_t *t = (uint8_t*)pvHmax;
  int32_t column_len = segLen * 16;
  for (i = 0; LIKELY(i < column_len); ++i, ++t) {
    int32_t temp;
    if (*t == max) {
      temp = i / 16 + i % 16 * segLen;
      if (temp < end_read) end_read = temp;
    }
  }

  free(pvHmax);
  free(pvE);
  free(pvHLoad);
  free(pvHStore);

  alignment_end* bests = (alignment_end*) calloc(2, sizeof(alignment_end));
  /* Record the best alignment. */
  bests[0].score = max + _bias >= 255 ? 255 : max;
  bests[0].ref = end_ref;
  bests[0].read = end_read;

  /* Find the most possible 2nd best alignment. */
  bests[1].score = 0;
  bests[1].ref = 0;
  bests[1].read = 0;
  
  //for (int col = 0; col < refLen; col ++) {
  //  std::cerr << "B, " << bests[0].score << ",\t"  << col << ",\t" << int(maxColumn[col]) << std::endl;
  //}

  //std::cerr << std::endl << std::endl;

  edge = (end_ref - maskLen) > 0 ? (end_ref - maskLen) : 0;
  //std::cerr << edge << std::endl;
  for (i = 0; i < edge; i ++) {
    if (maxColumn[i] > bests[1].score) {
      bests[1].score = maxColumn[i];
      bests[1].ref = i;
    }
  //  std::cerr << "B " << bests[0].score << "\t"  << i << "\t" << int(maxColumn[i]) << "\t" << bests[1].score << "\t" << bests[1].ref << std::endl;
  }
  edge = (end_ref + maskLen) > refLen ? refLen : (end_ref + maskLen);
  //std::cerr << edge << std::endl;
  for (i = edge + 1; i < refLen; i ++) {
    if (maxColumn[i] > bests[1].score) {
      bests[1].score = maxColumn[i];
      bests[1].ref = i;
    }
 //  std::cerr << "B " << bests[0].score << "\t" << i << "\t" << int(maxColumn[i]) << "\t" << bests[1].score << "\t" << bests[1].ref << std::endl;
  }

  //std::cerr << std::endl << std::endl;

  free(maxColumn);
  free(end_read_column);
  return bests;
}


alignment_end* s_profile_baseline::ssw_word (const int8_t* ref,
    int8_t ref_dir,	// 0: forward ref; 1: reverse ref
    int32_t refLen,
    int32_t readLen,
    const uint8_t weight_gapO, /* will be used as - */
    const uint8_t weight_gapE, /* will be used as - */
    const int16_t* profile,
    uint16_t terminate,
    int32_t maskLen) {

#define max8(m, vm) (vm) = _mm_max_epi16((vm), _mm_srli_si128((vm), 8)); \
    (vm) = _mm_max_epi16((vm), _mm_srli_si128((vm), 4)); \
    (vm) = _mm_max_epi16((vm), _mm_srli_si128((vm), 2)); \
    (m) = _mm_extract_epi16((vm), 0)

  uint16_t max = 0;		                     /* the max alignment score */
  int32_t end_read = readLen - 1;
  int32_t end_ref = 0; /* 1_based best alignment ending point; Initialized as isn't aligned - 0. */
  int32_t segLen = (readLen + 7) / 8; /* number of segment */

  /* array to record the largest score of each reference position */
  uint16_t* maxColumn = (uint16_t*) calloc(refLen, 2);

  /* array to record the alignment read ending position of the largest score of each reference position */
  int32_t* end_read_column = (int32_t*) calloc(refLen, sizeof(int32_t));

  /* Define 16 byte 0 vector. */
  __m128i vZero = _mm_set1_epi32(0);

  __m128i* pvHStore = (__m128i*) calloc(segLen, sizeof(__m128i));
  __m128i* pvHLoad = (__m128i*) calloc(segLen, sizeof(__m128i));
  __m128i* pvE = (__m128i*) calloc(segLen, sizeof(__m128i));
  __m128i* pvHmax = (__m128i*) calloc(segLen, sizeof(__m128i));

  int32_t i, j, k;
  /* 16 byte insertion begin vector */
  __m128i vGapO = _mm_set1_epi16(weight_gapO);

  /* 16 byte insertion extension vector */
  __m128i vGapE = _mm_set1_epi16(weight_gapE);

  __m128i vMaxScore = vZero; /* Trace the highest score of the whole SW matrix. */
  __m128i vMaxMark = vZero; /* Trace the highest score till the previous column. */
  __m128i vTemp;
  int32_t edge, begin = 0, end = refLen, step = 1;

  /* outer loop to process the reference sequence */
  if (ref_dir == 1) {
    begin = refLen - 1;
    end = -1;
    step = -1;
  }
  for (i = begin; LIKELY(i != end); i += step) {
    int32_t cmp;
    __m128i e, vF = vZero; /* Initialize F value to 0.
							   Any errors to vH values will be corrected in the Lazy_F loop.
     */
    __m128i vH = pvHStore[segLen - 1];
    vH = _mm_slli_si128 (vH, 2); /* Shift the 128-bit value in vH left by 2 byte. */

    /* Swap the 2 H buffers. */
    __m128i* pv = pvHLoad;

    __m128i vMaxColumn = vZero; /* vMaxColumn is used to record the max values of column i. */

    const __m128i* pvP = reinterpret_cast<const __m128i*>(profile) + ref[i] * segLen; /* Right part of the vProfile */
    pvHLoad = pvHStore;
    pvHStore = pv;

    /* inner loop to process the query sequence */
    for (j = 0; LIKELY(j < segLen); j ++) {
      vH = _mm_adds_epi16(vH, _mm_load_si128(pvP + j));

      /* Get max from vH, vE and vF. */
      e = _mm_load_si128(pvE + j);
      vH = _mm_max_epi16(vH, e);
      vH = _mm_max_epi16(vH, vF);
      vMaxColumn = _mm_max_epi16(vMaxColumn, vH);

      /* Save vH values. */
      _mm_store_si128(pvHStore + j, vH);

      /* Update vE value. */
      vH = _mm_subs_epu16(vH, vGapO); /* saturation arithmetic, result >= 0 */
      e = _mm_subs_epu16(e, vGapE);
      e = _mm_max_epi16(e, vH);
      _mm_store_si128(pvE + j, e);

      /* Update vF value. */
      vF = _mm_subs_epu16(vF, vGapE);
      vF = _mm_max_epi16(vF, vH);

      /* Load the next vH. */
      vH = _mm_load_si128(pvHLoad + j);
    }

    /* Lazy_F loop: has been revised to disallow adjacent insertion and then deletion, so don't update E(i, j), learn from SWPS3 */
    for (k = 0; LIKELY(k < 8); ++k) {
      vF = _mm_slli_si128 (vF, 2);
      for (j = 0; LIKELY(j < segLen); ++j) {
        vH = _mm_load_si128(pvHStore + j);
        vH = _mm_max_epi16(vH, vF);
        vMaxColumn = _mm_max_epi16(vMaxColumn, vH); //newly added line
        _mm_store_si128(pvHStore + j, vH);
        vH = _mm_subs_epu16(vH, vGapO);
        vF = _mm_subs_epu16(vF, vGapE);
        if (UNLIKELY(! _mm_movemask_epi8(_mm_cmpgt_epi16(vF, vH)))) goto end;
      }
    }

    end:
    vMaxScore = _mm_max_epi16(vMaxScore, vMaxColumn);
    vTemp = _mm_cmpeq_epi16(vMaxMark, vMaxScore);
    cmp = _mm_movemask_epi8(vTemp);
    if (cmp != 0xffff) {
      uint16_t temp;
      vMaxMark = vMaxScore;
      max8(temp, vMaxScore);
      vMaxScore = vMaxMark;

      if (LIKELY(temp > max)) {
        max = temp;
        end_ref = i;
        for (j = 0; LIKELY(j < segLen); ++j) pvHmax[j] = pvHStore[j];
      }
    }

    /* Record the max score of current column. */
    max8(maxColumn[i], vMaxColumn);
    if (maxColumn[i] == terminate) break;
  }

  /* Trace the alignment ending position on read. */
  uint16_t *t = (uint16_t*)pvHmax;
  int32_t column_len = segLen * 8;
  for (i = 0; LIKELY(i < column_len); ++i, ++t) {
    int32_t temp;
    if (*t == max) {
      temp = i / 8 + i % 8 * segLen;
      if (temp < end_read) end_read = temp;
    }
  }

  free(pvHmax);
  free(pvE);
  free(pvHLoad);
  free(pvHStore);

  /* Find the most possible 2nd best alignment. */
  alignment_end* bests = (alignment_end*) calloc(2, sizeof(alignment_end));
  bests[0].score = max;
  bests[0].ref = end_ref;
  bests[0].read = end_read;

  bests[1].score = 0;
  bests[1].ref = 0;
  bests[1].read = 0;

  //for (int col = 0; col < refLen; col ++) {
  //  std::cerr << "B0\t" << col << "\t" << int(maxColumn[col]) << std::endl;
  //}

  edge = (end_ref - maskLen) > 0 ? (end_ref - maskLen) : 0;
  for (i = 0; i < edge; i ++) {
    if (maxColumn[i] > bests[1].score) {
      bests[1].score = maxColumn[i];
      bests[1].ref = i;
    }
//    std::cerr << "B\t"  << i << "\t" << int(maxColumn[i]) << "\t" << bests[1].score << "\t" << bests[1].ref << std::endl;
  }
  edge = (end_ref + maskLen) > refLen ? refLen : (end_ref + maskLen);
  for (i = edge; i < refLen; i ++) {
    if (maxColumn[i] > bests[1].score) {
      bests[1].score = maxColumn[i];
      bests[1].ref = i;
    }
//    std::cerr << "B\t" << i << "\t" << int(maxColumn[i]) << "\t" << bests[1].score << "\t" << bests[1].ref << std::endl;
  }

  free(maxColumn);
  free(end_read_column);
  return bests;
}
