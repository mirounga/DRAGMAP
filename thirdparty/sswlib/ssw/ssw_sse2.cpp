#include <cstdint>
#include <iostream>

#include <emmintrin.h>

#include "ssw.hpp"
#include "ssw_internal.hpp"

	s_profile_sse2::s_profile_sse2(const int8_t* _read, const int32_t _readLen,
		const int8_t* _mat, const int32_t _n, const int32_t _bias)
        : s_profile(_read, _readLen, _mat,_n, _bias)
    {}

alignment_end* s_profile_sse2::ssw_byte (const int8_t* ref,
    int8_t ref_dir,
    int32_t refLen,
    int32_t readLen,
    const uint8_t weight_gapO,
    const uint8_t weight_gapE,
    const int8_t* profile,
    uint8_t terminate,
    int32_t maskLen) {

  uint8_t max = 0;		                     /* the max alignment score */
  int32_t end_read = readLen - 1;
  int32_t end_ref = -1; /* 0_based best alignment ending point; Initialized as isn't aligned -1. */
  int32_t segLen = (readLen + 15) / 16; /* number of segment */

  /* array to record the largest score of each reference position */
  uint8_t* maxColumn = (uint8_t*) calloc(refLen, 1);

  __m128i* pvHStore = (__m128i*) calloc(segLen, sizeof(__m128i));
  __m128i* pvHLoad = (__m128i*) calloc(segLen, sizeof(__m128i));
  __m128i* pvE = (__m128i*) calloc(segLen, sizeof(__m128i));
  __m128i* pvHmax = (__m128i*) calloc(segLen, sizeof(__m128i));

  /* Define 16 byte 0 vector. */
  __m128i vZero = _mm_set1_epi32(0);

  /* 16 byte insertion begin vector */
  __m128i vGapO = _mm_set1_epi8(weight_gapO);

  /* 16 byte insertion extension vector */
  __m128i vGapE = _mm_set1_epi8(weight_gapE);
  __m128i vGapExSlen = _mm_set1_epi8(weight_gapE * segLen);
  __m128i vGapExSlen1 = _mm_set1_epi8(weight_gapE * (segLen - 1));

  /* 16 byte bias vector */
  __m128i vBias = _mm_set1_epi8(_bias);

  int32_t begin = 0, end = refLen, step = 1;

  /* outer loop to process the reference sequence */
  if (ref_dir == 1) {
    begin = refLen - 1;
    end = -1;
    step = -1;
  }

  // Initialize F value to 0.
  // Any errors to vH values will be corrected in the Lazy_F scan.
  __m128i vF = vZero;
  __m128i vFMax = vZero;

  int i = begin;

  // Prologue
  {
    const __m128i* pvP = reinterpret_cast<const __m128i*>(profile) + ref[i] * segLen;

    /* inner loop to process the query sequence */
    for (int j = 0; LIKELY(j < segLen); ++j) {
      // Initialize vHmax value.
      _mm_store_si128(pvHmax + j, vZero);

      __m128i vH = _mm_load_si128(pvP + j);

      // vH will be always > 0
      vH = _mm_subs_epu8(vH, vBias); 

      // Get max from vH and vF.
      vH = _mm_max_epu8(vH, vF);

      // Save vH value.
      _mm_store_si128(pvHStore + j, vH);

      vH = _mm_subs_epu8(vH, vGapO);

      // Update vE value.
      _mm_store_si128(pvE + j, vH);

      // Update vF value.
      vF = _mm_subs_epu8(vF, vGapE);
      vF = _mm_max_epu8(vF, vH);
    }

	{
		// F Scan
		{
			__m128i vGapK = vGapExSlen;
			vF = _mm_max_epu8(
				vF,
				_mm_slli_si128(
					_mm_subs_epu8(
						vF,
						vGapK),
					1));

			vGapK = _mm_adds_epu8(vGapK, vGapK);

			vF = _mm_max_epu8(
				vF,
				_mm_slli_si128(
					_mm_subs_epu8(
						vF,
						vGapK),
					2));

			vGapK = _mm_adds_epu8(vGapK, vGapK);

			vF = _mm_max_epu8(
				vF,
				_mm_slli_si128(
					_mm_subs_epu8(
						vF,
						vGapK),
					4));

			vGapK = _mm_adds_epu8(vGapK, vGapK);

			vF = _mm_max_epu8(
				vF,
				_mm_slli_si128(
					_mm_subs_epu8(
						vF,
						vGapK),
					8));

			// Make it exclusive
			vF = _mm_slli_si128(vF, 1);
		}

		/* Swap the 2 H buffers. */
		__m128i* pvTemp = pvHLoad;
		pvHLoad = pvHStore;
		pvHStore = pvTemp;
	}

  }

  i += step;

  for (; LIKELY(i != end); i += step) {
	 __m128i vFS = vF;

	  // Initialize F value to 0.
	  // Any errors to vH values will be corrected in the Lazy_F scan.
      vF = vZero;

	  __m128i vMaxColumn = vZero;

    __m128i vH = pvHLoad[segLen - 1];
 	  vH = _mm_max_epu8(vH, _mm_subs_epu8(vFS, vGapExSlen1));

    vH = _mm_slli_si128 (vH, 1); /* Shift the 128-bit value in vH left by 1 byte. */
    const __m128i* pvP = reinterpret_cast<const __m128i*>(profile) + ref[i] * segLen; /* Right part of the vProfile */

    /* inner loop to process the query sequence */
    for (int j = 0; LIKELY(j < segLen); ++j) {
      __m128i vP = _mm_load_si128(pvP + j);

      vH = _mm_adds_epu8(vH, vP);
      vH = _mm_subs_epu8(vH, vBias); /* vH will be always > 0 */

      /* Get max from vH, vE and vF. */
      __m128i vE = _mm_load_si128(pvE + j);

      vH = _mm_max_epu8(vH, vF);
      vH = _mm_max_epu8(vH, vE);

      /* Save vH values. */
      _mm_store_si128(pvHStore + j, vH);

      /* Update vE value. */
      vH = _mm_subs_epu8(vH, vGapO); /* saturation arithmetic, result >= 0 */

      vE = _mm_subs_epu8(vE, vGapE);
      vE = _mm_max_epu8(vE, vH);
      _mm_store_si128(pvE + j, vE);

      /* Update vF value. */
      vF = _mm_subs_epu8(vF, vGapE);
      vF = _mm_max_epu8(vF, vH);

      /* Load the next vH. */
      vH = _mm_load_si128(pvHLoad + j);
	    vH = _mm_max_epu8(vH, vFS);

	    vMaxColumn = _mm_max_epu8(vMaxColumn, vH);	// newly added line

	    vFS = _mm_subs_epu8(vFS, vGapE);
    }

	// F Scan and vMaxColumn reduce in parallel
	{
		__m128i vGapK = vGapExSlen;

		vF = _mm_max_epu8(
			vF,
			_mm_slli_si128(
				_mm_subs_epu8(
					vF,
					vGapK),
				1));

		vMaxColumn = _mm_max_epu8(
			vMaxColumn,			
			_mm_srli_si128(vMaxColumn, 1));

		vGapK = _mm_adds_epu8(vGapK, vGapK);

		vF = _mm_max_epu8(
			vF,
			_mm_slli_si128(
				_mm_subs_epu8(
					vF,
					vGapK),
				2));

		vMaxColumn = _mm_max_epu8(
			vMaxColumn, 
			_mm_srli_si128(vMaxColumn, 2));

		vGapK = _mm_adds_epu8(vGapK, vGapK);

		vF = _mm_max_epu8(
			vF,
			_mm_slli_si128(
				_mm_subs_epu8(
					vF,
					vGapK),
				4));

		vMaxColumn = _mm_max_epu8(
			vMaxColumn,
			_mm_srli_si128(vMaxColumn, 4));

		vGapK = _mm_adds_epu8(vGapK, vGapK);

		vF = _mm_max_epu8(
			vF,
			_mm_slli_si128(
				_mm_subs_epu8(
					vF,
					vGapK),
				8));

		vMaxColumn = _mm_max_epu8(
			vMaxColumn,
			_mm_srli_si128(vMaxColumn, 8));

		// Make it exclusive
		vF = _mm_slli_si128(vF, 1);
	}

	uint8_t newMax = _mm_extract_epi8(vMaxColumn, 0);

	int32_t prev_col = i - step;

	if (LIKELY(newMax > max)) {
		vFMax = vF;
		max = newMax;
		if (max + _bias >= 255) break;	//overflow
		end_ref = prev_col;

		/* Swap the 3 H buffers. */
		__m128i* pvTemp = pvHmax;
		pvHmax = pvHLoad;
		pvHLoad = pvHStore;
		pvHStore = pvTemp;
	}
	else {
		/* Swap the 2 H buffers. */
		__m128i* pvTemp = pvHLoad;
		pvHLoad = pvHStore;
		pvHStore = pvTemp;
	}

	/* Record the max score of current column. */
	maxColumn[prev_col] = newMax;
	if (newMax == terminate) break;
  }

  // Epilogue
  {
	  int32_t last_col = end - step;

	  // Last H loop
	  __m128i vFS = vF, vMaxColumn = vZero;
	  for (int j = 0; LIKELY(j < segLen); ++j) {
		  // Load the next vH.
		  __m128i vH = _mm_load_si128(pvHLoad + j);
		  vH = _mm_max_epu8(vH, vFS);
	
		  vMaxColumn = _mm_max_epu8(vMaxColumn, vH);
		  vFS = _mm_subs_epu8(vFS, vGapE);
	  }

	  // MaxColumn reduce
	  {
		  vMaxColumn = _mm_max_epu8(
			  vMaxColumn,			
			  _mm_srli_si128(vMaxColumn, 1));

		  vMaxColumn = _mm_max_epu8(
			  vMaxColumn, 
			  _mm_srli_si128(vMaxColumn, 2));

		  vMaxColumn = _mm_max_epu8(
			  vMaxColumn,
			  _mm_srli_si128(vMaxColumn, 4));

		  vMaxColumn = _mm_max_epu8(
			  vMaxColumn,
			  _mm_srli_si128(vMaxColumn, 8));
	  }

	  uint8_t newMax = _mm_extract_epi8(vMaxColumn, 0);

	  if (LIKELY(newMax > max)) {
		  vFMax = vF;
		  max = newMax;
		  end_ref = last_col;

		  // Store the column with the highest alignment score in order to trace the alignment ending position on read.
		  __m128i* pvTemp = pvHmax;
		  pvHmax = pvHLoad;
		  pvHLoad = pvHStore;
		  pvHStore = pvTemp;
	  }

	  /* Record the max score of current column. */
	  maxColumn[last_col] = newMax;

	  // Adjust Hmax row
	  vFS = vFMax;
	  for (int j = 0; LIKELY(j < segLen); ++j) {
		  // Load the next vH.
		  __m128i vH = _mm_load_si128(pvHmax + j);
		  vH = _mm_max_epu8(vH, vFS);
		  _mm_store_si128(pvHmax + j, vH);
	
		  vFS = _mm_subs_epu8(vFS, vGapE);
	  }
  }

  // Trace the alignment ending position on read.
  uint8_t *t = (uint8_t*)pvHmax;
  int32_t column_len = segLen * 16;
  for (int i = 0; LIKELY(i < column_len); ++i, ++t) {
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

  int32_t edge = (end_ref - maskLen) > 0 ? (end_ref - maskLen) : 0;
  for (int32_t col = 0; col < edge; col ++) {
    if (maxColumn[col] > bests[1].score) {
      bests[1].score = maxColumn[col];
      bests[1].ref = col;
    }
//    std::cerr << "S\t"  << col << "\t" << int(maxColumn[col]) << "\t" << bests[1].score << "\t" << bests[1].ref << std::endl;
  }

  edge = (end_ref + maskLen) > refLen ? refLen : (end_ref + maskLen);
  for (int32_t col = edge + 1; col < refLen; col ++) {
    if (maxColumn[col] > bests[1].score) {
      bests[1].score = maxColumn[col];
      bests[1].ref = col;
    }
//    std::cerr << "S\t" << col << "\t" << int(maxColumn[col]) << "\t" << bests[1].score << "\t" << bests[1].ref << std::endl;
  }

  free(maxColumn);

  return bests;
}

alignment_end* s_profile_sse2::ssw_word (const int8_t* ref,
    int8_t ref_dir,	// 0: forward ref; 1: reverse ref
    int32_t refLen,
    int32_t readLen,
    const uint8_t weight_gapO, /* will be used as - */
    const uint8_t weight_gapE, /* will be used as - */
    const int16_t* profile,
    uint16_t terminate,
    int32_t maskLen) {

  uint16_t max = 0;		                     /* the max alignment score */
  int32_t end_read = readLen - 1;
  int32_t end_ref = 0; /* 1_based best alignment ending point; Initialized as isn't aligned - 0. */
  int32_t segLen = (readLen + 7) / 8; /* number of segment */

  /* array to record the largest score of each reference position */
  uint16_t* maxColumn = (uint16_t*) calloc(refLen, 2);

  /* Define 16 byte 0 vector. */
  __m128i vZero = _mm_set1_epi32(0);

  __m128i* pvHStore = (__m128i*) calloc(segLen, sizeof(__m128i));
  __m128i* pvHLoad = (__m128i*) calloc(segLen, sizeof(__m128i));
  __m128i* pvE = (__m128i*) calloc(segLen, sizeof(__m128i));
  __m128i* pvHmax = (__m128i*) calloc(segLen, sizeof(__m128i));

  /* 16 byte insertion begin vector */
  __m128i vGapO = _mm_set1_epi16(weight_gapO);

  /* 16 byte insertion extension vector */
  __m128i vGapE = _mm_set1_epi16(weight_gapE);
  __m128i vGapExSlen = _mm_set1_epi16(weight_gapE * segLen);
  __m128i vGapExSlen1 = _mm_set1_epi16(weight_gapE * (segLen - 1));

  __m128i vMaxScore = vZero; /* Trace the highest score of the whole SW matrix. */
  __m128i vMaxMark = vZero; /* Trace the highest score till the previous column. */
  int32_t begin = 0, end = refLen, step = 1;

  /* outer loop to process the reference sequence */
  if (ref_dir == 1) {
    begin = refLen - 1;
    end = -1;
    step = -1;
  }

  // Initialize F value to 0.
  // Any errors to vH values will be corrected in the Lazy_F scan.
  __m128i vF = vZero;
  __m128i vFMax = vZero;

  int32_t i = begin;

  // Prologue
  {
    const __m128i* pvP = reinterpret_cast<const __m128i*>(profile) + ref[i] * segLen; 

    /* inner loop to process the query sequence */
    for (int32_t j = 0; LIKELY(j < segLen); j ++) {
      // Initialize vHmax value.
      _mm_store_si128(pvHmax + j, vZero);

      __m128i vH = _mm_load_si128(pvP + j);

      /* Save vH values. */
      _mm_store_si128(pvHStore + j, vH);

      /* Update vE value. */
      vH = _mm_subs_epu16(vH, vGapO);
			_mm_store_si128(pvE + j, vH);

      /* Update vF value. */
      vF = _mm_subs_epu16(vF, vGapE);
      vF = _mm_max_epi16(vF, vH);
    }

		// F Scan
		{
			__m128i vGapK = vGapExSlen;

			// First pass
			vF = _mm_max_epi16(
				vF,
				_mm_slli_si128(
					_mm_subs_epu16(
						vF,
						vGapK),
					2));

			vGapK = _mm_adds_epu16(vGapK, vGapK);

			vF = _mm_max_epi16(
				vF,
				_mm_slli_si128(
					_mm_subs_epu16(
						vF,
						vGapK),
					4));

			vGapK = _mm_adds_epu16(vGapK, vGapK);

			vF = _mm_max_epi16(
				vF,
				_mm_slli_si128(
					_mm_subs_epu16(
						vF,
						vGapK),
					8));

			// Make it exclusive
			vF = _mm_slli_si128(vF, 2);
		}

		/* Swap the 2 H buffers. */
		__m128i* pvTemp = pvHLoad;
		pvHLoad = pvHStore;
		pvHStore = pvTemp;
		
		i += step;
  }

  for (; LIKELY(i != end); i += step) {
		__m128i vFS = vF;

    vF = vZero; /* Initialize F value to 0.
							   Any errors to vH values will be corrected in the Lazy_F loop.
     */
    __m128i vH = pvHLoad[segLen - 1];
 	  vH = _mm_max_epi16(vH, _mm_subs_epu16(vFS, vGapExSlen1));
    vH = _mm_slli_si128 (vH, 2); /* Shift the 128-bit value in vH left by 2 byte. */

    __m128i vMaxColumn = vZero; /* vMaxColumn is used to record the max values of column i. */

    const __m128i* pvP = reinterpret_cast<const __m128i*>(profile) + ref[i] * segLen; /* Right part of the vProfile */

    /* inner loop to process the query sequence */
    for (int32_t j = 0; LIKELY(j < segLen); j ++) {
      vH = _mm_adds_epi16(vH, _mm_load_si128(pvP + j));

      /* Get max from vH, vE and vF. */
      __m128i vE = _mm_load_si128(pvE + j);
      vH = _mm_max_epi16(vH, vE);
      vH = _mm_max_epi16(vH, vF);

      /* Save vH values. */
      _mm_store_si128(pvHStore + j, vH);

      /* Update vE value. */
      vH = _mm_subs_epu16(vH, vGapO); /* saturation arithmetic, result >= 0 */
      vE = _mm_subs_epu16(vE, vGapE);
      vE= _mm_max_epi16(vE, vH);
      _mm_store_si128(pvE + j, vE);

      /* Update vF value. */
      vF = _mm_subs_epu16(vF, vGapE);
      vF = _mm_max_epi16(vF, vH);

      /* Load the next vH. */
      vH = _mm_load_si128(pvHLoad + j);
	    vH = _mm_max_epi16(vH, vFS);

	    vMaxColumn = _mm_max_epi16(vMaxColumn, vH);	// newly added line

	    vFS = _mm_subs_epu16(vFS, vGapE);
    }

		// F Scan and vMaxColumn reduce in parallel
		{
			__m128i vGapK = vGapExSlen;

			vF = _mm_max_epi16(
				vF,
				_mm_slli_si128(
					_mm_subs_epu16(
						vF,
						vGapK),
					2));

		  vMaxColumn = _mm_max_epi16(
			  vMaxColumn, 
			  _mm_srli_si128(vMaxColumn, 2));

			vGapK = _mm_adds_epu16(vGapK, vGapK);

			vF = _mm_max_epi16(
				vF,
				_mm_slli_si128(
					_mm_subs_epu16(
						vF,
						vGapK),
					4));

		  vMaxColumn = _mm_max_epi16(
			  vMaxColumn,
			  _mm_srli_si128(vMaxColumn, 4));

			vGapK = _mm_adds_epu16(vGapK, vGapK);

			vF = _mm_max_epi16(
				vF,
				_mm_slli_si128(
					_mm_subs_epu16(
						vF,
						vGapK),
					8));

		  vMaxColumn = _mm_max_epi16(
			  vMaxColumn,
			  _mm_srli_si128(vMaxColumn, 8));

			// Make it exclusive
			vF = _mm_slli_si128(vF, 2);
		}

	  uint16_t newMax = _mm_extract_epi16(vMaxColumn, 0);

		int32_t prev_col = i - step;

	  if (LIKELY(newMax > max)) {
			vFMax = vF;
		  max = newMax;
      end_ref = prev_col;

			/* Swap the 3 H buffers. */
			__m128i* pvTemp = pvHmax;
			pvHmax = pvHLoad;
			pvHLoad = pvHStore;
			pvHStore = pvTemp;
		}
		else {
			/* Swap the 2 H buffers. */
			__m128i* pvTemp = pvHLoad;
			pvHLoad = pvHStore;
			pvHStore = pvTemp;
		}

    /* Record the max score of current column. */
    maxColumn[prev_col] = newMax;
    if (newMax == terminate) break;
  }

  // Epilogue
  {
	  int32_t last_col = end - step;

	  // Last H loop
	  __m128i vFS = vF, vMaxColumn = vZero;
	  for (int32_t j = 0; LIKELY(j < segLen); ++j) {
		  // Load the next vH.
		  __m128i vH = _mm_load_si128(pvHLoad + j);
		  vH = _mm_max_epi16(vH, vFS);
	
		  vMaxColumn = _mm_max_epi16(vMaxColumn, vH);
		  vFS = _mm_subs_epi16(vFS, vGapE);
	  }

	  // MaxColumn reduce
	  {
		  vMaxColumn = _mm_max_epi16(
			  vMaxColumn, 
			  _mm_srli_si128(vMaxColumn, 2));

		  vMaxColumn = _mm_max_epi16(
			  vMaxColumn,
			  _mm_srli_si128(vMaxColumn, 4));

		  vMaxColumn = _mm_max_epi16(
			  vMaxColumn,
			  _mm_srli_si128(vMaxColumn, 8));
	  }

	  uint16_t newMax = _mm_extract_epi16(vMaxColumn, 0);

	  if (LIKELY(newMax > max)) {
		  vFMax = vF;
		  max = newMax;
		  end_ref = last_col;

		  // Store the column with the highest alignment score in order to trace the alignment ending position on read.
		  __m128i* pvTemp = pvHmax;
		  pvHmax = pvHLoad;
		  pvHLoad = pvHStore;
		  pvHStore = pvTemp;
	  }

	  /* Record the max score of current column. */
	  maxColumn[last_col] = newMax;

	  // Adjust Hmax row
	  vFS = vFMax;
	  for (int j = 0; LIKELY(j < segLen); ++j) {
		  // Load the next vH.
		  __m128i vH = _mm_load_si128(pvHmax + j);
		  vH = _mm_max_epi16(vH, vFS);
		  _mm_store_si128(pvHmax + j, vH);
	
		  vFS = _mm_subs_epu16(vFS, vGapE);
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
  //  std::cerr << "S0\t" << col << "\t" << int(maxColumn[col]) << std::endl;
  //}

  int32_t edge = (end_ref - maskLen) > 0 ? (end_ref - maskLen) : 0;
  for (int32_t col = 0; col < edge; col ++) {
    if (maxColumn[col] > bests[1].score) {
      bests[1].score = maxColumn[col];
      bests[1].ref = col;
    }
    //std::cerr << "S\t"  << col << "\t" << int(maxColumn[col]) << "\t" << bests[1].score << "\t" << bests[1].ref << std::endl;
  }
  edge = (end_ref + maskLen) > refLen ? refLen : (end_ref + maskLen);
  for (int32_t col = edge; col < refLen; col ++) {
    if (maxColumn[col] > bests[1].score) {
      bests[1].score = maxColumn[col];
      bests[1].ref = col;
    }
    //std::cerr << "S\t" << col << "\t" << int(maxColumn[col]) << "\t" << bests[1].score << "\t" << bests[1].ref << std::endl;
  }

  free(maxColumn);

  return bests;
}
