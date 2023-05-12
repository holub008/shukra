const { Matrix, inverse, pseudoInverse } = require('ml-matrix');
const { getConnectedComponents } = require('./graph');
const { linearRegression, STD_NORMAL } = require("./util");


function invertNChoose2(x) {
  return (1 + Math.sqrt(8 * x + 1)) / 2;
}

/**
 * @param {Array} a
 * @param {Array} b
 * @return {Set}
 * @private
 */
function _setUnion(a, b) {
  const setA = new Set(a);
  const setB = new Set(b);

  const union = new Set(setA);
  for (let elem of setB) {
    union.add(elem);
  }
  return union;
}

/**
 * given a set of observed treatments (indexed), build each pairwise contrast (head to heads between treatments)
 *
 * @param {Array<Number>} treatmentIndicesA
 * @param {Array<Number>} treatmentIndicesB
 * @return {Matrix}
 * @private
 */
function _buildObservedPairwiseContrasts(treatmentIndicesA, treatmentIndicesB) {
  const nRows = treatmentIndicesA.length;
  const nCols = _setUnion(treatmentIndicesA, treatmentIndicesB).size;
  const B = Matrix.zeros(nRows, nCols);
  for (let i = 0; i < nRows; i++) {
    B.set(i, treatmentIndicesA[i], 1);
    B.set(i, treatmentIndicesB[i], -1);
  }

  return B;
}

/**
 * given a set of possible treatments, build all unique contrasts between them
 *
 * @param {Number} nTreatments
 * @return {Matrix}
 * @private
 */
function _buildAllPairwiseContrasts(nTreatments) {
  // n choose 2
  const nPairings = nTreatments * (nTreatments - 1) / 2;
  const B = Matrix.zeros(nPairings, nTreatments);

  let rowCount = 0;
  for (let i = 0; i < (nTreatments - 1); i++) {
    for (let j = (i + 1); j < nTreatments; j++) {
      B.set(rowCount, i, 1);
      B.set(rowCount, j, -1);
      rowCount += 1;
    }
  }

  return B;
}

/**
 * Compute CIs and perform hypothesis testing (two-sided) using Gaussian sampling distributions
 *
 * @param {Number} effect the observed effect
 * @param {Number} standardError the standard error of the observed effect size
 * @param {Function} transformation a function applied to all treatment effects (e.g. for exponentiating log ORs)
 * @param {Number} width the width of confidence intervals (0-1)
 * @param {Number} nullEffect the effect size used as a null in (two sided) hypothesis testing
 * @return {{p: Number, upper: Number, lower: Number}}
 * @private
 */
function _computeInferentialStatistics(effect, standardError, transformation, width = .95, nullEffect = 0) {
  const alpha = 1 - width;
  const lowerBound = effect - STD_NORMAL.ppf(1 - alpha / 2) * standardError;
  const upperBound = effect + STD_NORMAL.ppf(1 - alpha / 2) * standardError;
  const z = (effect - nullEffect) / standardError;
  const p = 2 * (1 - STD_NORMAL.cdf(Math.abs(z)));

  return {
    p: p,
    lower: transformation(lowerBound),
    upper: transformation(upperBound),
  };
}

function coalesceNumeric(x) {
  return Number.isNaN(x) ? 0 : x;
}

// for reference: https://github.com/guido-s/netmeta/blob/257630f656d90e0d21b5ab4715c05ec29947c637/R/meta-het.R#L53
function _computeISquared(q, df, width) {
  if (df === 0) {
    return undefined;
  }

  const backTransform = (x) => (Math.pow(x, 2) - 1) / Math.pow(x, 2);

  const k = df + 1;
  const H = Math.sqrt(q / df)
  let selogH;
  if (q > k) {
    if (k >= 2) {
      selogH = 0.5 * (Math.log(q) - Math.log(k - 1)) / (Math.sqrt(2 * q) - Math.sqrt(2 * k - 3));
    }
  } else {
    if (k > 2) {
      selogH = Math.sqrt(1 / (2 * (k - 2)) * (1 - 1 / (3 * Math.pow(k - 2, 2))));
    }
  }

  if (!selogH) {
    return {
      i2: backTransform(Math.max(H, 1)),
    };
  }

  const logH = Math.log(Math.max(H, 1));
  const infer = _computeInferentialStatistics(logH, selogH, (x) => Math.exp(x), width);
  return {
    i2: backTransform(Math.max(H, 1)),
    lower: backTransform(Math.max(infer.lower, 1)),
    upper: backTransform(Math.max(infer.upper, 1)),
  };
}

// an enum-like set of statistics used in measuring effects that are acceptable for NMA
const ComparisonStatistic = {
  OR: {
    transform: (x) => Math.exp(x),
  },
  MD: {
    transform: (x) => x,
  },
};

/**
 * a holder of the results of the NMA
 */
class NetworkMetaAnalysis {
  /**
   * @param {Matrix} aggregatedTreatmentEffects a square matrix with treatment effects
   * @param {Matrix} aggregatedStandardErrors a square matrix with effect standard errors
   * @param {Array} orderedTreatments the list of unique treatments corresponding to row and column indices
   * @param {Array} studyLevelEffects an array of objects with attributes `study`, `treatment1`, `treatment2`, `effect`, `se`,  `comparisonN`
   * @param {Object} comparison one of the `ComparisonStatistic`s that effects were generated for
   * @param q {Number} cochrane's Q derived from effects
   * @param dfQ {Number} degrees of freedom in computing cochrane's Q
   */
  constructor(aggregatedTreatmentEffects,
              aggregatedStandardErrors,
              orderedTreatments,
              studyLevelEffects,
              comparison,
              q,
              dfQ) {
    this._treatmentEffects = aggregatedTreatmentEffects;
    this._standardErrors = aggregatedStandardErrors;
    this._treatments = orderedTreatments;
    this._studyLevelEffects = studyLevelEffects;
    this._comparisonStatistic = comparison;
    // since we're operating on standardized values in all existing cases, the inversion function just flips across 0
    this._inversion = (x) => -x;
    this._q = q;
    this._dfQ = dfQ
  }

  /**
   * @param treatmentA
   * @param treatmentB
   * @return {Number} estimated effect size on original scale (even if transformed under the hood)
   */
  getEffect(treatmentA, treatmentB) {
    const i = this._treatments.indexOf(treatmentA);
    const j = this._treatments.indexOf(treatmentB);
    if (i < 0 || j < 0) {
      throw new Error('Requesting NMA for non-present treatment');
    }

    return this._comparisonStatistic.transform(this._treatmentEffects.get(i, j));
  }

  /**
   * @param treatmentA
   * @param treatmentB
   * @param {Number} width specifies the width of intervals
   * @param {Number} nullEffect specifies basis of comparison for any hypothesis test (untransformed)
   * @return {{p: Number, upper: Number, lower: Number}}
   */
  computeInferentialStatistics(treatmentA, treatmentB, width, nullEffect = 0) {
    const i = this._treatments.indexOf(treatmentA);
    const j = this._treatments.indexOf(treatmentB);
    if (i < 0 || j < 0) {
      throw new Error(`Requesting NMA for non-present treatment(s): ${treatmentA}, ${treatmentB}`);
    }

    return _computeInferentialStatistics(this._treatmentEffects.get(i, j), this._standardErrors.get(i, j),
      this._comparisonStatistic.transform, width, nullEffect);
  }

  /**
   * @return {Array} the treatments applied in the NMA
   */
  getTreatments() {
    return this._treatments.slice();
  }

  /**
   * compute SUCRA p-scores, implying a ranking of treatments
   * @param smallerBetter {Boolean} indicates if a lower value in the compared statistic (eg OR, mean, etc.) is better
   * @return {Array} an array of objects with attributes `treatment`, `pScore`. Sorted by pScore descending
   */
  computePScores(smallerBetter) {
    const unsortedPscores = [];
    for (let i = 0; i < this._treatments.length; i += 1) {
      const ps = [];
      for (let j = 0; j < this._treatments.length; j += 1) {
        const te = this._treatmentEffects.get(i, j);
        const se = this._standardErrors.get(i, j);
        const weight = te === 0 ? .5 : te > 0 ? 1 : 0;
        const { p: pValue } = _computeInferentialStatistics(te, se, this._comparisonStatistic.transform);
        // convert a two sided p-value to a one-sided
        if (smallerBetter) {
          ps.push((weight * pValue / 2) + (1 - weight) * (1 - pValue / 2));
        } else {
          ps.push((weight * (1 - pValue / 2)) + (1 - weight) * (pValue / 2));
        }
      }
      const presentCount = ps.filter((pScore) => !Number.isNaN(pScore)).length;
      unsortedPscores.push(ps.reduce((a, b) => coalesceNumeric(a) + coalesceNumeric(b), 0) / presentCount);
    }

    const result = unsortedPscores.map((pScore, ix) => ({
        treatment: this._treatments[ix],
        pScore,
      }));
    result.sort((a, b) => b.pScore - a.pScore);
    return result;
  }

  _getRawStudyLevelEffects(forTreatment) {
    const directionalEffects = this._studyLevelEffects
      .filter(({treatment1}) => treatment1 === forTreatment);
    const invertedEffects = this._studyLevelEffects
      .filter(({treatment2}) => treatment2 === forTreatment)
      .map((e)  => {
        const eCopy = { ...e };
        eCopy.effect = this._inversion(e.effect);
        const trt1 = eCopy.treatment1;
        eCopy.treatment1 = eCopy.treatment2;
        eCopy.treatment2 = trt1;
        eCopy.se = e.se;
        return eCopy;
      });

    return [...directionalEffects, ...invertedEffects]
  }

  /**
   * get the (direct) effects and inferential statistics from individual studies feeding into the pooled estimates
   * inferentials are built on normal approximations
   * @param treatment the baseline used in treatment effect calculation
   * @param {Number} width the confidence interval width [0-1]
   * @return {Array} study level effects. objects in the array will have `study`, `treatment1`, `treatment2`, `effect`, `lower`, `upper`, `p`, `comparisonN`
   */
  computeStudyLevelEffects(treatment, width=.95) {
    return this._getRawStudyLevelEffects(treatment).map((e) => {
      const inferentialStats = _computeInferentialStatistics(e.effect, e.se, this._comparisonStatistic.transform, width);
      inferentialStats.effect = this._comparisonStatistic.transform(e.effect); // with inferentials done, we convert to orig scale
      inferentialStats.treatment1 = e.treatment1;
      inferentialStats.treatment2 = e.treatment2;
      inferentialStats.study = e.study;
      inferentialStats.comparisonN = e.comparisonN;
      return inferentialStats;
    });
  }

  /**
   * get "comparison adjusted" study-level effects and standard errors, expected to be used in a comparison-adjusted funnel plot
   * Chaimani 2013: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3789683/
   * @param treatment the baseline of comparison used for generating effects
   * @param level {Number} confidence level [0-1] determining width of the funnel boundaries
   * @return an object with properties:
   *   `effects`, which has properties:
   *       `study`
   *       `treatment1` (= param treatment)
   *       `treatment2`
   *       `effect`,
   *       `se`
   *   `leftFunnel` (array of [x, y] points that can be linearly interpolated)
   *   `rightFunnel` (array of [x, y] points that can be linearly interpolated)
   *   `asymmetryP` p-value for ; undefined when test cannot be run (e.g. too few studies)
   *   `asymmetryTest` name of the test used in assessing asymmetry; undefined when `asymmetryP` is undefined
   * or undefined, if there is no data for the treatment
   */
  computeComparisonAdjustedEffects(treatment, level=0.95) {
    const equivalenceEffect = 0;

    const studyLevelEffects = this._getRawStudyLevelEffects(treatment);
    if (!studyLevelEffects.length) {
      return undefined;
    }

    const maxSE = Math.max(...studyLevelEffects.map(({ se }) => se));
    const funnelPoints = 500;
    const leftFunnel = [];
    const rightFunnel = [];
    for (let i = 0; i < funnelPoints; i++) {
      const seIter = maxSE * (i / (funnelPoints - 1));
      const bounds = _computeInferentialStatistics(equivalenceEffect, seIter, this._comparisonStatistic.transform, level);
      leftFunnel.push([bounds.lower, seIter]);
      rightFunnel.push([bounds.upper, seIter]);
    }

    const adjustedEffects = studyLevelEffects.map((sl) => {
      const ti = this._treatments.indexOf(sl.treatment1);
      const tj = this._treatments.indexOf(sl.treatment2);
      // this is a divergence from netmeta: https://github.com/guido-s/netmeta/issues/11
      // here, we include indirect evidence in calculating the adjustment
      const modeledEffect = this._treatmentEffects.get(ti, tj);

      const slCopy = {...sl};
      delete slCopy.comparisonN;
      // because we will be displaying effects with input `treatment` as the baseline, we invert the effects
      slCopy.effect = this._comparisonStatistic.transform(this._inversion(sl.effect) - this._inversion(modeledEffect));
      return slCopy;
    });

    const asymmetryResults = {};
    if (studyLevelEffects.length >= 5) {
      // this is wrapped in a try/catch to protect from a variety of numerical issues / degenerate inputs crashing the entire call
      try {
        const snd = adjustedEffects.map(({ effect, se }) => effect / se);
        const precision = adjustedEffects.map(({ se }) => 1 / se);
        const reg = linearRegression(snd, precision);
        asymmetryResults.asymmetryP = reg.interceptP;
        // for now, we only support Egger, because most other test types (e.g. Harbord) are incompatible with
        // transformed / adjusted effects that we have
        asymmetryResults.asymmetryTest = 'Egger';
      } catch (e) {
        console.error(e);
      }
    }

    return ({
      effects: adjustedEffects,
      leftFunnel,
      rightFunnel,
      ...asymmetryResults,
    });
  }

  /**
   * compute I^2, and its confidence interval on the entire network
   * @param width the confidence interval width [0-1]
   * @return an object with properties `i2`, `lower` (optional, insufficient data), `upper` (optional). or, undefined if there is insufficient comparisons
   */
  computeISquared(width=0.95) {
    return _computeISquared(this._q, this._dfQ, width);
  }
}

/**
 * mimics https://github.com/guido-s/netmeta/blob/bd409919141a6519d6791afa036529e2eb069961/R/nma.ruecker.R#L19
 * @param {Array} studies the study label, for each contrast
 * @return {number}
 * @private
 */
function _computeDF1(studies) {
  const lookup = {};
  studies.forEach((s) => {
    const existingCount = lookup[s];
    if (existingCount === undefined) {
      lookup[s] = 1
    } else {
      lookup[s] += 1;
    }
  });
  const studyArmCounts = studies.map((s) => invertNChoose2(lookup[s]));
  return 2 * (studyArmCounts.map((x) => 1 / x).reduce((a, b) => a + b, 0));
}

/**
 * perform an NMA from computed effects and SEs
 *
 * @param {Array<Number>} effects observed treatment effects
 * @param {Array<Number>} standardErrors standard errors of the treatments
 * @param {Array<Number>} treatmentIndicesA indexed 0:nTreatments
 * @param {Array<Number>} treatmentIndicesB indexed 0:nTreatments
 * @param {Array} studies study labels corresponding to studies that effects were observed in (in order)
 * @private
 */
function _NMA(effects, standardErrors, treatmentIndicesA, treatmentIndicesB, studies) {
  const m = effects.length;

  if (m !== standardErrors.length && m !== treatmentIndicesA.length && m !== bTreatmentIndices.length) {
    throw new Error(
      'Effects, SEs, and treatment indices must all have the same length');
  }

  // number of unique treatments
  const nTreatments = _setUnion(treatmentIndicesA, treatmentIndicesB).size;
  // per-study weights
  const W = Matrix.diagonal(standardErrors.map(se => 1 / Math.pow(se, 2)));
  // contrast matrices
  const BObserved = _buildObservedPairwiseContrasts(treatmentIndicesA, treatmentIndicesB);

  // linear algebra I don't understand
  const L = BObserved.transpose().mmul(W).mmul(BObserved);
  const LInv = inverse(Matrix.subtract(L,1 / nTreatments)).add(1 / nTreatments);

  const R = Matrix.zeros(nTreatments, nTreatments);
  for (let i = 0; i < nTreatments; i++) {
    for (let j = 0; j < nTreatments; j++) {
      R.set(i, j, LInv.get(i, i) + LInv.get(j, j) - 2 * LInv.get(i, j));
    }
  }

  const G = BObserved.mmul(LInv).mmul(BObserved.transpose());
  const H = G.mmul(W);

  // NMA effects at the study level
  const treatmentEffectMatrix = Matrix.columnVector(effects);
  // transform observed treatment effects to those consistent with the NMA
  const consistentContrastEffects = H.mmul(treatmentEffectMatrix).getColumn(0);

  // aggregated treatment effects
  const aggregatedTreatmentEffects = Matrix.zeros(nTreatments, nTreatments);
  for (let i = 0; i < nTreatments; i++) {
    for (let j = 0; j < nTreatments; j++) {
      aggregatedTreatmentEffects.set(i, j, NaN);
    }
  }

  // initialize with "direct" evidence
  for (let i = 0; i < m; i++) {
    aggregatedTreatmentEffects.set(treatmentIndicesA[i], treatmentIndicesB[i], consistentContrastEffects[i]);
  }

  // derive using indirect evidence
  for (let i = 0; i < nTreatments; i++) {
    for (let j = 0; j < nTreatments; j++) {
      for (let k = 0; k < nTreatments; k++) {
        const ij = aggregatedTreatmentEffects.get(i, j);
        const ik = aggregatedTreatmentEffects.get(i, k);
        const jk = aggregatedTreatmentEffects.get(j, k);
        const kj = aggregatedTreatmentEffects.get(k, j);

        if (!isNaN(ik) && !isNaN(jk)) {
          aggregatedTreatmentEffects.set(i, j, ik - jk);
          aggregatedTreatmentEffects.set(j, i, jk - ik);
        }

        if (!isNaN(ij) && !isNaN(kj)) {
          aggregatedTreatmentEffects.set(i, k, ij - kj);
          aggregatedTreatmentEffects.set(k, i, kj - ij);
        }

        if (!isNaN(ik) && !isNaN(ij)) {
          aggregatedTreatmentEffects.set(j, k, ik - ij);
          aggregatedTreatmentEffects.set(k, j, ij - ik);
        }
      }
    }
  }

  const aggregatedStandardErrors = Matrix.zeros(nTreatments, nTreatments);
  for (let i = 0; i < nTreatments; i++) {
    for (let j = 0; j < nTreatments; j++) {
      aggregatedStandardErrors.set(i, j, Math.sqrt(R.get(i, j)));
    }
  }

  // computing tau for random effects
  // i have little understanding of what's happening here - we just test it for correctness in transcription

  const E = Matrix.zeros(m, m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      if (studies[i] === studies[j]) {
        E.set(i, j, 1);
      }
    }
  }
  const df1 = _computeDF1(studies);
  const df = df1 - (nTreatments - 1);
  const v = Matrix.columnVector(effects);
  const teDiff = Matrix.subtract(v, Matrix.columnVector(consistentContrastEffects));
  const Q = teDiff.transpose().mmul(W).mmul(teDiff).get(0, 0);
  const I = Matrix.identity(m, m);
  const eMod = Matrix.multiply(BObserved.mmul(BObserved.transpose()), Matrix.divide(E, 2));
  const computedTau2 = (Q - df) / Matrix.subtract(I, H).mmul(eMod).mmul(W).trace();
  const tauComputed = Math.sqrt(Math.max(0, computedTau2));

  return {
    consistentContrastEffects: consistentContrastEffects,
    treatmentEffects: aggregatedTreatmentEffects,
    standardErrors: aggregatedStandardErrors,
    tau: tauComputed,
    q: Q,
    dfQ: df,
  };
}

/**
 *  Re-weight standard errors
 *
 * @param {Array<Number>} r weights proportional to SEs
 * @return {Array<Number>}
 * @private
 */
function _computeNewSEs(r) {
  const nPairings = r.length;
  const nStudyArms = invertNChoose2(nPairings);

  const B = _buildAllPairwiseContrasts(nStudyArms);
  const Bt = B.transpose();
  const Rm = Matrix.diagonal(r);
  const cachedProduct = Bt.mmul(Rm).mmul(B);
  const R = Matrix.diagonal(cachedProduct.diag()).subtract(cachedProduct);
  const BtB = Bt.mmul(B);
  const Lt = BtB.mmul(R).mmul(BtB).divide(-2 * Math.pow(nStudyArms, 2));
  // the epsilon selected here (much smaller than default) was found to better match netmeta/R behavior on near-singular matrices
  const L = pseudoInverse(Lt, .000001);

  const W = Matrix.diagonal(L.diagonal()).subtract(L);

  const v = new Array(nPairings);
  let edgeCount = 0;
  for (let i = 0; i < nStudyArms - 1; i++) {
    for (let j = i + 1; j < nStudyArms; j++) {
      v[edgeCount] = 1 / W.get(i, j);
      edgeCount += 1;
    }
  }

  return v;
}


/**
 * map treatments to indices and compute standard errors corrected for pairwise structure
 *
 * @param {Array<Number>} effectStandardErrors
 * @param {Array} treatmentsA treatment applied to the "numerator" in each contrast
 * @param {Array} treatmentsB treatment applied to the "denominator" in each contrast
 * @param {Array} studies a set of labels (one for each contrast), indicating which study the contrast occurred in
 * @param {Number} tau estimate of between study variance (random effects). 0 = fixed effects
 * @return {{treatmentIndicesB: Array<Number>, treatmentIndicesA: Array<Number>, orderedTreatments: Array, standardErrors: Array<Number>}}
 * @private
 */
function _computePrerequisites(effectStandardErrors, treatmentsA, treatmentsB, studies, tau=0) {
  // map treatments to unique indices, starting from 0
  const allTreatments = Array.from(_setUnion(treatmentsA, treatmentsB));
  const treatmentIndicesA = treatmentsA.map(trt => allTreatments.indexOf(trt));
  const treatmentIndicesB = treatmentsB.map(trt => allTreatments.indexOf(trt));

  const perPairWeights = effectStandardErrors.map(se => 1 / (Math.pow(se, 2) + Math.pow(tau, 2)));
  // TODO this loop is kind of slow, and could be replaced by a sort + linear scan
  // but that might provide confusing interfaces downstream
  Array.from(new Set(studies)).forEach(study => {
    const pairIndices = studies
      .map((s, ix) => [s === study, ix])
      .filter(tup => tup[0])
      .map(tup => tup[1]);
    const pairWeights = pairIndices.map(ix => 1 / perPairWeights[ix]);
    const correctedSEs = _computeNewSEs(pairWeights).map((w) => Math.sqrt(w));
    pairIndices.forEach((originalIx, studyIx) => perPairWeights[originalIx] = correctedSEs[studyIx]);
  });

  return {
    standardErrors: perPairWeights,
    treatmentIndicesA: treatmentIndicesA,
    treatmentIndicesB: treatmentIndicesB,
    orderedTreatments: allTreatments,
  };
}

/**
 * throws an error of any treatment is not unique in a study
 *
 * @param {Array} studies
 * @param {Array} treatments
 * @private
 */
function _preconditionUniqueTreatments(studies, treatments) {
  const uniqueStudies = new Set(studies);
  uniqueStudies.forEach(study => {
    const armIxs = studies
      .map((s, ix) => [s, ix])
      .filter(tup => tup[0] === study)
      .map(tup => tup[1]);
    const armTreatments = armIxs.map(ix => treatments[ix]);
    if (armTreatments.length !== (new Set(armTreatments)).size) {
      throw new Error(`For study '${study}', arm treatments (${armTreatments.join(',')}) are not unique, as required.`);
    }
  });
}

function _mergeComponentNMAResults(results) {
  // preconditions will have guaranteed at least one component and therefore one result
  let treatmentEffects = results[0].treatmentEffects;
  let standardErrors = results[0].standardErrors;
  let orderedTreatments = results[0].orderedTreatments;
  let studyLevelEffects = results[0].studyLevelEffects;

  let q = results[0].q;
  let df = results[0].dfQ;

  results.slice(1).forEach((r) => {
    orderedTreatments = orderedTreatments.concat(r.orderedTreatments);
    studyLevelEffects = studyLevelEffects.concat(r.studyLevelEffects);

    const newTreatmentEffects = Matrix.zeros(
      treatmentEffects.rows + r.treatmentEffects.rows,
      treatmentEffects.columns + r.treatmentEffects.columns).mul(Number.NaN);
    newTreatmentEffects.setSubMatrix(treatmentEffects, 0, 0);
    newTreatmentEffects.setSubMatrix(r.treatmentEffects, treatmentEffects.rows, treatmentEffects.columns);
    treatmentEffects = newTreatmentEffects;

    const newStandardErrors = Matrix.zeros(
      standardErrors.rows + r.standardErrors.rows,
      standardErrors.columns + r.standardErrors.columns);
    newStandardErrors.setSubMatrix(standardErrors, 0, 0);
    newStandardErrors.setSubMatrix(r.standardErrors, standardErrors.rows, standardErrors.columns);
    standardErrors = newStandardErrors;

    q += r.q;
    df += r.dfQ;
  });

  return {
    treatmentEffects,
    standardErrors,
    orderedTreatments,
    studyLevelEffects,
    q,
    dfQ: df,
  };
}

/**
 *
 * @param studies {Array}
 * @param treatments {Array}
 * @param buildContrasts {Function} a function like _buildAllPairsORStatistics that generates pairs of arms implying treatment effects
 * @param parameters {Object} }an object with array attributes to be consumed by `buildContrasts`. arrays should share length for correct indexing
 * @param comparisonStatistic {Object} one of the `ComparisonStatistics` used for mapping between modelled and human-interpretable spaces
 * @param randomEffects {Boolean} whether or not random effects should be modeled
 * @return {NetworkMetaAnalysis}
 */
function _generalizedNMA(studies, treatments, buildContrasts, parameters, comparisonStatistic, randomEffects=false) {
  if (studies.length === 0) {
    // https://github.com/mljs/matrix/issues/113 limits the API we can provide
    throw new Error('Must have 1 or more studies to perform an NMA');
  }

  const studyIxTuples = studies.map((s, ix) => [s, ix]);

  const components = getConnectedComponents(studies, treatments);
  const componentResults = components.map((comp) => {
    const componentIxs = [];
    treatments.forEach((t, ix) => {
      if (comp.indexOf(t) > -1) {
        componentIxs.push(ix);
      }
    });

    const uniqueStudiesInComponent = [ ...(new Set(studyIxTuples
      .filter(([s, ix]) => componentIxs.indexOf(ix) > -1)
      .map(([s, ix]) => s))) ];
    const treatmentsA = [];
    const treatmentsB = [];
    const effects = [];
    const standardErrors = [];
    const comparisonNs = [];
    const contrastStudies = [];

    uniqueStudiesInComponent.map((s) => {
      const studyIxs = studyIxTuples.filter(tup => tup[0] === s).map(tup => tup[1]);
      const studyTreatments = studyIxs.map((ix) => treatments[ix]);
      const studyParameters = {};
      Object.entries(parameters).forEach(([param, arr]) => {
        studyParameters[param] = studyIxs.map((ix) => arr[ix]);
      })

      const studyContrasts = buildContrasts(studyTreatments, studyParameters);
      treatmentsA.push(...studyContrasts.treatmentsA);
      treatmentsB.push(...studyContrasts.treatmentsB);
      effects.push(...studyContrasts.effects);
      standardErrors.push(...studyContrasts.standardErrors);
      comparisonNs.push(...studyContrasts.comparisonNs);
      for (let i = 0; i < studyContrasts.treatmentsA.length; i++) {
        contrastStudies.push(s);
      }
    });

    const studyLevelEffects = contrastStudies.map((s, ix) => ({
      study: s,
      treatment1: treatmentsA[ix],
      treatment2: treatmentsB[ix],
      effect: effects[ix],
      se: standardErrors[ix],
      comparisonN: comparisonNs[ix],
    }));

    const preprocessedData = _computePrerequisites(standardErrors, treatmentsA, treatmentsB, contrastStudies);
    let nmaData = _NMA(effects, preprocessedData.standardErrors, preprocessedData.treatmentIndicesA,
      preprocessedData.treatmentIndicesB, contrastStudies);
    const q = nmaData.q;
    const dfQ = nmaData.dfQ;

    if (randomEffects) {
      // with random effects, we are deriving tau from the fixed effects model - this amounts to a DerSimonian-Laird estimator
      const tau = nmaData.tau;
      // since we weight differently with tau, we need to recompute the "prerequisites"
      const preprocessedRFData = _computePrerequisites(standardErrors, treatmentsA, treatmentsB, contrastStudies, tau);
      nmaData = _NMA(effects, preprocessedRFData.standardErrors, preprocessedRFData.treatmentIndicesA,
        preprocessedRFData.treatmentIndicesB, contrastStudies);
    }

    return {
      treatmentEffects: nmaData.treatmentEffects,
      standardErrors: nmaData.standardErrors,
      orderedTreatments: preprocessedData.orderedTreatments,
      studyLevelEffects,
      q,
      dfQ,
    };
  });

  const { treatmentEffects, standardErrors, orderedTreatments, studyLevelEffects, q, dfQ } = _mergeComponentNMAResults(componentResults);
  return new NetworkMetaAnalysis(treatmentEffects, standardErrors, orderedTreatments, studyLevelEffects, comparisonStatistic, q, dfQ);
}

/**
 * note all ORs are log (base e) transformed to get a symmetric sampling distribution
 *
 * @param {Array} treatments condition applied to each study arm
 * @param {Number} incr Anscombe correction, added to all cells (regardless of zero counts) as a bias correction
 */
function _buildAllPairsORStatistics(treatments, params, incr = .5) {
  const { positiveCounts, totalCounts } = params;
  const nPairs = treatments.length * (treatments.length - 1) / 2;
  const treatmentsA = new Array(nPairs);
  const treatmentsB = new Array(nPairs);
  const logOddsRatios = new Array(nPairs);
  const logStandardErrors = new Array(nPairs);
  const comparisonNs = new Array(nPairs);
  let ix = 0; // because the ix = f(i,j) arithmetic is no fun
  for (let i = 0; i < treatments.length - 1; i++) {
    for (let j = i + 1; j < treatments.length; j++) {
      treatmentsA[ix] = treatments[i];
      treatmentsB[ix] = treatments[j];
      const pi = positiveCounts[i] + incr;
      const ni = totalCounts[i] - positiveCounts[i] + incr;
      const pj = positiveCounts[j] + incr;
      const nj = totalCounts[j] - positiveCounts[j] + incr;
      logOddsRatios[ix] = Math.log((pi / ni) / (pj / nj));
      logStandardErrors[ix] = Math.sqrt(1 / pi + 1 / ni + 1 / pj + 1 / nj);
      comparisonNs[ix] = totalCounts[i] + totalCounts[j];

      ix += 1;
    }
  }

  return {
    treatmentsA,
    treatmentsB,
    effects: logOddsRatios,
    standardErrors: logStandardErrors,
    comparisonNs: comparisonNs,
  };
}

/**
 * throws an error for various inputs not amenable to NMA or that are just inconsistent: mismatched lengths, incorrect
 * counts, non-unique treatment arms
 *
 * @param {Array} studies
 * @param {Array} treatments
 * @param {Array<Number>} positiveCounts
 * @param {Array<Number>} totalCounts
 * @private
 */
function _ORNMAPreconditions(studies, treatments, positiveCounts, totalCounts) {
  if (studies.length !== treatments.length || studies.length !== positiveCounts.length ||
    studies.length !== totalCounts.length) {
    throw new Error(
      `Studies (n=${studies.length}), treatments (n=${treatments.length}), and count data (nPos=${positiveCounts.length}, nTotal=${totalCounts.length}) do not have the same length, as required.`);
  }

  positiveCounts.forEach((pc, ix) => {
    if (pc > totalCounts[ix]) {
      throw new Error(`At row ${ix}, positive count (${pc}) is greater than total count (${totalCounts[ix]})`);
    }
  });

  _preconditionUniqueTreatments(studies, treatments);

}

/**
 * Perform a Network Meta-Analysis (NMA) on discrete, binomial outcomes, using an Odds Ratio (OR) as the
 * basis of comparison. Note that fixed effects models assume that all studies sample from the same effects
 * distribution. This is often false in practice, due to different experimental designs, procedures, etc. However, fixed
 * effects can be desirable in some cases for their simplicity and increased power over alternatives
 *
 * Input data is all array based (imagine arrays as columns of a table), with each "row" representing a study arm.
 *
 * @param {Array} studies unique labels indicating the study an arm belongs to
 * @param {Array} treatments the condition applied each arm of the study; can be any type, but must be unique
 * @param {Array<Number>} positiveCounts observed positive (numerator) outcomes in each study arm
 * @param {Array<Number>} totalCounts total number of units in each study arm
 * @param {Boolean} randomEffects whether or not random effects should be modeled (using the DerSimonian-Laird estimator)
 * @return {NetworkMetaAnalysis}
 */
function oddsRatioNMA(studies, treatments, positiveCounts, totalCounts, randomEffects=true) {
  _ORNMAPreconditions(studies, treatments, positiveCounts, totalCounts);

  return _generalizedNMA(studies, treatments, _buildAllPairsORStatistics, {
    positiveCounts,
    totalCounts,
  }, ComparisonStatistic.OR, randomEffects);
}

/**
 * throws an error for various inputs not amenable to NMA or that are just inconsistent: mismatched lengths or
 * non-unique treatment arms
 *
 * @param {Array} studies
 * @param {Array} treatments
 * @param {Array<Number>} means
 * @param {Array<Number>} standardDeviations
 * @private
 */
function _MDNMAPreconditions(studies, treatments, means, standardDeviations) {
  if (studies.length !== treatments.length || studies.length !== means.length ||
    studies.length !== standardDeviations.length) {
    throw new Error(
      `Studies (n=${studies.length}), treatments (n=${treatments.length}), means (n=${means.length}), and standard deviations (n=${standardDeviations.length}) do not have the same length, as required.`);
  }

  _preconditionUniqueTreatments(studies, treatments);
}

function _buildAllPairsMeanDifferenceStatistics(treatments, params) {
  const { means, standardDeviations, ns } = params;
  const nPairs = treatments.length * (treatments.length - 1) / 2;
  const treatmentsA = new Array(nPairs);
  const treatmentsB = new Array(nPairs);
  const meanDifferences = new Array(nPairs);
  const standardErrors = new Array(nPairs);
  const comparisonNs = new Array(nPairs);

  let ix = 0; // because the ix = f(i,j) arithmetic is no fun
  for (let i = 0; i < treatments.length - 1; i++) {
    for (let j = i + 1; j < treatments.length; j++) {
      treatmentsA[ix] = treatments[i];
      treatmentsB[ix] = treatments[j];
      // appeal to CLT for linear combination of Gaussian RVs
      meanDifferences[ix] = means[i] - means[j];
      standardErrors[ix] = Math.sqrt(Math.pow(standardDeviations[i], 2) / ns[i] +
        Math.pow(standardDeviations[j], 2) / ns[j]);
      comparisonNs[ix] = ns[i] + ns[j];

      ix += 1;
    }
  }

  return {
    treatmentsA: treatmentsA,
    treatmentsB: treatmentsB,
    effects: meanDifferences,
    standardErrors: standardErrors,
    comparisonNs: comparisonNs,
  }
}

/**
 * Perform a Network Meta-Analysis (NMA) on continuous outcomes, using a mean difference as the
 * basis of comparison. Note that fixed effects models assume that all studies sample from the same effects
 * distribution. This is often false in practice, due to different experimental designs, procedures, etc. However, fixed
 * effects may be desirable for their simplicity and increased power over alternatives
 *
 * @param {Array} studies unique labels indicating the study an arm belongs to
 * @param {Array} treatments the condition applied each arm of the study; can be any type, but must be unique
 * @param {Array<Number>} means the measured mean of the outcome within the study arm
 * @param {Array<Number>} standardDeviations the measured standard deviation of the outcome within the study arm
 * @param {Array<Number>} experimentalUnits the number of units measured within the study arm
 * @param {Boolean} randomEffects whether or not random effects should be modeled (using the DerSimonian-Laird estimator)
 * @return {NetworkMetaAnalysis}
 */
function meanDifferenceNMA(studies, treatments, means, standardDeviations, experimentalUnits, randomEffects=true) {
  _MDNMAPreconditions(studies, treatments, means, standardDeviations);

  return _generalizedNMA(studies, treatments, _buildAllPairsMeanDifferenceStatistics, {
    means,
    standardDeviations,
    ns: experimentalUnits,
  }, ComparisonStatistic.MD, randomEffects);
}

module.exports = {
  NetworkMetaAnalysis,
  ComparisonStatistic,
  oddsRatioNMA: oddsRatioNMA,
  meanDifferenceNMA: meanDifferenceNMA,
};