const { Matrix, inverse, pseudoInverse } = require('ml-matrix');
const  distributions = require('distributions');

const STD_NORMAL = distributions.Normal(0, 1);

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
        B.set(i, treatmentIndicesA[i],1);
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
    for (let i = 0; i < (nTreatments - 1) ; i++) {
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
 * @return {{p: Number, upperBound: Number, lowerBound: Number}}
 * @private
 */
function _computeInferentialStatistics(effect, standardError, transformation, width=.95, nullEffect=0) {
    const alpha = 1 - width;
    const lowerBound = effect - STD_NORMAL.inv(1 - alpha / 2) * standardError;
    const upperBound = effect + STD_NORMAL.inv(1 - alpha / 2) * standardError;
    const z = (effect - nullEffect) / standardError;
    const p = 2 * (1 - STD_NORMAL.cdf(Math.abs(z)));

    return {
        p: p,
        lowerBound: transformation(lowerBound),
        upperBound: transformation(upperBound),
    };
}

/**
 * a holder of the results of the NMA
 */
class NetworkMetaAnalysis {
    /**
     * @param {Matrix} aggregatedTreatmentEffects a square matrix with treatment effects
     * @param {Matrix} aggregatedStandardErrors a square matrix with effect standard errors
     * @param {Array} orderedTreatments the list of unique treatments corresponding to row and column indices
     * @param {Function} transformation a function applied to all treatment effects (e.g. if effects are on log scale)
     */
    constructor(aggregatedTreatmentEffects, aggregatedStandardErrors, orderedTreatments, transformation = x => x) {
        this._treatmentEffects = aggregatedTreatmentEffects;
        this._standardErrors = aggregatedStandardErrors;
        this._treatments = orderedTreatments;
        this._transformation = transformation;
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

        return this._transformation(this._treatmentEffects.get(i, j));
    }

    /**
     * @param treatmentA
     * @param treatmentB
     * @param {Number} width specifies the width of intervals
     * @param {Number} nullEffect specifies basis of comparison for any hypothesis test (untransformed)
     * @return {{p: Number, upperBound: Number, lowerBound: Number}}
     */
    computeInferentialStatistics(treatmentA, treatmentB, width, nullEffect=0) {
        const i = this._treatments.indexOf(treatmentA);
        const j = this._treatments.indexOf(treatmentB);
        if (i < 0 || j < 0) {
            throw new Error(`Requesting NMA for non-present treatment(s): ${treatmentA}, ${treatmentB}`);
        }

        return _computeInferentialStatistics(this._treatmentEffects.get(i, j), this._standardErrors.get(i, j),
            this._transformation, width, nullEffect);
    }

    /**
     * @return {Array} the treatments applied in the NMA
     */
    getTreatments() {
        return this._treatments.slice();
    }
}

/**
 * perform a fixed effect NMA
 *
 * @param {Array<Number>} effects observed treatment effects
 * @param {Array<Number>} standardErrors standard errors of the treatments
 * @param {Array<Number>} treatmentIndicesA indexed 0:nTreatments
 * @param {Array<Number>} treatmentIndicesB indexed 0:nTreatments
 * @return {{aggregatedInferentialStatistics: any[], aggregatedTreatmentEffects: Matrix, consistentContrastEffects: Array<Number>
 * @private
 */
function _fixedEffectsNMA(effects, standardErrors, treatmentIndicesA, treatmentIndicesB) {
    const m = effects.length;

    if (m !== standardErrors.length && m !== treatmentIndicesA.length && m !== bTreatmentIndices.length) {
        throw new Error(
            'Effects, SEs, and treatment indices must all have the same length');
    }

    // number of unique treatments
    const nTreatments =  _setUnion(treatmentIndicesA, treatmentIndicesB).size;
    // per-study weights
    const W = Matrix.diagonal(standardErrors.map(se => 1 / Math.pow(se, 2)));
    // contrast matrices
    const BObserved = _buildObservedPairwiseContrasts(treatmentIndicesA, treatmentIndicesB);

    // linear algebra I don't understand
    const L = BObserved.transpose().mmul(W).mmul(BObserved);
    const LInv = inverse(L.subtract(1 / nTreatments)).add(1 / nTreatments);

    const R = Matrix.zeros(nTreatments, nTreatments);
    for (let i = 0; i < nTreatments; i++) {
        for (let j = 0; j < nTreatments; j++) {
            R.set(i, j, LInv.get(i ,i) + LInv.get(j, j) - 2 * LInv.get(i, j));
        }
    }

    const V = new Array(m);
    for (let i = 0; i < V.length; i++) {
        V[i] = R.get(treatmentIndicesA[i], treatmentIndicesB[i]);
    }

    const G = BObserved.mmul(LInv).mmul(BObserved.transpose());
    const H = G.mmul(W);

    // NMA effects at the study level
    const treatmentEffectMatrix = Matrix.columnVector(effects);
    // transform observed treatment effects to those consistent with the NMA
    const consistentContrastEffects = H.mmul(treatmentEffectMatrix).getColumn(0);
    // if CIs are ever needed:
    //const consistentContrastCIs = consistentContrastEffects.map(e =>
    //    _computeInferentialStatistics(e, Math.sqrt(e), width));

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

    return {
        consistentContrastEffects: consistentContrastEffects,
        aggregatedTreatmentEffects: aggregatedTreatmentEffects,
        aggregatedStandardErrors: aggregatedStandardErrors,
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
    const nStudyArms = (1 + Math.sqrt(8 * nPairings + 1)) / 2;

    const B = _buildAllPairwiseContrasts(nStudyArms);
    const Bt = B.transpose();
    const Rm = Matrix.diagonal(r);
    const cachedProduct = Bt.mmul(Rm).mmul(B);
    const R = Matrix.diagonal(cachedProduct.diag()).subtract(cachedProduct);
    const BtB = Bt.mmul(B);
    const Lt = BtB.mmul(R).mmul(BtB).divide(-2 * Math.pow(nStudyArms, 2));
    const L = pseudoInverse(Lt);

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
 * @param {Array} studies a set of labels (one for each contrast), indicating which study the contrast occured in
 * @return {{treatmentIndicesB: Array<Number>, treatmentIndicesA: Array<Number>, orderedTreatments: Array, standardErrors: Array<Number>}}
 * @private
 */
function _computePrerequisites(effectStandardErrors, treatmentsA, treatmentsB, studies) {
    // map treatments to unique indices, starting from 0
    const allTreatments = Array.from(_setUnion(treatmentsA, treatmentsB));
    const treatmentIndicesA = treatmentsA.map(trt => allTreatments.indexOf(trt));
    const treatmentIndicesB = treatmentsB.map(trt => allTreatments.indexOf(trt));

    const perPairWeights = effectStandardErrors.map(se => 1 / Math.pow(se, 2));
    // TODO this loop is kind of slow, and could be replaced by a sort + linear scan
    // but that might provide confusing interfaces downstream
    Array.from(new Set(studies)).forEach(study => {
       const pairIndices = studies
           .map((s, ix) => [s === study, ix])
           .filter(tup => tup[0])
           .map(tup => tup[1]);
       const pairWeights = pairIndices.map(ix => 1 / perPairWeights[ix]);
       const correctedSEs = _computeNewSEs(pairWeights).map(w => Math.sqrt(w));
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
 * note all ORs are log (base e) transformed to get a symmetric sampling distribution
 *
 * @param {Array} treatments condition applied to each study arm
 * @param {Array<Number>} positiveCounts observed positive (numerator) outcomes in each study arm
 * @param {Array<Number>} totalCounts total number of units in each study arm
 * @param {Number} incr Anscombe correction, added to all cells (regardless of zero counts) as a bias correction
 */
function _buildAllPairsORStatistics(treatments, positiveCounts, totalCounts, incr=.5) {
    const nPairs = treatments.length * (treatments.length - 1) / 2;
    const treatmentsA = new Array(nPairs);
    const treatmentsB = new Array(nPairs);
    const logOddsRatios = new Array(nPairs);
    const logStandardErrors = new Array(nPairs);
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
            logStandardErrors[ix] = Math.sqrt(1 / pi +  1/ ni + 1 / pj +  1 / nj);
            ix += 1;
        }
    }

    return {
        treatmentsA: treatmentsA,
        treatmentsB: treatmentsB,
        effects: logOddsRatios,
        standardErrors: logStandardErrors,
    }
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
function _fixedEffectsORNMAPreconditions(studies, treatments, positiveCounts, totalCounts) {
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
 * Perform a fixed effects Network Meta-Analysis (NMA) on discrete, binomial outcomes, using an Odds Ratio (OR) as the
 * basis of comparison. Note that fixed effects models assume that all studies sample from the same effects
 * distribution. This is often false in practice, due to different experimental designs, procedures, etc. However, fixed
 * effects can be desirable for their simplicity and increased power over alternatives (e.g. random effects).
 *
 * Input data is all array based (imagine arrays as columns of a table), with each "row" representing a study arm.
 *
 * @param {Array} studies unique labels indicating the study an arm belongs to
 * @param {Array} treatments the condition applied each arm of the study; can be any type, but must be unique
 * @param {Array<Number>} positiveCounts observed positive (numerator) outcomes in each study arm
 * @param {Array<Number>} totalCounts total number of units in each study arm
 * @return {NetworkMetaAnalysis}
 */
function fixedEffectsOddsRatioNMA(studies, treatments, positiveCounts, totalCounts) {
    _fixedEffectsORNMAPreconditions(studies, treatments, positiveCounts, totalCounts);

    const uniqueStudies = new Set(studies);
    const studyIxTuples = studies.map((s, ix) => [s, ix]);

    const treatmentsA = [];
    const treatmentsB = [];
    const effects = [];
    const standardErrors = [];
    const contrastStudies = [];
    uniqueStudies.forEach(s => {
        const studyIxs = studyIxTuples.filter(tup => tup[0] === s).map(tup => tup[1]);
        const studyTreatments = studyIxs.map(ix => treatments[ix]);
        const studyPositiveCounts = studyIxs.map(ix => positiveCounts[ix]);
        const studyTotalCounts = studyIxs.map(ix => totalCounts[ix]);

        const studyContrasts = _buildAllPairsORStatistics(studyTreatments, studyPositiveCounts, studyTotalCounts);
        treatmentsA.push(...studyContrasts.treatmentsA);
        treatmentsB.push(...studyContrasts.treatmentsB);
        effects.push(...studyContrasts.effects);
        standardErrors.push(...studyContrasts.standardErrors);
        for (let i = 0; i < studyContrasts.treatmentsA.length; i++) {
            contrastStudies.push(s);
        }
    });

    if (contrastStudies.length < 1) {
        throw new Error('Cannot perform an NMA with no treatment contrasts!')
    }

    const preprocessedData = _computePrerequisites(standardErrors, treatmentsA, treatmentsB, contrastStudies);
    const nmaData = _fixedEffectsNMA(effects, preprocessedData.standardErrors, preprocessedData.treatmentIndicesA,
        preprocessedData.treatmentIndicesB);

    return new NetworkMetaAnalysis(nmaData.aggregatedTreatmentEffects, nmaData.aggregatedStandardErrors,
        preprocessedData.orderedTreatments, x => Math.exp(x));
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
function _fixedEffectsMDNMAPreconditions(studies, treatments, means, standardDeviations) {
    if (studies.length !== treatments.length || studies.length !== means.length ||
        studies.length !== standardDeviations.length) {
        throw new Error(
            `Studies (n=${studies.length}), treatments (n=${treatments.length}), means (n=${means.length}), and standard deviations (n=${standardDeviations.length}) do not have the same length, as required.`);
    }

    _preconditionUniqueTreatments(studies, treatments);
}

function _buildAllPairsMeanDifferenceStatistics(treatments, means, standardDeviations, ns) {
    const nPairs = treatments.length * (treatments.length - 1) / 2;
    const treatmentsA = new Array(nPairs);
    const treatmentsB = new Array(nPairs);
    const meanDifferences = new Array(nPairs);
    const standardErrors = new Array(nPairs);

    let ix = 0; // because the ix = f(i,j) arithmetic is no fun
    for (let i = 0; i < treatments.length - 1; i++) {
        for (let j = i + 1; j < treatments.length; j++) {
            treatmentsA[ix] = treatments[i];
            treatmentsB[ix] = treatments[j];
            // appeal to CLT for linear combination of Gaussian RVs
            meanDifferences[ix] = means[i] - means[j];
            standardErrors[ix] = Math.sqrt(Math.pow(standardDeviations[i], 2) / ns[i] +
                Math.pow(standardDeviations[j], 2) / ns[j]);

            ix += 1;
        }
    }

    return {
        treatmentsA: treatmentsA,
        treatmentsB: treatmentsB,
        effects: meanDifferences,
        standardErrors: standardErrors,
    }
}

/**
 * Perform a fixed effects Network Meta-Analysis (NMA) on continuous outcomes, using a mean difference as the
 * basis of comparison. Note that fixed effects models assume that all studies sample from the same effects
 * distribution. This is often false in practice, due to different experimental designs, procedures, etc. However, fixed
 * effects can be desirable for their simplicity and increased power over alternatives (e.g. random effects).
 *
 * @param {Array} studies unique labels indicating the study an arm belongs to
 * @param {Array} treatments the condition applied each arm of the study; can be any type, but must be unique
 * @param {Array<Number>} means the measured mean of the outcome within the study arm
 * @param {Array<Number>} standardDeviations the measured standard deviation of the outcome within the study arm
 * @param {Array<Number>} experimentalUnits the number of units measured within the study arm
 * @return {NetworkMetaAnalysis}
 */
function fixedEffectsMeanDifferenceNMA(studies, treatments, means, standardDeviations, experimentalUnits) {
    _fixedEffectsMDNMAPreconditions(studies, treatments, means, standardDeviations);

    const uniqueStudies = new Set(studies);
    const studyIxTuples = studies.map((s, ix) => [s, ix]);

    const treatmentsA = [];
    const treatmentsB = [];
    const effects = [];
    const standardErrors = [];
    const contrastStudies = [];

    uniqueStudies.forEach(s => {
        const studyIxs = studyIxTuples.filter(tup => tup[0] === s).map(tup => tup[1]);
        const studyTreatments = studyIxs.map(ix => treatments[ix]);
        const studyMeans = studyIxs.map(ix => means[ix]);
        const studyStandardDeviations = studyIxs.map(ix => standardDeviations[ix]);
        const studyN = studyIxs.map(ix => experimentalUnits[ix]);

        const studyContrasts = _buildAllPairsMeanDifferenceStatistics(studyTreatments, studyMeans,
            studyStandardDeviations, studyN);
        treatmentsA.push(...studyContrasts.treatmentsA);
        treatmentsB.push(...studyContrasts.treatmentsB);
        effects.push(...studyContrasts.effects);
        standardErrors.push(...studyContrasts.standardErrors);
        for (let i = 0; i < studyContrasts.treatmentsA.length; i++) {
            contrastStudies.push(s);
        }
    });

    const preprocessedData = _computePrerequisites(standardErrors, treatmentsA, treatmentsB, contrastStudies);
    const nmaData = _fixedEffectsNMA(effects, preprocessedData.standardErrors, preprocessedData.treatmentIndicesA,
        preprocessedData.treatmentIndicesB);

    return new NetworkMetaAnalysis(nmaData.aggregatedTreatmentEffects, nmaData.aggregatedStandardErrors,
        preprocessedData.orderedTreatments);
}

module.exports = {
    NetworkMetaAnalysis,
    fixedEffectsOddsRatioNMA: fixedEffectsOddsRatioNMA,
    fixedEffectsMeanDifferenceNMA: fixedEffectsMeanDifferenceNMA,
};