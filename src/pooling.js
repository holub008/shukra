const  distributions = require('distributions');
const { Matrix, inverse } = require('ml-matrix');
const { sum, preconditionNotNull, preconditionLengthEquality, preconditionAllPositive, preconditionRange,
  weightedQuantile, weightedMean, logistic } = require('./util');

const STD_NORMAL = distributions.Normal(0, 1);

function crossProduct(A, B) {
  return A.transpose().mmul(B);
}

/**
 * computes the DerSimonian-Laird estimator of tau^2 (heterogeneity)
 * this routine is largely similar to what's implemented in metafor::rma.uni
 *
 * note minEstimator=0 is taken from rma.uni's default control settings (makes sense it must be at least non-negative)
 */
function computeDLEstimator(te, se, minEstimator=0) {
  const k = se.length;
  const p = 1; // this is the number of "moderators" (predictors in the regression). we hardcode to 1, for the intercept
  // no moderators, just an intercept
  const X = Matrix.columnVector(se.map(() => 1));
  const Y = Matrix.columnVector(te);

  const wVector = se.map((s) => 1 / Math.pow(s, 2));
  const W = Matrix.diagonal(wVector);
  const stXWX = inverse(X.transpose().mmul(W).mmul(X));
  const P = W.subtract(W.mmul(X).mmul(stXWX).mmul(crossProduct(X, W)));
  const rss = crossProduct(Y, P).mmul(Y).get(0, 0);
  const trP = sum(P.diag());
  return Math.max((rss - (k - p)) / trP, minEstimator);
}

function pooling(te, w) {
  return {
    tePooled: weightedMean(te, w),
    seTEPooled: Math.sqrt(1 / sum(w)),
  };
}

function randomEffectsPooling(te, seTE) {
  const tau2 = computeDLEstimator(te, seTE);
  const w = seTE.map((se) => {
    const denominator = (Math.pow(se, 2) + tau2);
    return denominator ? 1 / denominator : 0;
  });

  return pooling(te, w);
}

function fixedEffectsPooling(te, seTE) {
  const w = seTE.map((se) => {
    const denominator = (Math.pow(se, 2));
    return denominator ? 1 / denominator : 0;
  });
  return pooling(te, w);
}

/**
 * Compute a pooled mean using inverse variance weighting & random effects.
 * If using random effects, Tau is computed via the DerSimonian-Laird estimator
 * Confidence intervals are obtained through normal approximations.
 *
 * Although it should be avoided as possible, missing SDs are imputed.
 * https://training.cochrane.org/handbook/archive/v6/chapter-06#section-6-5-2 weakly suggests using the average
 * or a larger quantile of in-sample SDs. `shukra` use the average for simplicity - buyer beware. The R package `meta` offers several
 * methods of computation from medians/iqrs/ranges; we may add at some point.
 *
 * @param n an array of the number of units behind a mean point estimate
 * @param mean an array of the mean point estimates
 * @param sd an array of the standard deviations. note, entries may be undefined/null and will be imputed.
 * @param randomEffects whether or not to estimate with random (true) or fixed (false) effects
 * @param width the desired width [0-1] of the CI on the pooled median estimate
 * @return an object with attributes `estimate` (pooled mean), `lower` (lower bound of pooled mean CI), `upper`, and `studyEstimates`.
 * if no point estimates are supplied, an empty object is returned
 */
function pooledMean(n, mean, sd, randomEffects=true, width=.95) {
  preconditionNotNull(n, 'n');
  preconditionNotNull(mean, 'mean');
  preconditionAllPositive(sd, 'sd');
  preconditionAllPositive(n, 'n');
  // we DO allow sd to be null
  preconditionLengthEquality(n, mean, 'n', 'mean');
  preconditionLengthEquality(mean, sd, 'mean', 'sd');
  preconditionRange(width, 0, 1, 'width');

  if (!n.length) {
    return {studyEstimates: []};
  }

  const validSds = sd.filter((s) => s || s === 0);
  if (!validSds.length && sd.length) {
    throw new Error('Insufficient data for imputing missing SDs');
  }
  const imputedSd = sum(validSds) / validSds.length;
  const sdImputed = sd.map(sd => sd || sd === 0 ? sd : imputedSd);

  const seTE  = sdImputed.map((sdi, ix) => Math.sqrt(Math.pow(sdi, 2) / n[ix]));

  const z = STD_NORMAL.inv((1 - width) / 2)
  const studyTEs = mean.map((m, ix) => {
    const studySE = seTE[ix];
    // if we were using an exact distribution (a Beta dist, I believe), we wouldn't need limits on these
    // regardless, these should only be triggered for small n0 or n1
    const lower = m + studySE * z;
    const upper = m - studySE * z;
    return {
      lower,
      estimate: m,
      upper,
    };
  });

  let tePooled, seTEPooled;
  if (mean.length > 1) {
    const { tePooled: a, seTEPooled: b } = randomEffects ? randomEffectsPooling(mean, seTE) : fixedEffectsPooling(mean, seTE);
    tePooled = a;
    seTEPooled = b;
  }
  else {
    // nothing to meta-analyze, just use the single point estimate
    tePooled = mean[0];
    seTEPooled = seTE[0];
  }

  return {
    estimate: tePooled,
    lower: tePooled + z * seTEPooled,
    upper: tePooled - z * seTEPooled,
    studyEstimates: studyTEs,
  };
}

function validateBinomial(events, n) {
  events.forEach((eventCount, ix) => {
    if (eventCount > n[ix]) {
      throw new Error(`event is greater than n at index ${ix}`);
    }
  });
}

/**
 * Compute a pooled rate using a inverse variance pooling
 * If using random effects, Tau is computed via the DerSimonian-Laird estimator.
 * Confidence intervals are obtained through normal approximations. All computation is done on log odds, before transforming back to probabilities for the caller.
 *
 * @param n an array of the number of units
 * @param events an array of the number of positive occurrences within the n units
 * @param randomEffects whether or not to estimate with random (true) or fixed (false) effects
 * @param width the desired width [0-1] of the CI on the pooled median estimate
 * @return an object with attributes `estimate` (pooled rate), `lower` (lower bound of pooled rate CI), `upper`, and `studyEstimates`
 */
function pooledRate(n, events, randomEffects=true, width=.95) {
  preconditionNotNull(n, 'n');
  preconditionNotNull(events, 'events');
  preconditionAllPositive(n, 'n', true);
  preconditionAllPositive(events, 'n');
  preconditionLengthEquality(n, events, 'n', 'events');
  preconditionRange(width, 0, 1, 'width');

  if (!n.length) {
    return {studyEstimates: []};
  }
  validateBinomial(events, n);

  // apply a small correction to cells with 0 events. i'm not sure what best practice is for this (add to event & n or just event)
  // and `meta` gives option for either. I'll do both, which aligns with haldane-anscambe
  const te = events.map((e, ix) => e && e !== n[ix] ? Math.log(e / (n[ix] - e)) : Math.log((e + .5) / (n[ix] - e + .5)));
  const seTE = events.map((e, ix) => e && e !== n[ix] ? Math.sqrt(1 / e + 1 / (n[ix] - e)) : Math.sqrt(1 / (e + .5) + (1 / (n[ix] - e + .5))));
  // our reference `meta`, computes CIs using R's binom.test. that, in turn uses quantiles from the beta distribution
  // since that's a beast to compute, we opt for the simple normal approximation
  const z = STD_NORMAL.inv((1 - width) / 2);
  const studyTEs = events.map((e, ix) => {
    const pointEstimate = e / n[ix];
    const studySE = Math.sqrt(pointEstimate * (1 - pointEstimate) / n[ix]);
    // if we were using an exact distribution (a Beta dist, I believe), we wouldn't need limits on these
    // regardless, these should only be triggered for small n0 or n1
    const lower = Math.max(0, pointEstimate + studySE * z);
    const upper = Math.min(1, pointEstimate - studySE * z);
    return {
      lower,
      estimate: pointEstimate,
      upper,
    };
  });

  let tePooled, seTEPooled;
  if (te.length > 1) {
    const { tePooled: a, seTEPooled: b } = randomEffects ? randomEffectsPooling(te, seTE) : fixedEffectsPooling(te, seTE);
    tePooled = a;
    seTEPooled = b;
  }
  else {
    // nothing to meta-analyze, just use the single point estimate
    tePooled = te[0];
    seTEPooled = seTE[0];
  }

  // since all pooled estimates were computed on logits (log odds), we transform back to probability space
  return {
    estimate: logistic(tePooled),
    lower: Math.max(0, logistic(tePooled + z * seTEPooled)),
    upper: Math.min(1, logistic(tePooled - z * seTEPooled)),
    studyEstimates: studyTEs,
  };
}

/**
 * Compute a pooled arithmetic mean for descriptive purposes, typically where dispersion measures are not available.
 * @param n an array of the number of units behind a mean point estimate
 * @param mean an array of the mean point estimates
 * @return a number representing the pooled estimate
 */
function arithmeticPooledMean(n, mean) {
  preconditionNotNull(n, 'n');
  preconditionNotNull(mean, 'mean');
  preconditionAllPositive(n, 'n');
  preconditionLengthEquality(n, mean, 'n',  'mean');

  if (!n.length) {
    return {};
  }

  const summedN = sum(n);
  const summedValues = sum(mean.map((x, ix) => x * n[ix]));

  return {
    estimate: summedValues / summedN,
  } ;
}

/**
 * compute the Weighted Median of Medians (WM) for a pooled median estimate
 * for a full description,see https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.8013
 *
 * Note: it's not clear how this CI could be at all accurate. if the sample sizes (n) increase, our inferential power
 * remains fixed. intuitively this makes no sense.
 *
 * @param n an array of the number of units behind a mean point estimate
 * @param median an array of the median point estimates
 * @param width the desired width [0-1] of the CI on the pooled median estimate
 * @return an object with attributes `lower`, `pooled`, and `upper`
 */
function pooledMedian(n, median, width=.95) {
  preconditionNotNull(n, 'n');
  preconditionNotNull(median, 'median');
  preconditionAllPositive(n, 'n');
  preconditionLengthEquality(n, median, 'n',  'median');
  preconditionRange(width, 0, 1, 'width');

  if (!n.length) {
    return {};
  }

  const observations = median.length
  const quantile = Math.min(STD_NORMAL.inv(0.5 * width + 0.5) / (2 * Math.sqrt(observations)), .5);
  const quantiles = [0.5 - quantile, 0.5, 0.5 + quantile];

  // note that although the source paper normalizes the weights to sum to 1, it doesn't matter
  // for our weighted quantile implementation which only uses proportions of the weights
  const estimates = weightedQuantile(median, n, quantiles);
  return {
    estimate: estimates[1],
    lower: estimates[0],
    upper: estimates[2],
  };
}

module.exports = {
  pooledMean: pooledMean,
  arithmeticPooledMean: arithmeticPooledMean,
  pooledMedian: pooledMedian,
  pooledRate: pooledRate,
};