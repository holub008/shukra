const  distributions = require('distributions');
const { Matrix, inverse } = require('ml-matrix');
const { sum, preconditionNotNull, preconditionLengthEquality,
  weightedQuantile, weightedMean, logistic } = require('./util');

const STD_NORMAL = distributions.Normal(0, 1);

function crossProduct(A, B) {
  return A.transpose().mmul(B);
}

/**
 * computes the DerSimonian-Laird estimator of tau^2 (heterogeneity)
 * this routine is largely similar to what's implemented in metafor::rma.uni
 */
function computeDLEstimator(te, se) {
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
  return (rss - (k - p)) / trP;
}

function randomEffectsPooling(te, seTE) {
  const tau2 = computeDLEstimator(te, seTE);
  const w = seTE.map((se) => {
    const denominator = (se^2 + tau2);
    return denominator ? 1 / denominator : 0;
  });
  return {
    tePooled: weightedMean(te, w),
    seTEPooled: Math.sqrt(1 / sum(w)),
  };
}

/**
 * Compute a pooled mean using a random effects model.
 * Tau is computed via the DerSimonian-Laird estimator and confidence intervals are obtained through normal
 * approximations.
 *
 * Although it should be avoided as possible, missing SDs are imputed.
 * https://training.cochrane.org/handbook/archive/v6/chapter-06#section-6-5-2 weakly suggests using the average
 * or a larger quantile of in-sample SDs. `shukra` use the average for simplicity - buyer beware. The R package `meta` offers several
 * methods of computation from medians/iqrs/ranges; we may add at some point.
 *
 * @param n an array of the number of units behind a mean point estimate
 * @param mean an array of the mean point estimates
 * @param sd an array of the standard deviations. note, entries may be undefined/null and will be imputed.
 * @param width the desired width [0-1] of the CI on the pooled median estimate
 * @return an object with attributes `estimate` (pooled mean), `lower` (lower bound of pooled mean CI), `upper`, and `studyEstimates`
 */
function randomEffectsPooledMean(n, mean, sd, width) {
  preconditionNotNull(n, 'n');
  preconditionNotNull(mean, 'mean');
  // we DO allow sd to be null
  preconditionLengthEquality(n, mean, 'n', 'mean');
  preconditionLengthEquality(mean, sd, 'mean', 'sd');

  const validSds = sd.filter(sd => sd || sd === 0);
  const imputedSd = sum(validSds) / validSds.length;
  const sdImputed = sd.map(sd => sd || sd === 0 ? sd : imputedSd);

  const seTE  = sdImputed.map((sdi) => Math.sqrt(Math.pow(sdi, 2) / n));

  const z = STD_NORMAL.inv((1 - width) / 2)
  const studyTEs = mean.map((m, ix) => {
    const studySE = seTE[ix];
    // if we were using an exact distribution (a Beta dist, I believe), we wouldn't need limits on these
    // regardless, these should only be triggered for small n0 or n1
    const lower = Math.max(0, mean - studySE * z);
    const upper = Math.min(1, mean + studySE * z);
    return {
      lower,
      estimate: m,
      upper,
    };
  });

  const { tePooled, seTEPooled } = randomEffectsPooling(mean, seTE);
  return {
    estimate: tePooled,
    lower: tePooled - z * seTEPooled,
    upper: tePooled + z * seTEPooled,
    studyEstimates: studyTEs,
  };
}

/**
 * Compute a pooled rate using a random effects model.
 * Tau is computed via the DerSimonian-Laird estimator and confidence intervals are obtained through normal
 * approximations. All computation is done on log odds, before transforming back to probabilities for the caller.
 *
 * @param n an array of the number of units
 * @param events an array of the number of positive occurrences within the n units
 * @param width the desired width [0-1] of the CI on the pooled median estimate
 * @return an object with attributes `estimate` (pooled rate), `lower` (lower bound of pooled rate CI), `upper`, and `studyEstimates`
 */
function randomEffectsPooledRate(n, events, width=.95) {
  preconditionNotNull(n, 'n');
  preconditionNotNull(events, 'events');
  preconditionLengthEquality(n, events, 'n', 'events');

  // TODO, we may want an OR correction in the case of 0/1 probabilities (i think this is called an anscombe correction)
  // note we compute logits as the treatment effects
  const te = events.map((e, ix) => Math.log(e / (n[ix] - e)));
  const seTE = events.map((e, ix) => Math.sqrt(1 / e + 1 / (n[ix] - e)));
  // our reference `meta`, computes CIs using R's binom.test. that, in turn uses quantiles from the beta distribution
  // since that's a beast to compute, we opt for the simple normal approximation
  const z = STD_NORMAL.inv((1 - width) / 2);
  const studyTEs = events.map((e, ix) => {
    const pointEstimate = e / n[ix];
    const studySE = Math.sqrt(pointEstimate * (1 - pointEstimate) / n[ix]);
    // if we were using an exact distribution (a Beta dist, I believe), we wouldn't need limits on these
    // regardless, these should only be triggered for small n0 or n1
    const lower = Math.max(0, pointEstimate - studySE * z);
    const upper = Math.min(1, pointEstimate + studySE * z);
    return {
      lower,
      estimate: pointEstimate,
      upper,
    };
  });

  const { tePooled, seTEPooled } = randomEffectsPooling(te, seTE);

  // since all pooled estimates were computed on logits (log odds), we transform back to probability space
  return {
    estimate: logistic(tePooled),
    lower: Math.max(0, logistic(tePooled - z * seTEPooled)),
    upper: Math.min(1, logistic(tePooled + z * seTEPooled)),
    studyEstimates: studyTEs,
  };
}

/**
 * Compute a pooled arithmetic mean for descriptive purposes, typically where dispersion measures are not available.
 * @param n an array of the number of units behind a mean point estimate
 * @param mean an array of the mean point estimates
 * @return a number representing the pooled estimate
 */
function pooledMean(n, mean) {
  preconditionNotNull(n, 'n');
  preconditionNotNull(mean, 'mean');
  preconditionLengthEquality(n, mean, 'n',  'mean');

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
  preconditionLengthEquality(n, median, 'n',  'median');

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
  randomEffectsPooledMean: randomEffectsPooledMean,
  pooledMean: pooledMean,
  pooledMedian: pooledMedian,
  randomEffectsPooledRate: randomEffectsPooledRate,
}