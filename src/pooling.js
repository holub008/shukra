const  distributions = require('distributions');
const { Matrix, inverse } = require('ml-matrix');
const { sum, preconditionNotNull, preconditionLengthEquality, weightedQuantile } = require('./util');

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

/**
 *
 * @param n
 * @param mean
 * @param sd
 *
 *
 *
 * Although it should be avoided as possible, missing SDs are imputed.
 * https://training.cochrane.org/handbook/archive/v6/chapter-06#section-6-5-2 weakly suggests using the average
 * or a larger quantile of in-sample SDs. `shukra` use the average for simplicity - buyer beware. The R package `meta` offers several
 * methods of computation from medians/iqrs/ranges; we may add at some point.
 */
function randomEffectsPooledMean(n, mean, sd) {

}


function randomEffectsPooledRate(n, events, width=.95) {
    preconditionLengthEquality(n, events, 'n', 'events');
    // TODO, we may want an OR correction in the case of 0/1 probabilities (i think this is called an anscombe correction)
    // note, we are  computing effects as ORs - they will be referred to generally as TE from here on
    const te = events.map((e, ix) => Math.log(e / (n[ix] - e)));
    const seTE = events.map((e, ix) => Math.sqrt(1 / e + 1 / (n[ix] - e)));
    // our reference `meta`, computes CIs using R's binom.test. that, in turn uses quantiles from the beta distribution
    // since that's a beast to compute, we opt for the simple normal approximation
    const z = STD_NORMAL.inv((1 - width) / 2)
    const cis = events.map((e, ix) => {
      const pointEstimate = e / n[ix];
      const adjustment = Math.sqrt(pointEstimate * (1 - pointEstimate) / n[ix]);
      // if we were using an exact distribution (a Beta dist, I believe), we wouldn't need limits on these
      // regardless, these should only be triggered for small n0 or n1
      const lower = Math.max(0, pointEstimate - adjustment * z);
      const upper = Math.min(1, pointEstimate + adjustment * z);
      return [lower, upper];
    });

    const tau2 = computeDLEstimator()
    const w = seTE.map((se, ix) => 1 / (se^2) + tau2[ix]);
    const tePooled =
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

  return summedValues / summedN;
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
 * @param width the desired width of the CI on the pooled median estimate
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
    pooled: estimates[1],
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