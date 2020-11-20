function sum(arr) {
  return arr.reduce((a, b) => a + b, 0);
}

function weightedMean(arr, weights) {
  preconditionLengthEquality(arr, weights);
  return sum(arr.map((a, ix) => a * weights[ix])) / sum(weights);
}

function cumSum(arr) {
  let runningSum = 0;
  const s = [0];
  arr.forEach((a) => {
    runningSum += a;
    s.push(runningSum);
  });
  return s;
}

function order(arr) {
  return arr.map((val, ind) => ({ind, val}))
    .sort((a, b) => a.val > b.val ? 1 : a.val === b.val ? 0 : -1)
    .map((obj) => obj.ind);
}

function _type7CDF(u, n, h) {
  if (u < 0 || u > 1) {
    throw new Error('u must be a probability to be drawn from the CDF');
  }

  if (u < (h - 1) / n) {
    return 0;
  } else if (u <= (h / n)) {
    return u * n - h + 1;
  } else {
    return 1;
  }
}

/**
 * compute a weighted quantile, specifically a "type 7" weighted quantile
 * https://en.wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample
 * https://aakinshin.net/posts/weighted-quantiles/
 */
function weightedQuantile(x, w, ps) {
  preconditionLengthEquality(x, w);
  const orderedIndices = order(x);
  const orderedX = orderedIndices.map((ix) => x[ix]);
  const orderedW = orderedIndices.map((ix) => w[ix]);

  const n = orderedX.length;
  return ps.map((p) => {
    const h = p * (n - 1) + 1;
    const s = cumSum(orderedW);
    const wNew = [];
    for (let ix = 0; ix < n; ix++) {
      // I think we only expect our CDF differences to be non-zero at two locations
      const wi = _type7CDF(s[ix + 1] / s[n], n, h) - _type7CDF(s[ix] / s[n], n, h);
      wNew.push(wi)
    }
    return sum(orderedX.map((xi, index) => xi * wNew[index]));
  });
}

function preconditionLengthEquality(left, right, leftName, rightName) {
  if (left.length !== right.length) {
    throw new Error(`Array lengths are not equal (${leftName ? `${leftName}=` : ''}${left.length} vs. ${rightName ? `${rightName}=` : ''}${right.length})`);
  }
}

function preconditionNotNull(arr, arrName) {
  const ix = arr.findIndex((x) => !x && x !== 0);
  if (ix >= 0) {
    throw new Error(`Element at index ${ix} is missing${arrName ? ` in ${arrName}` : ''}`);
  }
}

function logistic(x) {
  return 1 / (1 + Math.exp(x));
}

module.exports = {
  sum,
  weightedQuantile,
  weightedMean,
  logistic,
  preconditionLengthEquality,
  preconditionNotNull,
};