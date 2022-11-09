const gaussian = require('gaussian');
const { Matrix, inverse, pseudoInverse } = require('ml-matrix');


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
  if (!left) {
    throw new Error(`Expected an array-like${leftName ? ` ${leftName}` : ''}`);
  }
  if (!right) {
    throw new Error(`Expected an array-like${rightName ? ` ${rightName}` : ''}`);
  }

  if (left.length !== right.length) {
    throw new Error(`Array lengths are not equal (${leftName ? `${leftName}=` : ''}${left.length} vs. ${rightName ? `${rightName}=` : ''}${right.length})`);
  }
}

function preconditionNotNull(arr, arrName) {
  if (!arr) {
    throw new Error(`Expected an array-like${arrName ? ` ${arrName}` : ''}`);
  }

  const ix = arr.findIndex((x) => !x && x !== 0);
  if (ix >= 0) {
    throw new Error(`Element at index ${ix} is missing${arrName ? ` in ${arrName}` : ''}`);
  }
}

function preconditionAllPositive(arr, arrName, strictlyPositive=false) {
  const ix = arr.findIndex((x) => (x || x === 0) && (strictlyPositive ? x <= 0 : x < 0));
  if (ix >= 0) {
    throw new Error(`Element at index ${ix} is non-positive${arrName ? ` in ${arrName}` : ''}`);
  }
}

function preconditionRange(x, lower, upper, name) {
  const inRange = x >= lower && x <= upper;
  if (!inRange) {
    throw new Error(`Value${name ? ` ${name}`: ''} not in range ${lower} to ${upper}`);
  }
}

function logistic(x) {
  return 1 / (1 + Math.exp(-x));
}


const STD_NORMAL = gaussian(0, 1);

// note: this is an approximation described in 10.1016/S0167-9473(99)00070-5
// its max absolute error is an order of magnitude less than what users will care about for p-values, more for df > 5
// we prefer this approximation because current offerings on NPM are bad:
// -distributions: assumes its running on NodeJS (polyfill required to run on browser)
// -stdlib: requires 175 sub-dependencies to get a CDF
function _studentTCDF(q, df) {
  if (df < 3) {
    throw new Error(`3 or degrees of freedom are required; got ${df}`);
  }
  const g = (df - 1.5) / Math.pow(df - 1, 2);
  const z = Math.sqrt(Math.log(1 + Math.pow(q, 2) / df) / g);
  return q < 0 ? 1 - STD_NORMAL.cdf(z) : STD_NORMAL.cdf(z);
}

// perform classic matrix-formulation least squares + generate inferentials on coefficients
// assumes a simple slope + intercept design matrix, all inferentials computed against a coef=0 two-sided null
function linearRegression(y, x) {
  if (y.length !== x.length) {
    throw new Error('x and y lengths must be equal');
  }
  if (x.length <= 2) {
    throw new Error('Too few observations to fit a slope + intercept')
  }
  // our design matrix X has a column of 1s appended, for the intercept
  const ones = x.map(() => 1);
  const X = Matrix.columnVector(ones).addColumn(1, x);
  const Y = Matrix.columnVector(y);

  const beta = pseudoInverse(X.transpose().mmul(X)).mmul(X.transpose()).mmul(Y);

  // standard errors
  const residuals = Matrix.subtract(Y, X.mmul(beta));
  const squaredResiduals = Matrix.pow(residuals, 2);
  const sigmaSquaredHat = squaredResiduals.sum() / (X.rows - X.columns);
  const covarianceMatrix = pseudoInverse(X.transpose().mmul(X)).mul(sigmaSquaredHat);
  const betaSEs = covarianceMatrix.diag().map((x) => Math.sqrt(x));

  // inferential tests
  const coefs = beta.to1DArray();
  const tStats = coefs.map((b, ix) => b / betaSEs[ix]);
  const testP = tStats.map((t) => 2 * (1 - _studentTCDF(t, X.rows - X.columns - 1)));

  return {
    intercept: coefs[0],
    interceptP: testP[0],
    slope: coefs[1],
    slopeP: testP[1],
  };
}

module.exports = {
  sum,
  weightedQuantile,
  weightedMean,
  logistic,
  preconditionLengthEquality,
  preconditionNotNull,
  preconditionAllPositive,
  preconditionRange,
  linearRegression,
};