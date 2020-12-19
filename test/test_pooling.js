const assert = require('assert');
const {pooledMedian, randomEffectsPooledMean, randomEffectsPooledRate, pooledMean} = require('../src/pooling');

function within(real, expected, epsilon=.01) {
  if ((isNaN(real) && !isNaN(expected)) || (!isNaN(real) && isNaN(expected))) {
    console.log(`Found value ${real} and expected value ${expected} are not equal.`);
    return false;
  }

  if (isNaN(real) &&  isNaN(expected))   {
    return true;
  }

  const outside = Math.abs(real - expected) > epsilon;
  if (outside) {
    console.log(`Found value ${real} is not near expectation ${expected}`);
  }
  return !outside;
}

describe('Random effects mean pooling', function () {
  /*
   replicate expected estimates in r with:
   library(meta)
   means <- c(10, 15, 20)
   sds <- c(1, 2, 2.4)
   ns <- c(1000, 50, 75)
   mm <- metamean(ns, means, sds)
  */
  it('should produce correct results for full data', function () {
    const means = [10, 15, 20];
    const sds = [1, 2, 2.4];
    const ns = [1000, 50, 75];
    const { estimate, lower, upper, studyEstimates } = randomEffectsPooledMean(ns, means, sds);
    assert.ok(within(estimate, 14.996));
    assert.ok(within(lower, 8.643));
    assert.ok(within(upper, 21.35));

    assert.strictEqual(studyEstimates.length, 3);
    assert.ok(within(studyEstimates[0].estimate, 10));
    assert.ok(within(studyEstimates[0].lower, 9.938));
    assert.ok(within(studyEstimates[0].upper, 10.062));

    assert.ok(within(studyEstimates[1].estimate, 15));
    assert.ok(within(studyEstimates[1].lower, 14.446));
    assert.ok(within(studyEstimates[1].upper, 15.554));

    assert.ok(within(studyEstimates[2].estimate, 20));
    assert.ok(within(studyEstimates[2].lower, 19.457));
    assert.ok(within(studyEstimates[2].upper, 20.543));
  });

  /*
   while we don't have a basis of comparison for imputation, we know the imputed sd is 1.7, so we can replicate
    expected estimates in r with:
   library(meta)
   means <- c(10, 15, 20)
   sds <- c(1, 1.7, 2.4)
   ns <- c(1000, 50, 75)
   mm <- metamean(ns, means, sds)
  */
  it('should produce correct results with missing sds', function () {
    const means = [10, 15, 20];
    const sds = [1, undefined, 2.4];
    const ns = [1000, 50, 75];
    const { estimate, lower, upper, studyEstimates } = randomEffectsPooledMean(ns, means, sds);
    assert.ok(within(estimate, 14.996));
    assert.ok(within(lower, 8.955));
    assert.ok(within(upper, 21.036));

    assert.strictEqual(studyEstimates.length, 3);
    assert.ok(within(studyEstimates[1].estimate, 15));
    assert.ok(within(studyEstimates[1].lower, 14.528));
    assert.ok(within(studyEstimates[1].upper, 15.471));
  });

  /*
   library(meta)
   means <- c(10, -5, 20)
   sds <- c(1, 2, 2.4)
   ns <- c(1000, 50, 75)
   mm <- metamean(ns, means, sds)
  */
  it('should function properly with negative means', function() {
    const means = [10, -5, 20];
    const sds = [1, 2, 2.4];
    const ns = [1000, 50, 75];
    const { estimate, lower, upper } = randomEffectsPooledMean(ns, means, sds);
    assert.ok(within(estimate, 8.3340));
    assert.ok(within(lower, -1.9188));
    assert.ok(within(upper, 18.5869));
  });

  /*
   library(meta)
   means <- c(10)
   sds <- c(1)
   ns <- c(1000)
   mm <- metamean(ns, means, sds)
  */
  it('should handle a single point estimate', function() {
    const means = [10];
    const sds = [1];
    const ns = [1000];
    const { estimate, lower, upper } = randomEffectsPooledMean(ns, means, sds);
    assert.ok(within(estimate, 10));
    assert.ok(within(lower, 9.94));
    assert.ok(within(upper, 10.06));
  });

  it('should handle no point estimates', function() {
    const x = randomEffectsPooledMean([], [], []);
    assert.deepStrictEqual(x, {studyEstimates: []});
  });

  it('should throw on missing critical values', function() {
    assert.throws(() => randomEffectsPooledMean([10, 20], [undefined, 100], [undefined, 5]),
      {
        message: 'Element at index 0 is missing in mean'
      });

    assert.throws(() => randomEffectsPooledMean([10, undefined], [110, 100], [undefined, 5]),
      {
        message: 'Element at index 1 is missing in n'
      });
  });

  it('should throw on negative sd', function() {
    assert.throws(() => randomEffectsPooledMean([10, 20], [95, 100], [-1, 5]),
      {
        message: 'Element at index 0 is non-positive in sd'
      });
  });

  it('should throw on non-positive n', function() {
    assert.throws(() => randomEffectsPooledMean([10, 20], [95, 100], [-1, 5]),
      {
        message: 'Element at index 0 is non-positive in sd'
      });
  });

  it('should throw on invalid width', function() {
    assert.throws(() => randomEffectsPooledMean([10, 20], [95, 100], [1, 5], 95),
      {
        message: 'Value width not in range 0 to 1'
      });
  });
});

describe('Arithmetic mean pooling', function () {
  it('should produce correct results for full data', function () {
    const means = [10, 15, 20];
    const ns = [10, 20, 100];
    const { estimate } = pooledMean(ns, means);
    assert.ok(within(estimate,18.461));
  });


  it('should throw on missing critical values', function() {
    assert.throws(() => pooledMean([10, 20], [undefined, 100]),
      {
        message: 'Element at index 0 is missing in mean'
      });

    assert.throws(() => pooledMean([10, undefined], [110, 100]),
      {
        message: 'Element at index 1 is missing in n'
      });
  });

  it('should handle no point estimates', function() {
    const x = pooledMean([], []);
    assert.deepStrictEqual(x, {});
  });

  it('should handle a single point estimate', function() {
    const { estimate } = pooledMean([10], [50]);
    assert.strictEqual(estimate, 50);
  });
});

describe('Random effects rate pooling', function () {
  /*
    replicate expected estimates in r with:
    library(meta)
    events_p  <- c(10, 15, 20, 24, 25, 39, 10)
    ns_p <- c(50, 65, 90,  100, 95, 150, 160)
    mp_i <- metaprop(events_p, ns_p, method='Inverse')
  */
  it('should produce correct results for full data', function () {
    const events = [10, 15, 20, 24, 25, 39, 10];
    const ns = [50, 65, 90,  100, 95, 150, 160];
    const { estimate, lower, upper, studyEstimates } = randomEffectsPooledRate(ns, events);
    assert.ok(within(estimate, .204));
    assert.ok(within(lower, .15));
    assert.ok(within(upper, .27));

    assert.strictEqual(studyEstimates.length, 7);
    // note that since we are using a normal approximation over a beta, our CIs will be shifted left a bit more than is proper
    // I believe that drives the difference ~1% differences in what `meta` is giving
    assert.ok(within(studyEstimates[0].estimate, .2));
    assert.ok(within(studyEstimates[0].lower, .09));
    assert.ok(within(studyEstimates[0].upper, .31));

    assert.ok(within(studyEstimates[6].estimate, .0625));
    assert.ok(within(studyEstimates[6].lower, .0250));
    assert.ok(within(studyEstimates[6].upper, .100));
  });

  /*
    library(meta)
    events_p  <- c(50, 20, 10, 0)
    ns_p <- c(100, 100, 50, 40)
    mp_i <- metaprop(events_p, ns_p, method='Inverse')
  */
  it('should handle 0 event counts', function() {
    const events = [50, 20, 10, 0];
    const ns = [100, 100, 50, 40];
    const { estimate, lower, upper } = randomEffectsPooledRate(ns, events);
    assert.ok(within(estimate, .222));
    assert.ok(within(lower, .090));
    assert.ok(within(upper, .451));
  });

  /*
    library(meta)
    events_p  <- c(100, 85, 50, 30)
    ns_p <- c(100, 100, 50, 40)
    mp_i <- metaprop(events_p, ns_p, method='Inverse')
  */
  it('should handle event = total counts', function() {
    const events = [100, 85, 50, 30];
    const ns = [100, 100, 50, 40];
    const { estimate, lower, upper } = randomEffectsPooledRate(ns, events);
    assert.ok(within(estimate, .913));
    assert.ok(within(lower, .756));
    assert.ok(within(upper, .972));
  });

  /*
    library(meta)
    events_p  <- c(50)
    ns_p <- c(100)
    mp_i <- metaprop(events_p, ns_p, method='Inverse')
  */
  it('should handle a single point estimate', function() {
    const { estimate, lower, upper} = randomEffectsPooledRate([100], [50]);
    assert.ok(within(estimate, .5));
    assert.ok(within(lower, .403));
    assert.ok(within(upper, .597));
  });

  /*
  library(meta)
  events_p  <- c(5, 1)
  ns_p <- c(86, 24)
  mp_i <- metaprop(events_p, ns_p, method='Inverse')

  this test case is useful for asserting that we cap our tau^2 estimator at 0
 */
  it('should handle a small number of observations', function() {
    const { estimate, lower, upper} = randomEffectsPooledRate([86, 24], [5, 1]);
    assert.ok(within(estimate, .055));
    assert.ok(within(lower, .0249));
    assert.ok(within(upper, .117));
  });

  it('should handle no point estimates', function() {
    const x = randomEffectsPooledRate([], []);
    assert.deepStrictEqual(x, {studyEstimates: []})
  });

  it('should throw on missing critical values', function() {
    assert.throws(() => randomEffectsPooledRate([10, 20], [5, undefined]),
      {
        message: 'Element at index 1 is missing in events',
      });

    assert.throws(() => randomEffectsPooledRate([10, undefined], [5, 10]),
      {
        message: 'Element at index 1 is missing in n',
      });
  });

  it('should throw on event > n', function() {
    assert.throws(() => randomEffectsPooledRate([10, 20], [9, 21]),
      {
        message: 'event is greater than n at index 1',
      });
  });

  it('should throw on non-positive n', function() {
    assert.throws(() => randomEffectsPooledRate([-10, 20], [-11, 9]),
      {
        message: 'Element at index 0 is non-positive in n',
      });
  });

  it('should throw on 0 n', function() {
    assert.throws(() => randomEffectsPooledRate([0, 20], [0, 9]),
      {
        message: 'Element at index 0 is non-positive in n',
      });
  });
});

describe('Median Pooling', function () {
  /*
    reproduce in R with the below. note we expect some leeway, since our quantile method differs
    library(metamedian)
    pool.med(c(10, 15, 17, 20, 16.5, 14), c(500, 500, 75, 400, 250, 300), TRUE, .95)
   */
  it('should produce correct results for toy data', function () {
    const medians = [10, 15, 17, 20, 16.5, 14];
    const ns = [500, 500, 75, 400, 250, 300];
    const { estimate, lower, upper } = pooledMedian(ns, medians, .95);
    assert.ok(within(estimate, 15, 1));
    assert.ok(within(lower, 10, 1));
    assert.ok(within(upper, 19.5, 1));
  });

  /*
    reproduce in R with the below. note we expect some leeway, since our quantile method differs
    library(metamedian)
    pool.med(c(10, 15, 17, 20, 16.5, 14, 22, 23, 11, 17, 21), c(500, 500, 75, 400, 250, 300, 11, 300, 250, 200, 225), TRUE, .95)
   */
  it('should produce correct results for full data', function () {
    const medians = [10, 15, 17, 20, 16.5, 14, 22, 23, 11, 17, 21];
    const ns = [500, 500, 75, 400, 250, 300, 11, 300, 250, 200, 225];
    const { estimate, lower, upper } = pooledMedian(ns, medians, .95);
    assert.ok(within(estimate, 15, 1));
    assert.ok(within(lower, 11, 1));
    assert.ok(within(upper, 19.5, 1));
  });

  it('should handle no point estimates', function() {
    const x = pooledMedian([], []);
    assert.deepStrictEqual(x, {});
  });

  it('should handle a single point estimate', function() {
    const { estimate, lower, upper } = pooledMedian([100], [50]);
    assert.ok(within(estimate, 50));
    // this is obviously an incorrect estimate, but it's the functionality we'll go with for a degenerate input
    assert.ok(within(lower, 50));
    assert.ok(within(upper, 50));
  });

  it('should throw on missing critical values', function() {
    assert.throws(() => pooledMedian([10, 20], [5, undefined]),
      {
        message: 'Element at index 1 is missing in median',
      });

    assert.throws(() => pooledMedian([10, undefined], [5, 10]),
      {
        message: 'Element at index 1 is missing in n',
      });
  });

  it('should throw on non-positive n', function() {
    assert.throws(() => pooledMedian([10, -10], [5, -15]),
      {
        message: 'Element at index 1 is non-positive in n',
      });
  });
})