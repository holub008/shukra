const assert = require('assert');
const {NetworkMetaAnalysis, fixedEffectsOddsRatioNMA, fixedEffectsMeanDifferenceNMA} = require('../src/nma');
const {pooledMedian, randomEffectsPooledMean, randomEffectsPooledRate} = require('../src/pooling');
const {Matrix} = require('ml-matrix');

/**
 * tests for NMA module
 */

describe('NMA Holder Class', function () {
  const trt = new Matrix([[0, 5], [-5, 0]]);
  const se = new Matrix([[0, 2], [2, 0]]);
  const trtLabel = ['Band-aid', 'Stitch'];
  const meanDiffNMA = new NetworkMetaAnalysis(trt, se, trtLabel);

  it('should echo treatment effect', function () {
    assert.strictEqual(meanDiffNMA.getEffect('Band-aid', 'Stitch'), 5);
    assert.strictEqual(meanDiffNMA.getEffect('Stitch', 'Band-aid'), -5);
  });

  const inferentialStats = meanDiffNMA.computeInferentialStatistics('Band-aid', 'Stitch', .95);

  it('should compute inferential statistics', function () {
    // determined via the following r code:
    // 2 * pnorm(0, 5, 2)
    assert.ok(inferentialStats.p > .0124 && inferentialStats.p < .0125);
    // qnorm(.975, 5, 2)
    assert.ok(inferentialStats.upperBound > 8.91 && inferentialStats.upperBound < 8.92);
    // qnorm(.025, 5, 2)
    assert.ok(inferentialStats.lowerBound > 1.08 && inferentialStats.lowerBound < 1.09);
  });

  it('should complain if you ask for a non-existent treatment', function () {
    assert.throws(() => meanDiffNMA.getEffect("Super glue", "Stitch"));
    assert.throws(() =>
      meanDiffNMA.computeInferentialStatistics("Stitch", "Super glue", .95));
  });
});

describe('Odds Ratio FE NMA preconditions', function () {
  const study = [1, 1, 2, 2, 2];
  const treatmentBad = ["A", "C", "D", "B", "D"]; // duplicates treatment D in study 2
  const treatmentGood = ["A", "C", "B", "C", "D"];
  const positiveCountBad = [5000, 23, 10, 11, 12]; // larger than total count
  const positiveCountGood = [9, 23, 10, 11, 12];
  const totalCountBad = [140, 140, 138, 78]; // too short
  const totalCountGood = [140, 140, 138, 78, 85];

  it('succeeds with healthy data', function () {
    const nma = fixedEffectsOddsRatioNMA(study, treatmentGood, positiveCountGood, totalCountGood);
    assert.strictEqual(nma.getTreatments().length, 4);
  });

  it('should not produce a model for duplicate treatment arms', function () {
    assert.throws(() => fixedEffectsOddsRatioNMA(study, treatmentBad, positiveCountGood, totalCountGood));
  });

  it('should not produce a model for unequal input lengths', function () {
    assert.throws(() => fixedEffectsOddsRatioNMA(study, treatmentGood, positiveCountGood, totalCountBad));
  });

  it('should not produce a model for unequal input lengths', function () {
    assert.throws(() => fixedEffectsOddsRatioNMA(study, treatmentGood, positiveCountBad, totalCountGood));
  });
});

describe('Odds Ratio FE NMA', function () {
  // data originates from:
  // Dias S, Welton NJ, Sutton AJ, Caldwell DM, Lu G and Ades AE (2013): Evidence Synthesis for Decision Making 4: Inconsistency in networks of evidence based on randomized controlled trials. Medical Decision Making, 33, 641–56
  /** generated with R code:
   sc <- smokingcessation %>% mutate_at(c('treat1', 'treat2', 'treat3'), as.character)
   long<- lapply(1:nrow(sc), function(ix) {
              row <- sc[ix,]

              res <-data.frame(
                treatment = c(row$treat1, row$treat2),
                positiveCounts = c(row$event1, row$event2),
                totalCounts = c(row$n1, row$n2),
                study = ix,
                stringsAsFactors = FALSE
              )

              if (row$treat3 != '') {
                res <- res %>% rbind(list(
                  treatment = row$treat3,
                  positiveCounts = row$event3,
                  totalCounts = row$n3,
                  study = ix),
                  stringsAsFactors = FALSE)
              }

              res
        }) %>%
   bind_rows()

   cat(paste0(long$treatment, collapse='", "'))
   paste0(long$positiveCounts, collapse=', ')
   paste0(long$totalCounts, collapse=', ')
   paste0(long$study, collapse=', ')
   */

  const treatment = ["A", "C", "D", "B", "C", "D", "A", "C", "A", "C", "A", "C", "A", "C", "A", "C", "A", "C", "A",
    "C", "A", "B", "A", "B", "A", "C", "A", "C", "A", "C", "A", "D", "A", "B", "A", "C", "A", "C", "A", "C", "A",
    "C", "B", "C", "B", "D", "C", "D", "C", "D"];
  const positiveCount = [9, 23, 10, 11, 12, 29, 75, 363, 2, 9, 58, 237, 0, 9, 3, 31, 1, 26, 6, 17, 79, 77, 18, 21,
    64, 107, 5, 8, 20, 34, 0, 9, 8, 19, 95, 143, 15, 36, 78, 73, 69, 54, 20, 16, 7, 32, 12, 20, 9, 3];
  const totalCount = [140, 140, 138, 78, 85, 170, 731, 714, 106, 205, 549, 1561, 33, 48, 100, 98, 31, 95, 39, 77,
    702, 694, 671, 535, 642, 761, 62, 90, 234, 237, 20, 20, 116, 149, 1107, 1031, 187, 504, 584, 675, 1177, 888, 49,
    43, 66, 127, 76, 74, 55, 26];
  // note, these could just as well be integers
  const study = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
    15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24];

  const nma = fixedEffectsOddsRatioNMA(study, treatment, positiveCount, totalCount);

  it('should produce reasonable effect size estimates', function () {
    /** checked via the following R code:
     p2 <- pairwise(treat = list(treat1, treat2, treat3), event = list(event1, event2, event3),
     n = list(n1, n2, n3), data = smokingcessation, sm = "OR", addincr=TRUE)
     net2 <- netmeta(TE, seTE, treat1, treat2, studlab, data = p2, comb.fixed = TRUE, comb.random = FALSE)
     summary(net2)
     */
    const ab = nma.getEffect("A", "B");
    const ba = nma.getEffect("B", "A");
    assert.ok(ab > .81 && ab < .82);
    assert.ok(ba > 1.22 && ba < 1.23);

    const aa = nma.getEffect("A", "A");
    assert.ok(Math.abs(aa - 1) < .00001);

    const db = nma.getEffect("D", "B");
    assert.ok(db > 1.64 && db < 1.65);
  });

  it('should produce reasonable inferential statistics', function () {
    const cb = nma.computeInferentialStatistics("C", "B", .95);
    assert.ok(cb.lowerBound > 1.20 && cb.lowerBound < 1.21);
    assert.ok(cb.upperBound > 2.02 && cb.upperBound < 2.03);
    assert.ok(cb.p < .01);

    const ab = nma.computeInferentialStatistics("A", "B", .95);
    assert.ok(ab.lowerBound > .635 && ab.lowerBound < .645);
    assert.ok(ab.upperBound > 1.04 && ab.upperBound < 1.05);
    assert.ok(ab.p > .05);

    const ab99 = nma.computeInferentialStatistics("A", 'B', .99);
    assert.ok(ab.lowerBound > ab99.lowerBound && ab.upperBound < ab99.upperBound);
  });
});

describe('Mean Difference FE NMA', function () {
  // this is a faux dataset
  const studies = ['A', 'A', 'B', 'B', 'B', 'C', 'C'];
  const trts = [1, 2, 1, 2, 3, 3, 2];
  const means = [8, 10, 7, 10.5, 10.5, 10, 11];
  const sds = [4.923423, 3.867062, 3.250787, 6.349051, 6.664182, 4.324474, 4.301156];
  const ns = [63, 45, 35, 44, 53, 75, 29];

  const nma = fixedEffectsMeanDifferenceNMA(studies, trts, means, sds, ns);

  it('should produce reasonable effect size estimates', function () {
    /** checked via the following R code
     set.seed(55414)
     data <- data.frame(
     mean = c(8, 10, 7, 10.5, 10.5, 10, 11),
     sd = abs(rnorm(7, 5)),
     studlab = c('A', 'A', 'B', 'B', 'B', 'C', 'C'),
     treat = c(1, 2, 1, 2, 3, 3, 2),
     n = round(rnorm(7, 50, 10))
     )

     p <- with(data, pairwise(treat, mean=mean, sd=sd, studlab = studlab, n=n))
     net <- netmeta(p$TE, p$seTE, p$treat1, p$treat2, p$studlab, sm='MD', comb.fixed = TRUE, comb.random = FALSE)
     summary(net)
     */

    const oneTwo = nma.getEffect(1, 2);
    const twoOne = nma.getEffect(2, 1);

    assert.ok(oneTwo < -2.818 && oneTwo > -2.819);
    assert.ok(twoOne > 2.818 && twoOne < 2.819);

    const twoTwo = nma.getEffect(1, 1);
    assert.ok(Math.abs(twoTwo) < .00001);

    const threeTwo = nma.getEffect(3, 2);
    assert.ok(threeTwo < -0.312 && threeTwo > -0.313)
  });

  it('should produce reasonable inferential statistics', function () {
    const oneThree95 = nma.computeInferentialStatistics(1, 3, .95);
    assert.ok(oneThree95.lowerBound > -4.095 && oneThree95.lowerBound < -4.09);
    assert.ok(oneThree95.upperBound > -0.918 && oneThree95.upperBound < -0.917);
    assert.ok(oneThree95.p < .05);

    const twoThree95 = nma.computeInferentialStatistics(2, 3, .95);
    assert.ok(twoThree95.lowerBound > -1.12 && twoThree95.lowerBound < -1.11);
    assert.ok(twoThree95.upperBound > 1.735 && twoThree95.upperBound < 1.745);
    assert.ok(twoThree95.p > .05);
  });

});

describe('Errors for degenerate inputs', function () {
  const studies = [10, 11, 12];
  const treatments = [1, 2, 1];
  const positive = [10, 12, 15];
  const total = [13, 20, 35];

  // shukra is already generous in taking (and ignoring) single arm studies- but if 0 contrasts are available,
  // absolutely nothing can be done
  it('should throw if 0 contrasts are present', function () {
    assert.throws(() => fixedEffectsOddsRatioNMA(studies, treatments, positive, total));
  });
});


/**
 * tests for pooling module
 */
/**
 describe('Random effects mean pooling', function() {
    /**
 * replicate in r with:
     library(meta)
     means <- c(10, 15, 20)
     sds <- c(1, 2, 2.4)
     ns <- c(1000, 50, 75)
     mm <- metamean(ns, means, sds)

 *//**
 it('should produce correct results for full data', function() {
        const means = [10, 15, 20];
        const sds = [1, 2, 2.4];
        const ns = [1000, 50, 75];
        assert.throws(() => fixedEffectsOddsRatioNMA(studies, treatments, positive, total));
    });

 it('should produce correct results with missing sds', function() {
        const means = [10, 15, 20];
        const sds = [1, 2, 2.4];
        const ns = [1000, 50, 75];
        assert.throws(() => fixedEffectsOddsRatioNMA(studies, treatments, positive, total));
    });
 });
 */

describe('Median Pooling', function () {
  it('should produce correct results for full data', function () {
    const medians = [10, 15, 17, 20, 16.5, 14];
    const ns = [500, 500, 75, 400, 250, 300];
    const estimates = pooledMedian(ns, medians, .95);
    console.log(estimates);
    assert.ok(false);
  });
})