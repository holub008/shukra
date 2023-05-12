const assert = require('assert');
const { NetworkMetaAnalysis, oddsRatioNMA, meanDifferenceNMA } = require('../src/nma');
const { Matrix } = require('ml-matrix');
const {ComparisonStatistic} = require("../src");

/**
 * tests for NMA module
 */

describe('NMA Holder Class', function () {
  const trt = new Matrix([[0, 5], [-5, 0]]);
  const se = new Matrix([[0, 2], [2, 0]]);
  const trtLabel = ['Band-aid', 'Stitch'];
  const stubStudyLevelEffects = [];
  const meanDiffNMA = new NetworkMetaAnalysis(trt, se, trtLabel, stubStudyLevelEffects, ComparisonStatistic.MD, 202.3334, 23);

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
    assert.ok(inferentialStats.upper > 8.91 && inferentialStats.upper < 8.92);
    // qnorm(.025, 5, 2)
    assert.ok(inferentialStats.lower > 1.08 && inferentialStats.lower < 1.09);
  });

  it('should complain if you ask for a non-existent treatment', function () {
    assert.throws(() => meanDiffNMA.getEffect("Super glue", "Stitch"));
    assert.throws(() =>
      meanDiffNMA.computeInferentialStatistics("Stitch", "Super glue", .95));
  });

  it('should report correct heterogeneity stats', function() {
    /** checked via the following R code:
     p2 <- pairwise(treat = list(treat1, treat2, treat3), event = list(event1, event2, event3),
     n = list(n1, n2, n3), data = smokingcessation, sm = "OR", addincr=TRUE)
     net2 <- netmeta(TE, seTE, treat1, treat2, studlab, data = p2, comb.fixed = TRUE, comb.random = FALSE)
     c(net2$I2, net2$lower.I2, net2$upper.I2)
     */
    const i2Stats = meanDiffNMA.computeISquared()
    assert.ok(i2Stats.i2 > 0.886 && i2Stats.i2 < 0.887);
    assert.ok(i2Stats.lower > 0.843 && i2Stats.lower < 0.844);
    assert.ok(i2Stats.upper > 0.917 && i2Stats.upper < 0.918);
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
    const nma = oddsRatioNMA(study, treatmentGood, positiveCountGood, totalCountGood, false);
    assert.strictEqual(nma.getTreatments().length, 4);
  });

  it('should not produce a model for duplicate treatment arms', function () {
    assert.throws(() => oddsRatioNMA(study, treatmentBad, positiveCountGood, totalCountGood, false));
  });

  it('should not produce a model for unequal input lengths', function () {
    assert.throws(() => oddsRatioNMA(study, treatmentGood, positiveCountGood, totalCountBad, false));
  });

  it('should not produce a model for unequal input lengths', function () {
    assert.throws(() => oddsRatioNMA(study, treatmentGood, positiveCountBad, totalCountGood, false));
  });
});

describe('Odds Ratio FE NMA', function () {
  // data originates from:
  // Dias S, Welton NJ, Sutton AJ, Caldwell DM, Lu G and Ades AE (2013): Evidence Synthesis for Decision Making 4: Inconsistency in networks of evidence based on randomized controlled trials. Medical Decision Making, 33, 641–56
  /** generated with R code:
   data(smokingcessation)
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
  const study = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
    15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24];


  it('should produce reasonable fixed effects effect size estimates', function () {
    /** checked via the following R code:
     p2 <- pairwise(treat = list(treat1, treat2, treat3), event = list(event1, event2, event3),
     n = list(n1, n2, n3), data = smokingcessation, sm = "OR", addincr=TRUE)
     net2 <- netmeta(TE, seTE, treat1, treat2, studlab, data = p2, comb.fixed = TRUE, comb.random = FALSE)
     summary(net2)
     */

    const nmaFE = oddsRatioNMA(study, treatment, positiveCount, totalCount, false);

    const ab = nmaFE.getEffect("A", "B");
    const ba = nmaFE.getEffect("B", "A");
    assert.ok(ab > .81 && ab < .82);
    assert.ok(ba > 1.22 && ba < 1.23);

    const aa = nmaFE.getEffect("A", "A");
    assert.ok(Math.abs(aa - 1) < .00001);

    const db = nmaFE.getEffect("D", "B");
    assert.ok(db > 1.64 && db < 1.65);
  });

  it('should produce reasonable random effects effect size estimates', function () {
    /** checked via the following R code:
     net1 <- netmeta(TE, seTE, treat1, treat2, studlab, data = p2, comb.fixed = FALSE, comb.random = TRUE)
     summary(net1)
     */
    const nmaRE = oddsRatioNMA(study, treatment, positiveCount, totalCount, true);
    const ab = nmaRE.getEffect("A", "B");
    const ba = nmaRE.getEffect("B", "A");

    assert.ok(ba > 1.50 && ba < 1.51);
    assert.ok(ab > 0.66 && ab < 0.67);

    const aa = nmaRE.getEffect("A", "A");
    assert.ok(Math.abs(aa - 1) < .00001);

    const db = nmaRE.getEffect("D", "B");
    assert.ok(db > 1.595 && db < 1.605);
  });

  it('should produce reasonable fixed effects inferential statistics', function () {
    const nmaFE = oddsRatioNMA(study, treatment, positiveCount, totalCount, false);

    const cb = nmaFE.computeInferentialStatistics("C", "B", .95);
    assert.ok(cb.lower > 1.20 && cb.lower < 1.21);
    assert.ok(cb.upper > 2.02 && cb.upper < 2.03);
    assert.ok(cb.p < .01);

    const ab = nmaFE.computeInferentialStatistics("A", "B", .95);
    assert.ok(ab.lower > .635 && ab.lower < .645);
    assert.ok(ab.upper > 1.04 && ab.upper < 1.05);
    assert.ok(ab.p > .05);

    const ab99 = nmaFE.computeInferentialStatistics("A", 'B', .99);
    assert.ok(ab.lower > ab99.lower && ab.upper < ab99.upper);
  });

  it('should produce reasonable random effects inferential statistics', function () {
    const nmaRE = oddsRatioNMA(study, treatment, positiveCount, totalCount, true);

    const ba = nmaRE.computeInferentialStatistics("B", "A", .95);
    assert.ok(ba.lower > 0.73 && ba.lower < 0.74);
    assert.ok(ba.upper > 3.06 && ba.upper < 3.07);
    assert.ok(ba.p < 0.28 && ba.p > 0.25);
  });

  it('should produce fixed effects study level effects', function() {
    const nmaFE = oddsRatioNMA(study, treatment, positiveCount, totalCount, false);

    const effectsA = nmaFE.computeStudyLevelEffects('A');
    /*
      sum(p2$treat1 == 'A')
     */
    assert.deepStrictEqual(effectsA.length, 20);
    const effectsA1C = effectsA.filter(({ study, treatment2 }) => study  === 1 && treatment2 === 'C');
    assert.deepStrictEqual(effectsA1C.length, 1);
    /*
      effect <- exp(p2$TE[1]) # should match effect below (note these effects are anscambe corrected .5
      log_effect <- p2$TE[1]
      se <- p2$seTE[1]
      error <- qnorm(0.975)*se
      exp(c(log_effect - error, log_effect + error)) # should match lower,upper
     */
    assert.deepStrictEqual(effectsA1C[0], {
      p: 0.011903876743789787,
      lower: 0.1633538803799905,
      upper: 0.7987415160010132,
      effect:  0.36121673003802274,
      comparisonN: 280,
      treatment1: 'A',
      treatment2: 'C',
      study: 1
    });
  });

  it('should produce random effects study level effects', function() {
    const nmaRE = oddsRatioNMA(study, treatment, positiveCount, totalCount, true);

    const effectsA = nmaRE.computeStudyLevelEffects('A');
    assert.deepStrictEqual(effectsA.length, 20);
    const effectsA1C = effectsA.filter(({ study, treatment2 }) => study === 1 && treatment2 === 'C');
    assert.deepStrictEqual(effectsA1C.length, 1);
    /*
      effect <- exp(p2$TE[1]) # should match effect below (note these effects are anscambe corrected .5
      log_effect <- p2$TE[1]
      se <- p2$seTE[1]
      error <- qnorm(0.975)*se
      exp(c(log_effect - error, log_effect + error)) # should match lower,upper
     */
    assert.deepStrictEqual(effectsA1C[0], {
      p:  0.011903876743789787,
      lower: 0.1633538803799905,
      upper: 0.7987415160010132,
      effect: 0.36121673003802274,
      comparisonN: 280,
      treatment1: 'A',
      treatment2: 'C',
      study: 1
    });
  });

  it('should produce valid heterogeneity statistics', function () {
    const nmaFE = oddsRatioNMA(study, treatment, positiveCount, totalCount, false);
    const nmaRE = oddsRatioNMA(study, treatment, positiveCount, totalCount, true);

    const ris = nmaRE.computeISquared()
    assert.deepStrictEqual(ris, {
      i2: 0.8863262063832255,
      lower: 0.8437926735418156,
      upper: 0.9172783271552367,
    });
    const fis = nmaFE.computeISquared();
    assert.deepStrictEqual(fis, ris);
  });

  it('should compute accurate comparison adjusted stats', function() {
    /* checked with R code (!note, effects will differ a bit due to using indirect evidence in the adjustment calculation)
      r <- funnel(net2, order='B', studlab=T, level=.99)
      library(jsonlite)
      r %>%
        mutate(
          study = studlab,
          treatment1 = treat2,
          treatment2 = treat1,
          effect = exp(TE.adj),
          se = seTE,
          study = as.integer(study)
        ) %>%
        select(study, treatment1, treatment2, effect, se) %>%
        arrange(study, treatment1) %>%
        toJSON()
     */
    const nmaFE = oddsRatioNMA(study, treatment, positiveCount, totalCount, false);
    const { effects, leftFunnel, rightFunnel, asymmetryP, asymmetryTest  } = nmaFE.computeComparisonAdjustedEffects('B');
    const orderedStudies = effects
      .sort((a, b) => a.treatment2 < b.treatment2 ? -1 : 1)
      .sort((a, b) => a.study < b.study ? -1 : 1);
    // this has been changed from netmeta: https://github.com/guido-s/netmeta/issues/11
    assert.deepStrictEqual(orderedStudies, [
      {"study":2,"treatment1":"B","treatment2":"C","effect":0.6395259758462293,"se":0.44201445534146955},
      {"study":2,"treatment1":"B","treatment2":"D","effect":0.7422349294922779,"se":0.3778052136324726},
      {"study":10,"treatment1":"B","treatment2":"A","effect":1.2422745360612997,"se":0.16942586755039812},
      {"study":11,"treatment1":"B","treatment2":"A","effect":0.8283706607044433,"se":0.32255160585540343},
      {"study":16,"treatment1":"B","treatment2":"A","effect":0.641086661173091,"se":0.4310551417352239},
      {"study":21,"treatment1":"B","treatment2":"C","effect":0.5531577328382871,"se":0.4238496075960444},
      {"study":22,"treatment1":"B","treatment2":"D","effect":1.6375914850930329,"se":0.43747056012961766}
    ]);

    assert.deepStrictEqual(leftFunnel.length, 500);
    // strictly decreasing in effect size
    assert.deepStrictEqual(leftFunnel, leftFunnel.sort((a, b) => a[0] > b[0] ? -1 : 1))
    // strictly increasing in SE
    assert.deepStrictEqual(leftFunnel, leftFunnel.sort((a, b) => a[1] > b[1] ? 1 : -1))
    // exp(ci(log(1), .44201445534146955, .95)$lower)
    // where .442014 is our max SE
    assert.deepStrictEqual(leftFunnel[leftFunnel.length - 1],  [0.4204909185483156, 0.44201445534146955]);
    assert.deepStrictEqual(leftFunnel[0],  [1, 0]);

    assert.deepStrictEqual(rightFunnel.length, 500);
    // strictly decreasing in effect size
    assert.deepStrictEqual(rightFunnel, rightFunnel.sort((a, b) => a[0] > b[0] ? -1 : 1))
    // strictly increasing in SE
    assert.deepStrictEqual(rightFunnel, rightFunnel.sort((a, b) => a[1] > b[1] ? 1 : -1))
    // exp(ci(log(1), .44201445534146955, .95)$upper)
    // where .442014 is our max SE
    assert.deepStrictEqual(rightFunnel[rightFunnel.length - 1],  [2.3781726450891165, 0.44201445534146955]);
    assert.deepStrictEqual(rightFunnel[0],  [1, 0]);
  });

  it('should perform correct Egger Tests', function() {
    /* validate:
      # close, but not equivalent, since we are using NMA effects for adjustment, not direct effects
      funnel(net2, order='A, method.bias='egger')
      funnel(net2, order='C, method.bias='egger')
     */
    const nmaFE = oddsRatioNMA(study, treatment, positiveCount, totalCount, false);
    const { asymmetryP, asymmetryTest  } = nmaFE.computeComparisonAdjustedEffects('A');
    assert.deepStrictEqual(asymmetryP, 0.4483014839311841);
    assert.deepStrictEqual(asymmetryTest, 'Egger');

    const { asymmetryP: ap2, asymmetryTest: at2  } = nmaFE.computeComparisonAdjustedEffects('C');
    assert.deepStrictEqual(ap2, 0.9673980987346369);
    assert.deepStrictEqual(at2, 'Egger');
  })
});

describe('Mean Difference NMA', function () {
  // this is a faux dataset
  const studies = ['A', 'A', 'B', 'B', 'B', 'C', 'C'];
  const trts = [1, 2, 1, 2, 3, 3, 2];
  const means = [8, 10, 7, 10.5, 10.5, 10, 11];
  const sds = [4.923423, 3.867062, 3.250787, 6.349051, 6.664182, 4.324474, 4.301156];
  const ns = [63, 45, 35, 44, 53, 75, 29];

  it('should produce reasonable fixed effects effect size estimates', function () {
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

    const nmaFE = meanDifferenceNMA(studies, trts, means, sds, ns, false);

    const oneTwo = nmaFE.getEffect(1, 2);
    const twoOne = nmaFE.getEffect(2, 1);

    assert.ok(oneTwo < -2.818 && oneTwo > -2.819);
    assert.ok(twoOne > 2.818 && twoOne < 2.819);

    const twoTwo = nmaFE.getEffect(1, 1);
    assert.ok(Math.abs(twoTwo) < .00001);

    const threeTwo = nmaFE.getEffect(3, 2);
    assert.ok(threeTwo < -0.312 && threeTwo > -0.313);
  });

  it('should produce reasonable random effects effect size estimates', function () {
    /** checked via the following R code
     net2 <- netmeta(p$TE, p$seTE, p$treat1, p$treat2, p$studlab, sm='MD', comb.fixed = FALSE, comb.random = TRUE)
     summary(net2)
     */
    const nmaRE = meanDifferenceNMA(studies, trts, means, sds, ns, true);
    const oneTwo = nmaRE.getEffect(1, 2);

    assert.ok(oneTwo > -2.848 && oneTwo < -2.847);

    const twoTwo = nmaRE.getEffect(1, 1);
    assert.ok(Math.abs(twoTwo) < .00001);

    const threeTwo = nmaRE.getEffect(3, 2);
    assert.ok(threeTwo > -0.308 && threeTwo < -0.307)
  });

  it('should produce reasonable fixed effects inferential statistics', function () {
    const nmaFE = meanDifferenceNMA(studies, trts, means, sds, ns, false);
    const oneThree95 = nmaFE.computeInferentialStatistics(1, 3, .95);
    assert.ok(oneThree95.lower > -4.095 && oneThree95.lower < -4.09);
    assert.ok(oneThree95.upper > -0.918 && oneThree95.upper < -0.917);
    assert.ok(oneThree95.p < .05);

    const twoThree95 = nmaFE.computeInferentialStatistics(2, 3, .95);
    assert.ok(twoThree95.lower > -1.12 && twoThree95.lower < -1.11);
    assert.ok(twoThree95.upper > 1.735 && twoThree95.upper < 1.745);
    assert.ok(twoThree95.p > .05);
  });

  it('should produce reasonable random effects inferential statistics', function () {
    const nmaRE = meanDifferenceNMA(studies, trts, means, sds, ns, true);
    const oneThree95 = nmaRE.computeInferentialStatistics(1, 3, .95);
    assert.ok(oneThree95.lower > -4.33 && oneThree95.lower < -4.32);
    assert.ok(oneThree95.upper > -0.76 && oneThree95.upper < -0.75);
    assert.ok(oneThree95.p < .05);
  });

  it('should produce study level effects', function() {
    const nmaFE = meanDifferenceNMA(studies, trts, means, sds, ns, false);
    const effects = nmaFE.computeStudyLevelEffects(1);
    assert.deepStrictEqual(effects.length, 3);

    /* verify with R code:
      se <- p$seTE[1]
      effect <- p$TE[1] # -2
      error <- qnorm(0.975)*se
      c(effect - error, effect + error) # should match lower,upper
     */
    const effectsA = effects.filter(({ study }) => study === 'A');
    assert.deepStrictEqual(effectsA.length, 1);
    assert.deepStrictEqual(effectsA[0],   {
      p: 0.018185488496468505,
      lower: -3.659706743960371,
      upper: -0.340293256039629,
      effect: -2,
      comparisonN: 108,
      treatment1: 1,
      treatment2: 2,
      study: 'A'
    },);

    const effects2 = nmaFE.computeStudyLevelEffects(2);
    assert.deepStrictEqual(effects2.length, 4);
    /* verify with R code:
      se <- p$seTE[5]
      effect <- -p$TE[5] # we invert since netmeta uses 3 as the reference
      error <- qnorm(0.975)*se
      c(effect - error, effect + error) # should match lower,upper
     */
    const effects2C = effects2.filter(({ study }) => study === 'C');
    assert.deepStrictEqual(effects2C.length, 1);
    assert.deepStrictEqual(effects2C[0],   {
        p: 0.28840673752917056,
        lower: -0.8461952560237911,
        upper: 2.8461952560237913,
        effect: 1,
        comparisonN: 104,
        treatment1: 2,
        treatment2: 3,
        study: 'C'
      }
    );
    /* verify with R code:
      se <- p$seTE[1]
      effect <- -p$TE[1] # we invert since netmeta uses 3 as the reference
      error <- qnorm(0.975)*se
      c(effect - error, effect + error) # should match lower,upper
     */
    const effects2A = effects2.filter(({ study }) => study === 'A');
    assert.deepStrictEqual(effects2A.length, 1);
    assert.deepStrictEqual(effects2A[0],   {
      p: 0.018185488496468505,
      lower: 0.340293256039629,
      upper: 3.659706743960371,
      effect: 2,
      comparisonN: 108,
      treatment1: 2,
      treatment2: 1,
      study: 'A',
      });
  });

  /* check with R code
    netrank(net)
    netrank(net, 'bad')
   */
  it('should produce valid SUCRA scores', function() {
    const nmaFE = meanDifferenceNMA(studies, trts, means, sds, ns, false);
    const smallerBetterRanks = nmaFE.computePScores(true)
      .map((x) => {
        x.pScore = Math.round(x.pScore * 10000) / 10000;
        return x;
      });
    assert.deepStrictEqual(smallerBetterRanks, [
      {
        treatment: 1,
        pScore: 0.9995,
      },
      {
        treatment: 3,
        pScore: 0.3336,
      },
      {
        treatment: 2,
        pScore: 0.1669,
      },
    ]);

    const biggerBetterRanks = nmaFE.computePScores(false)
      .map((x) => {
        x.pScore = Math.round(x.pScore * 10000) / 10000;
        return x;
      });
    assert.deepStrictEqual(biggerBetterRanks, [
      {
        treatment: 2,
        pScore: 0.8331,
      },
      {
        treatment: 3,
        pScore: 0.6664,
      },
      {
        treatment: 1,
        pScore: 0.0005,
      },
    ]);
  });

  /** test by checking I2, lower.I2, upper.I2 properties of the model */
  it('should produce valid heterogeneity statistics', function() {
    const nmaFE = meanDifferenceNMA(studies, trts, means, sds, ns, false);
    assert.deepStrictEqual(nmaFE.computeISquared(), {
      i2: 0.19595891881666125,
      lower: 0,
      upper: 0.9163635910636153,
    })
  });

  it('should gracefully skip egger test (too few observations)', function () {
    const nmaFE = meanDifferenceNMA(studies, trts, means, sds, ns, false);
    const res = nmaFE.computeComparisonAdjustedEffects(1);
    assert.deepStrictEqual(res.asymmetryP, undefined);
    assert.deepStrictEqual(res.asymmetryTest, undefined);
  })
});

/*
  data <- data.frame(
    studlab = c(10, 10),
    treat = c(1, 2),
    positive = c(10, 12),
    n = c(13, 20)
  )

  p <- with(data, pairwise(treat, event=positive, studlab = studlab, n=n))
  net <- netmeta(p$TE, p$seTE, p$treat1, p$treat2, p$studlab, sm='MD', comb.fixed = TRUE, comb.random = FALSE)
  summary(net)
  netrank(net)
 */
describe('single study NMA', function() {
  const studies = [10, 10];
  const treatments = [1, 2];
  const positive = [10, 12];
  const total = [13, 20];

  it('should produce valid effects', function() {
    const nma = oddsRatioNMA(studies, treatments, positive, total, false);
    assert.deepStrictEqual(nma.getEffect(1, 2), (10.5 / 3.5) / (12.5 / 8.5));
    assert.deepStrictEqual(nma.computeInferentialStatistics(1, 2, .95), {
        p: 0.3486138896968354,
        lower: 0.45936463957514945,
        upper: 9.05946962711131,
      }
    );
    assert.deepStrictEqual(nma.computeStudyLevelEffects(1,.95), [
        {
          p: 0.34861388969683516,
          lower: 0.45936463957514956,
          upper: 9.059469627111307,
          effect: 2.04,
          treatment1: 1,
          treatment2: 2,
          study: 10,
          comparisonN: 33
        },
      ]
    );
    assert.deepStrictEqual(nma.computePScores(true), [
        { treatment: 2, pScore: 0.8256930551515823 },
        { treatment: 1, pScore: 0.1743069448484177 }
      ]);
  });
});

describe('NMA Errors for degenerate inputs', function () {
  const studies = [10, 11, 12];
  const treatments = [1, 2, 1];
  const positive = [10, 12, 15];
  const total = [13, 20, 35];

  // shukra is already generous in taking (and ignoring) single arm studies- but if 0 contrasts are available,
  // absolutely nothing can be done
  it('should throw if 0 contrasts are present', function () {
    assert.throws(() => oddsRatioNMA(studies, treatments, positive, total, false));
  });

  it('should not allow an empty NMA', function() {
    assert.throws(() => meanDifferenceNMA([], [], [], [], []),
      {
        message: 'Must have 1 or more studies to perform an NMA',
      });
  });

  it('should not allow unequal parameter lengths', function() {
    assert.throws(() => meanDifferenceNMA([1], [1], [1, 2], [1], [1]),
      {
        message: 'Studies (n=1), treatments (n=1), means (n=2), and standard deviations (n=1) do not have the same length, as required.',
      }
    );
  })
});

describe('NMA for networks with disconnected components', function() {
  const studies = ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'];
  const trts = [1, 2, 1, 2, 3, 4, 3, 4];
  // the above defines a disconnected graph where 1 is only compared to 2, and 3 to 4
  const means = [8, 10, 7, 10.5, 10.5, 10, 11, 10];
  const sds = [4.923423, 3.867062, 3.250787, 6.349051, 6.664182, 4.324474, 4.301156, 5];
  const ns = [63, 45, 35, 44, 53, 75, 29, 100];

  it('should produce NaN effect estimates at isolated component contrasts', function() {
    const nma = meanDifferenceNMA(studies, trts, means, sds, ns, false);

    assert.deepStrictEqual(nma.getEffect(1, 3), NaN);
    assert.deepStrictEqual(nma.computeInferentialStatistics(1, 3, .95), {
      lower: Number.NaN,
      upper: Number.NaN,
      p: Number.NaN,
    });
  });

  /*
    # reproduce in R with:
    data <- data.frame(
      mean = c(8, 10, 7, 10.5),
      sd = c(4.923423, 3.867062, 3.250787, 6.349051),
      studlab = c('A', 'A', 'B', 'B'),
      treat = c(1, 2, 1, 2),
      n = c(63, 45, 35, 44)
    )

    p <- with(data, pairwise(treat, mean=mean, sd=sd, studlab = studlab, n=n))
    net <- netmeta(p$TE, p$seTE, p$treat1, p$treat2, p$studlab, sm='MD', comb.fixed = TRUE, comb.random = FALSE)
    summary(net)
   */
  it('should give valid effects for treatments in the same component', function() {
    const nma = meanDifferenceNMA(studies, trts, means, sds, ns, false);

    assert.deepStrictEqual(nma.getEffect(1, 2), -2.5558296023438998);
    assert.deepStrictEqual(nma.computeInferentialStatistics(1, 2, .95), {
      lower: -3.872602621303358,
      p: 0.000142234438276434,
      upper: -1.2390565833844414,
    });
    // this is dubious math... since we have no evidence to do comparisons between e.g. 1 and 4, yet we rank them
    assert.deepStrictEqual(nma.computePScores(true), [
        { treatment: 1, pScore: 0.9999288827808618 },
        { treatment: 4, pScore: 0.866255259950658 },
        { treatment: 3, pScore: 0.133744740049342 },
        { treatment: 2, pScore: 0.000071117219138217 }
      ]);

    assert.deepStrictEqual(nma.computeStudyLevelEffects(1), [
        {
          p: 0.018185488496468505,
          lower: -3.659706743960371,
          upper: -0.340293256039629,
          effect: -2,
          treatment1: 1,
          treatment2: 2,
          study: 'A',
          comparisonN: 108
        },
        {
          p: 0.0015178471200330002,
          lower: -5.663145399828594,
          upper: -1.3368546001714057,
          effect: -3.5,
          treatment1: 1,
          treatment2: 2,
          study: 'B',
          comparisonN: 79
        }
      ]
    );
  });
});

describe('NMA on a near-singular effect SE adjustment', () => {
  // this data produces a nearly-singular (condition number = 2M) matrix L in _computeNewSEs
  // which we then invert. this test makes sure we are handling that situation reasonably, by zeroing out effectively
  // 0 values that numerically are not so. this is done with the threshold parameter to pseudoinverse
  // this matches what R/netmeta produces, which seems more consistent with effect sizes
  /* validation R code:
        data <- data.frame(
       mean = c(3, -20, 20, -4.8, -50.6, -20.7, -57.2, 0.9, -12.1, 2.7, -72.4, -5.1, -58.9, 2.9, -56.3, 9, -14.9, 1.2, -45.9, 8.3, -61, 0.8, -59.4, -18.4, 7.1, -57.4, -0.1, -62, -1, -55.8, 4.4, -54.35, 3.17, -56, -20.3, -48.2, -2.3, -57.1, 6.3, -65.7, 2.6, -59.5, 0.9),
       sd = c(22, 29, 20.9, 21.5, 30.3, 29.4, 22.7, 21.6, 22, 27.9, 17.2, 17.3, 24.3, 24.2, 24.2, 21, 22, 27.9, 47.9, 47.8, 27.4, 27.9, 28.7, 26, 27.4, 28.4, 28.7, 24.1, 28, 27.8, 28.1, 23.5, 22.8, 30.1, 28.8, 27.8, 28.1, 29.6, 29.3, 31.2, 27, 26.4, 26.4),
       studlab = c(20646181, 20646181, 20646181, 20646181, 20646185, 20646185, 20646200, 20646200, 20646201, 20646201, 20646214, 20646214, 20646223, 20646223, 20646224, 20646224, 20646227, 20646227, 20646229, 20646229, 20646231, 20646231, 20646232, 20646232, 20646232, 20646234, 20646234, 20646235, 20646235, 20646237, 20646237, 20646242, 20646242, 20646243, 20646243, 20646244, 20646244, 20646245, 20646245, 20646246, 20646246, 20656849, 20656849),
       treat = c(253203, 253204, 253205, 253202, 253197, 253202, 253197, 253205, 253203, 253205, 253197, 253205, 253201, 253205, 253201, 253205, 253203, 253205, 253201, 253205, 253197, 253205, 253197, 253202, 253205, 253197, 253205, 253197, 253205, 253197, 253205, 253197, 253205, 253197, 253202, 253197, 253205, 253197, 253205, 253197, 253205, 253201, 253205),
       n = c(54, 50, 26, 49, 467, 240, 158, 158, 522, 257, 29, 31, 168, 165, 96, 56, 1397, 707, 810, 807, 1530, 780, 1117, 221, 558, 384, 156, 964, 954, 602, 750, 599, 302, 403, 208, 205, 106, 97, 102, 159, 82, 781, 780)
       )

     p <- with(data, pairwise(treat, mean=mean, sd=sd, studlab = studlab, n=n))
     net <- netmeta(p$TE, p$seTE, p$treat1, p$treat2, p$studlab, sm='MD', comb.fixed = TRUE, comb.random = TRUE)
     summary(net)
   */
  const studies = [20646181, 20646181, 20646181, 20646181, 20646185, 20646185, 20646200, 20646200, 20646201, 20646201, 20646214, 20646214, 20646223, 20646223, 20646224, 20646224, 20646227, 20646227, 20646229, 20646229, 20646231, 20646231, 20646232, 20646232, 20646232, 20646234, 20646234, 20646235, 20646235, 20646237, 20646237, 20646242, 20646242, 20646243, 20646243, 20646244, 20646244, 20646245, 20646245, 20646246, 20646246, 20656849, 20656849];
  const interventions = [253203, 253204, 253205, 253202, 253197, 253202, 253197, 253205, 253203, 253205, 253197, 253205, 253201, 253205, 253201, 253205, 253203, 253205, 253201, 253205, 253197, 253205, 253197, 253202, 253205, 253197, 253205, 253197, 253205, 253197, 253205, 253197, 253205, 253197, 253202, 253197, 253205, 253197, 253205, 253197, 253205, 253201, 253205];
  const means = [3, -20, 20, -4.8, -50.6, -20.7, -57.2, 0.9, -12.1, 2.7, -72.4, -5.1, -58.9, 2.9, -56.3, 9, -14.9, 1.2, -45.9, 8.3, -61, 0.8, -59.4, -18.4, 7.1, -57.4, -0.1, -62, -1, -55.8, 4.4, -54.35, 3.17, -56, -20.3, -48.2, -2.3, -57.1, 6.3, -65.7, 2.6, -59.5, 0.9];
  const sds = [22, 29, 20.9, 21.5, 30.3, 29.4, 22.7, 21.6, 22, 27.9, 17.2, 17.3, 24.3, 24.2, 24.2, 21, 22, 27.9, 47.9, 47.8, 27.4, 27.9, 28.7, 26, 27.4, 28.4, 28.7, 24.1, 28, 27.8, 28.1, 23.5, 22.8, 30.1, 28.8, 27.8, 28.1, 29.6, 29.3, 31.2, 27, 26.4, 26.4];
  const ns = [54, 50, 26, 49, 467, 240, 158, 158, 522, 257, 29, 31, 168, 165, 96, 56, 1397, 707, 810, 807, 1530, 780, 1117, 221, 558, 384, 156, 964, 954, 602, 750, 599, 302, 403, 208, 205, 106, 97, 102, 159, 82, 781, 780];

  it('should produce valid effect estimates for fixed effects', () => {
    const nma = meanDifferenceNMA(studies, interventions, means, sds, ns, false);

    assert.deepStrictEqual(nma.getEffect(253203, 253197), 45.06322178943843)
    assert.deepStrictEqual(nma.computeInferentialStatistics(253203, 253197, 0.95), {
      p: 0,
      lower: 42.84752174111436,
      upper: 47.278921837762496,
    });
  });

  it('should produce valid effect estimates for random effects', () => {
    const nma = meanDifferenceNMA(studies, interventions, means, sds, ns, true);

    assert.deepStrictEqual(nma.getEffect(253203, 253197), 44.37721658860796)
    assert.deepStrictEqual(nma.computeInferentialStatistics(253203, 253197, 0.95), {
      p: 0,
      lower: 39.228611794776306,
      upper: 49.52582138243961,
    });
  });
});