const assert = require('assert');
const {NetworkMetaAnalysis, oddsRatioNMA, meanDifferenceNMA} = require('../src/nma');
const {Matrix} = require('ml-matrix');

/**
 * tests for NMA module
 */

describe('NMA Holder Class', function () {
  const trt = new Matrix([[0, 5], [-5, 0]]);
  const se = new Matrix([[0, 2], [2, 0]]);
  const trtLabel = ['Band-aid', 'Stitch'];
  const stubStudyLevelEffects = [];
  const meanDiffNMA = new NetworkMetaAnalysis(trt, se, trtLabel, stubStudyLevelEffects);

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
    const nma = oddsRatioNMA(study, treatmentGood, positiveCountGood, totalCountGood);
    assert.strictEqual(nma.getTreatments().length, 4);
  });

  it('should not produce a model for duplicate treatment arms', function () {
    assert.throws(() => oddsRatioNMA(study, treatmentBad, positiveCountGood, totalCountGood));
  });

  it('should not produce a model for unequal input lengths', function () {
    assert.throws(() => oddsRatioNMA(study, treatmentGood, positiveCountGood, totalCountBad));
  });

  it('should not produce a model for unequal input lengths', function () {
    assert.throws(() => oddsRatioNMA(study, treatmentGood, positiveCountBad, totalCountGood));
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
  const study = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
    15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24];

  const nma = oddsRatioNMA(study, treatment, positiveCount, totalCount);

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
    assert.ok(cb.lower > 1.20 && cb.lower < 1.21);
    assert.ok(cb.upper > 2.02 && cb.upper < 2.03);
    assert.ok(cb.p < .01);

    const ab = nma.computeInferentialStatistics("A", "B", .95);
    assert.ok(ab.lower > .635 && ab.lower < .645);
    assert.ok(ab.upper > 1.04 && ab.upper < 1.05);
    assert.ok(ab.p > .05);

    const ab99 = nma.computeInferentialStatistics("A", 'B', .99);
    assert.ok(ab.lower > ab99.lower && ab.upper < ab99.upper);
  });

  it('should produce study level effects', function() {
    const effectsA = nma.computeStudyLevelEffects('A');
    /*
      sum(p2$treat1 == 'A')
     */
    assert.deepStrictEqual(effectsA.length, 20);
    effectsA1C = effectsA.filter(({ study, treatment2 }) => study  === 1 && treatment2 === 'C');
    assert.deepStrictEqual(effectsA1C.length, 1);
    /*
      effect <- exp(p2$TE[1]) # should match effect below (note these effects are anscambe corrected .5
      log_effect <- p2$TE[1]
      se <- p2$seTE[1]
      error <- qnorm(0.975)*se
      exp(c(log_effect - error, log_effect + error)) # should match lower,upper
     */
    assert.deepStrictEqual(effectsA1C[0], {
      p: 0.01190387738868881,
      lower: 0.16335387790814926,
      upper: 0.7987415280874249,
      effect:  0.36121673003802274,
      comparisonN: 280,
      treatment1: 'A',
      treatment2: 'C',
      study: 1
    });
  });
});

describe('Mean Difference FE NMA', function () {
  // this is a faux dataset
  const studies = ['A', 'A', 'B', 'B', 'B', 'C', 'C'];
  const trts = [1, 2, 1, 2, 3, 3, 2];
  const means = [8, 10, 7, 10.5, 10.5, 10, 11];
  const sds = [4.923423, 3.867062, 3.250787, 6.349051, 6.664182, 4.324474, 4.301156];
  const ns = [63, 45, 35, 44, 53, 75, 29];

  const nma = meanDifferenceNMA(studies, trts, means, sds, ns);

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
    assert.ok(oneThree95.lower > -4.095 && oneThree95.lower < -4.09);
    assert.ok(oneThree95.upper > -0.918 && oneThree95.upper < -0.917);
    assert.ok(oneThree95.p < .05);

    const twoThree95 = nma.computeInferentialStatistics(2, 3, .95);
    assert.ok(twoThree95.lower > -1.12 && twoThree95.lower < -1.11);
    assert.ok(twoThree95.upper > 1.735 && twoThree95.upper < 1.745);
    assert.ok(twoThree95.p > .05);
  });

  it('should produce study level effects', function() {
    const effects = nma.computeStudyLevelEffects(1);
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
      p: 0.01818548988800539,
      lower: -3.659706775608142,
      upper: -0.34029322439185816,
      effect: -2,
      comparisonN: 108,
      treatment1: 1,
      treatment2: 2,
      study: 'A'
    },);

    const effects2 = nma.computeStudyLevelEffects(2);
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
        p: 0.2884067201816114,
        lower: -0.8461952912275792,
        upper: 2.846195291227579,
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
      p: 0.01818548988800539,
      lower: 0.34029322439185816,
      upper: 3.659706775608142,
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
    const smallerBetterRanks = nma.computePScores(true)
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

    const biggerBetterRanks = nma.computePScores(false)
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
    const nma = oddsRatioNMA(studies, treatments, positive, total);
    assert.deepStrictEqual(nma.getEffect(1, 2), (10.5 / 3.5) / (12.5 / 8.5));
    assert.deepStrictEqual(nma.computeInferentialStatistics(1, 2, .95), {
        p: 0.3486138799297245,
        lower: 0.4593646265162551,
        upper: 9.059469884655424
      }
    );
    assert.deepStrictEqual(nma.computeStudyLevelEffects(1,.95), [
        {
          p: 0.34861387992972426,
          lower: 0.4593646265162552,
          upper: 9.059469884655424,
          effect: 2.04,
          treatment1: 1,
          treatment2: 2,
          study: 10,
          comparisonN: 33
        },
      ]
    );
    assert.deepStrictEqual(nma.computePScores(true), [
        { treatment: 2, pScore: 0.8256930600351378 },
        { treatment: 1, pScore: 0.17430693996486224 }
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
    assert.throws(() => oddsRatioNMA(studies, treatments, positive, total));
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

  const nma = meanDifferenceNMA(studies, trts, means, sds, ns);

  it('should produce NaN effect estimates at isolated component contrasts', function() {
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
    assert.deepStrictEqual(nma.getEffect(1, 2), -2.5558296023438998);
    assert.deepStrictEqual(nma.computeInferentialStatistics(1, 2, .95), {
      lower:  -3.872602646411969,
      p: 0.00014223442580374446,
      upper:  -1.2390565582758304,
    });
    // this is dubious math... since we have no evidence to do comparisons between e.g. 1 and 4, yet we rank them
    assert.deepStrictEqual(nma.computePScores(true), [
        { treatment: 1, pScore: 0.9999288827870981 },
        { treatment: 4, pScore: 0.866255268796531 },
        { treatment: 3, pScore: 0.13374473120346897 },
        { treatment: 2, pScore: 0.00007111721290187223 }
      ]);

    assert.deepStrictEqual(nma.computeStudyLevelEffects(1), [
        {
          p: 0.01818548988800539,
          lower: -3.659706775608142,
          upper: -0.34029322439185816,
          effect: -2,
          treatment1: 1,
          treatment2: 2,
          study: 'A',
          comparisonN: 108
        },
        {
          p: 0.0015178470364287655,
          lower: -5.663145441076081,
          upper: -1.3368545589239198,
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