# shukra
Tools for meta-analytical statistics, including Network Meta-Analysis (NMA), in NodeJS

## Example
A fixed effects NMA on difference in means:
```javascript
const { meanDifferenceNMA } = require('shukra/nma');

const studies = [1, 1, 2, 2, 2, 3, 3];
const trts = ['A', 'B', 'A', 'B', 'C', 'C', 'B'];
const means = [8, 10, 7, 10.5, 10.5, 10, 11];
const sds = [4.92, 3.87, 3.25, 6.35, 6.66, 4.32, 4.30];
const ns = [63, 45, 35, 44, 53, 75, 29];

const nma = meanDifferenceNMA(studies, trts, means, sds, ns, false);

nma.getEffect('A', 'B');
/*
-2.819
 */

nma.computeInferentialStatistics('A', 'B');
/*
{
  p: 0.000008,
  lowerBound: -4.06,
  upperBound: -1.58
}
*/   
```
Computing a pooled mean under random effects using inverse variance weighting:

```javascript
const { pooledMean } = require('shukra/pooling');
pooledMean(ns, means, sds);
/*
{
  estimate: 9.493324459831722,
  lower: 8.334404617356348,
  upper: 10.652244302307096,
  studyEstimates: [
    { lower: 6.785093322564656, estimate: 8, upper: 9.214906677435344 },
    { lower: 8.86928592265621, estimate: 10, upper: 11.13071407734379 },
    { lower: 5.923293264578356, estimate: 7, upper: 8.076706735421643 },
    ...
  ]
}
*/
```

## Install
Install the current release on [npm](https://www.npmjs.com/package/shukra):
```bash
npm install shukra
```

Install the most recent development version:
```bash
npm install --save https://github.com/holub008/shukra/tarball/master
```

## About Shukra
`shukra` is a web targeted meta-analytical statistics toolkit. It was specifically designed for simplicity & speed over 
 comprehensiveness (as in formal research publication); this means `shukra` can fit an NMA in the scope of a web request (<10 milliseconds). 

`shukra` is developed for and by [Nested Knowledge](https://nested-knowledge.com/), but we welcome all users and contributions. 
It is predominantly transcribed from the `meta` and `netmeta` packages, and carries their licenses. These packages are also
used for verifying correctness (see `test/` for code samples).

The feature set is growing, but currently offers several modules:

* NMA (`shukra/nma`)
    * Random & Fixed effects models
    * Mean Difference (continuous outcomes) & Odds Ratios (binomial outcomes)
    * Inferential statistics on effect size estimates and individual study point estimates
* Pooling (`shukra/pooling`)
    * Inverse variance weighting
    * Random & Fixed effects models
    * Imputation of missing data
    * Inferential statistics on pooled estimates and individual study point estimates

<p align="center"> 
  <img src="/docs/images/shukra.jpeg">
</p>

## About NMA

Network Meta-Analysis is a model for performing research synthesis. Given multi-arm (2+) studies, where each arm represents a discrete experimental treatment, and a measured outcome for each arm, an NMA produces an effect size estimate and associated inferential statistics.

NMA is exciting in that it models direct contrasts (i.e. head to head treatment comparisons in studies) *and* indirect evidence (i.e. using knowledge about A vs. B in one study and B vs. C in another to make inference on A vs. C).

<p align="center">
  <img src="/docs/images/ischemic_stroke_recanalization_network.jpeg">
</p>

