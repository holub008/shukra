# shukra
Network Meta-Analysis (NMA) in NodeJS

## Example

```javascript
    const {fixedEffectsMeanDifferenceNMA} = require('shukra');
    const studies = [1, 1, 2, 2, 2, 3, 3];
    const trts = ['A', 'B', 'A', 'B', 'C', 'C', 'B'];
    const means = [8, 10, 7, 10.5, 10.5, 10, 11];
    const sds = [4.92, 3.87, 3.25, 6.35, 6.66, 4.32, 4.30];
    const ns = [63, 45, 35, 44, 53, 75, 29];

    const nma = fixedEffectsMeanDifferenceNMA(studies, trts, means, sds, ns);
    
    nma.getEffect('A', 'B');
    /**
     *  -2.819
     */

    nma.computeInferentialStatistics('A', 'B');
    /**
     * {
     *   p: 0.000008,
     *   lowerBound: -4.06,
     *    upperBound: -1.58
     * }
    */   
```

## Install
Install the most recent development version:
```bash
npm install --save https://github.com/holub008/shukra/tarball/master
```

## About NMA

Network Meta-Analysis is a model for performing research synthesis. Given multi-arm (2+) studies, where each arm represents a discrete experimental treatment, and a measured outcome for each arm, an NMA produces a synthesized effect size and inferential statistics.

NMA is exciting in that it models direct contrasts (i.e. head to head treatment comparisons in studies) *and* indirect evidence (i.e. using knowledge about A vs. B in one study and B vs. C in another to make inference on A vs. C).

![network graph](/docs/images/ischemic_stroke_recanalization_network.jpeg?raw=true "Network Graph")

## About Shukra
Shukra is a web targetted NMA implementation. It was specifically designed with simplicity & speed in mind over flexibility (as in exploratory analyses); this means shukra can *fit* an NMA in the scope of a web request (<10 milliseconds) and convey highly interprettable results to end-users. 

The growing feature set currently offers:
  * Fixed effects models
  * Mean Difference (continuous outcomes) & Odds Ratios (binomial outcomes)
  * Inferential statistics on effect size estimates

![shukra](docs/images/shukra.jpeg?raw=true "Shukra")
