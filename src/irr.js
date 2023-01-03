const {STD_NORMAL} = require("./util");

function preconditionStructure(items, categories) {
  if (!items.length) {
    throw new Error(`Must supply at least 1 item`);
  }

  const uniqueCategories = new Set();
  const uniqueReviewers = new Set();
  items.forEach((i) => {
    if (!Array.isArray(i.ratings)) {
      throw new Error(`Each item must include a 'ratings' array`);
    }
    if (!Array.isArray(i.raters)) {
      throw new Error(`Each item must include a 'raters' array`);
    }
    if (i.ratings.length !== i.raters.length || i.ratings.length !== 2) {
      throw new Error(`Each item must have length 2 ratings and raters arrays`);
    }
    i.raters.forEach((r) => uniqueReviewers.add(r));
    i.ratings.forEach((c) => uniqueCategories.add(c));
  });

  if (uniqueReviewers.size !== 2) {
    throw new Error(`Found ${uniqueReviewers.size} raters; expected 2`);
  }

  const reviewerMapping = {};
  [...uniqueReviewers].forEach((r, ix) => {
    reviewerMapping[r] = ix;
  })

  if (uniqueCategories.size > categories) {
    throw new Error(`Specified ${categories} but found more (${uniqueCategories.size}) in the data`);
  }

  const categoryMapping = {};
  [...uniqueCategories].forEach((c, ix) => {
    categoryMapping[c] = ix;
  });

  return {
    categoryMapping,
    reviewerMapping,
  };
}

/**
 * @param items Array of objects with attributes `raters` and `ratings`
 * @param categories Number the number of categories of rating to be considered. must be less than or equal to the number
 * in the items
 * @param width Number 0-1 for width of confidence interval
 * @returns {{lower: number, upper: number, kappa: number}}
 */
function cohensKappa(items, categories, width=0.95) {
  const { categoryMapping, reviewerMapping } = preconditionStructure(items, categories);

  const coincidence = Array.from(Array(categories), _ => Array(categories).fill(0));
  let agreements = 0;
  items.forEach((r) => {
    r.ratings.forEach((c,ix) => {
      coincidence[categoryMapping[c]][reviewerMapping[r.raters[ix]]] += 1;
    });
    if (r.ratings[0] === r.ratings[1]) {
      agreements += 1;
    }
  });

  let reviewerProductSum = 0;
  for (let i = 0; i < categories; i++) {
    reviewerProductSum += coincidence[i][0] * coincidence[i][1];
  }

  const randomAgreement = Math.pow(items.length, -2) * reviewerProductSum;
  const agreement = agreements / items.length;

  const kappa = (agreement - randomAgreement) / (1 - randomAgreement);
  const se = Math.sqrt((agreement * (1 - agreement)) / (items.length * Math.pow(1 - randomAgreement, 2)));

  const q = 1 - (1 - width) / 2;
  const z = STD_NORMAL.ppf(q)

  return ({
    kappa,
    lower: kappa - z * se,
    upper: Math.min(kappa + z * se, 1),
  });
}


module.exports = {
  cohensKappa,
};