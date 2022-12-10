const assert = require("assert");
const { cohensKappa } = require("../src/irr");

describe('cohen\'s kappa', function() {
  it('should complain for no items', () => {
    assert.throws(() => cohensKappa([], 2), new Error('Must supply at least 1 item'));
  });

  it('should complain for invalid item structure', () => {
    assert.throws(() => cohensKappa([{ raters: ['a'], ratings: ['include', 'exclude']}], 2), new Error('Each item must have length 2 ratings and raters arrays'));
  });

  it('works for 1 item', () => {
    assert.deepStrictEqual(
      cohensKappa([{
        raters: ['a', 'b'],
        ratings: ['include', 'include'],
      }], 2),
      {
        kappa: NaN,
        lower: NaN,
        upper: NaN,
      }
    );
  });

  /* check with r:
    library(irr)
    kappa2(data.frame(
      raterA=c('i', 'e', 'e', 'e', 'i', 'i'),
      raterB=c('i', 'i', 'e', 'i', 'i', 'i')
     ))
   */
  it('works for multiple items', () => {
    assert.deepStrictEqual(
      cohensKappa([
        {
          raters: ['a', 'b'],
          ratings: ['include', 'include'],
        },
        {
          raters: ['b', 'a'],
          ratings: ['include', 'exclude'],
        },
        {
          raters: ['a', 'b'],
          ratings: ['exclude', 'exclude'],
        },
        {
          raters: ['a', 'b'],
          ratings: ['exclude', 'include'],
        },
        {
          raters: ['a', 'b'],
          ratings: ['include', 'include'],
        },
        {
          raters: ['b', 'a'],
          ratings: ['include', 'include'],
        },
      ], 2),
      {
        kappa: 0.33333333333333326,
        lower: -0.4210571416658443,
        upper: 1,
      }
    );
  });

  it('works for multiple items', () => {
    const items = [];
    for (let i = 0; i < 20; i++) {
      items.push({
        raters: ['a', 'b'],
        ratings: ['include', 'include'],
      });
    }
    for (let i = 0; i < 15; i++) {
      items.push({
        raters: ['a', 'b'],
        ratings: ['exclude', 'exclude'],
      });
    }
    for (let i = 0; i < 10; i++) {
      items.push({
        raters: ['a', 'b'],
        ratings: ['exclude', 'include'],
      });
    }
    for (let i = 0; i < 5; i++) {
      items.push({
        raters: ['a', 'b'],
        ratings: ['include', 'exclude'],
      });
    }

    // from wiki: https://en.wikipedia.org/wiki/Cohen%27s_kappa#Examples
    assert.deepStrictEqual(
      cohensKappa(items, 2),
      {
        kappa: 0.4,
        lower: 0.14595963760366826,
        upper: 0.6540403623963318,
      }
    );
  });


  /* check with r:
    library(irr)
    kappa2(data.frame(
      raterA=c('i', 'e', 'e', 'e', 'i', 'u'),
      raterB=c('u', 'i', 'e', 'i', 'i', 'i')
     ))
  */
  it('works for multiple items', () => {
    assert.deepStrictEqual(
      cohensKappa([
        {
          raters: ['a', 'b'],
          ratings: ['include', 'unknown'],
        },
        {
          raters: ['b', 'a'],
          ratings: ['include', 'exclude'],
        },
        {
          raters: ['a', 'b'],
          ratings: ['exclude', 'exclude'],
        },
        {
          raters: ['a', 'b'],
          ratings: ['exclude', 'include'],
        },
        {
          raters: ['a', 'b'],
          ratings: ['include', 'include'],
        },
        {
          raters: ['b', 'a'],
          ratings: ['include', 'unknown'],
        },
      ], 3),
      {
        kappa: 0,
        lower: -0.5657928562493831,
        upper: 0.5657928562493831,
      }
    );
  });
});
