const { getConnectedComponents } = require('../src/graph');
const assert = require('assert');

describe('getConnectedComponents', function() {
  it('should return an empty array for an empty network', () => {
    assert.deepStrictEqual(getConnectedComponents([], []), [])
  });

  it('should compute a single component on a single node', () => {
    assert.deepStrictEqual(getConnectedComponents([101], ['A']), [['A']])
  });

  it('should compute a single component on a multi-node graph with one component', () => {
    // A-B
    assert.deepStrictEqual(getConnectedComponents([101, 101], ['A', 'B']), [['A', 'B']]);
    // A-B-C
    assert.deepStrictEqual(getConnectedComponents([101, 101, 102, 102], ['A', 'B', 'C', 'B']), [['A', 'B', 'C']]);
  });

  it('should compute a single component on a multi-node graph with repeated edges', () => {
    // A-B
    assert.deepStrictEqual(getConnectedComponents([101, 101, 102, 102], ['A', 'B', 'A', 'B']), [['A', 'B']]);
  });

  it('should compute a two components on a multi-node graph with repeated edges', () => {
    // A-B C-D
    assert.deepStrictEqual(getConnectedComponents([101, 101, 102, 102, 103, 103],
      ['A', 'B', 'A', 'B', 'C', 'D']),
      [['A', 'B'], ['C', 'D']]);
    /*
       A-B   D-E-F
       |
       C
     */
    assert.deepStrictEqual(getConnectedComponents([101, 101, 102, 102, 103, 103, 104, 104],
      ['A', 'B', 'A', 'C', 'D', 'E', 'F', 'E']),
      [['A', 'B', 'C'], ['D', 'E', 'F']]);
  });

  it('should compute components from multi-arm studies', () => {
    /*
       A-B   D-E-F G
       |
       C
     */
    assert.deepStrictEqual(getConnectedComponents([101, 101, 101, 102, 102, 103, 103, 104],
      ['A', 'B', 'C', 'D', 'E', 'F', 'E', 'G']),
      [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]);
  });

  it('should compute components from prickly networks', () => {
    /*
       A-B-D-E-G   H-I    K-z-m
       |     |
       C     F
     */
    assert.deepStrictEqual(getConnectedComponents([
        101, 101, 101, 101, 102, 102, 103, 103, 103, 103,
        104, 104, 105, 105,
        106, 106, 107, 107
      ],
      [
        'A', 'B', 'C', 'D', 'E', 'D', 'D', 'E', 'F', 'G',
        'H', 'I', 'I', 'H',
        'K', 'z', 'z', 'm',
      ]),
      [['A', 'B', 'C', 'D', 'E', 'F', 'G'], ['H', 'I'], ['K', 'z', 'm']]);
  });
});