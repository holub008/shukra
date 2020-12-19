const { Matrix } = require('ml-matrix');

function dfsAccumulate(currentIx, adjacency, visitedIxs) {
  const neighbors = adjacency.getRow(currentIx)
    .map((value, nix) => [value, nix])
    .filter(([value, nix]) => value)
    .map(([value, nix]) => nix);
  neighbors.forEach((nix) => {
    if (!visitedIxs.has(nix)) {
      visitedIxs.add(nix); // MUTATION!
      dfsAccumulate(nix, adjacency, visitedIxs);
    }
    // else we already discovered, move on
  })
}

function getNextSearchStart(components, nodes) {
  const allNodes = new Set();
  components.forEach((comp) => {
    comp.forEach((node) => allNodes.add(node));
  });

  return nodes.findIndex((n, ix) => !allNodes.has(ix));
}

/**
 * identify connected componenets where treatments are vertices and studies are edges
 * note this method relies on reference equality
 * @param studies {Array}
 * @param treatments {Array}
 * @return an array of array of treatments. the sub-arrays will correspond to components of the graph
 */
function getConnectedComponents(studies, treatments) {
  if (studies.length !== treatments.length) {
    throw new Error(`Studies (${studies.length}) and treatments (${treatments.length}) must share length.`);
  }
  if (!studies.length) {
    return [];
  }

  const nodes = [ ...(new Set(treatments)) ];
  const adjacency = Matrix.zeros(nodes.length, nodes.length);
  const uniqueStudies = [ ...(new Set(studies)) ];
  const indexedStudies = studies.map((s, ix) => [s, ix]);
  uniqueStudies.forEach((s) => {
    const studyMatchingIxs = indexedStudies
      .filter(([study, index]) => study === s)
      .map(([study, index]) => index);
    const studyTreatments = studyMatchingIxs.map((ix) => treatments[ix]);
    const nodeIxs = studyTreatments.map((t) => nodes.indexOf(t));
    nodeIxs.forEach((nix1) => {
      nodeIxs.forEach((nix2) => {
        adjacency.set(nix1, nix2, 1);
      });
    });
  });

  let currentComponent = new Set([]);
  dfsAccumulate(0, adjacency, currentComponent)
  const components = [currentComponent];
  let searchStart = getNextSearchStart(components, nodes);
  while(searchStart >= 0) {
    currentComponent = new Set([]);
    dfsAccumulate(searchStart, adjacency, currentComponent);
    components.push(currentComponent);
    searchStart = getNextSearchStart(components, nodes);
  }

  return components.map((c) => {
    return [...c].map((nix) => nodes[nix]);
  });
}

module.exports = {
  getConnectedComponents,
};