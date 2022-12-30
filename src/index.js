const nma = require('./nma.js');
const pooling = require('./pooling');
const irr = require('./irr');

module.exports = {
  ...nma,
  ...pooling,
  ...irr,
};