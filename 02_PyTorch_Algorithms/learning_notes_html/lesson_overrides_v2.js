const flashAttention = require("./lesson_overrides_20_flashattention");
const lessons12To15 = require("./lesson_overrides_12_15");
const lessons16To19 = require("./lesson_overrides_16_19");
const lessons21To24 = require("./lesson_overrides_21_24");
const lessons25To28 = require("./lesson_overrides_25_28");
const lessons29To32 = require("./lesson_overrides_29_32");

module.exports = {
  ...lessons12To15,
  ...lessons16To19,
  "20": flashAttention,
  ...lessons21To24,
  ...lessons25To28,
  ...lessons29To32
};
