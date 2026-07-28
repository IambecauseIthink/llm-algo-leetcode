const flashAttention = require("./lesson_overrides_20_flashattention");
const lessons12To15 = require("./lesson_overrides_12_15");
const lessons16To19 = require("./lesson_overrides_16_19");
const lessons21To24 = require("./lesson_overrides_21_24");
const lessons25To28 = require("./lesson_overrides_25_28");
const lessons29To32 = require("./lesson_overrides_29_32");
const refreshed06To11 = require("./assets/lesson_overrides_refresh_06_11");
const refreshed12To18 = require("./assets/lesson_overrides_refresh_12_18");
const refreshed19To24 = require("./assets/lesson_overrides_refresh_19_24");
const refreshed25To32 = require("./assets/lesson_overrides_refresh_25_32");
const lessons33To37 = require("./assets/lesson_overrides_33_37");
const lessons38To42 = require("./assets/lesson_overrides_38_42");

module.exports = {
  ...lessons12To15,
  ...lessons16To19,
  "20": flashAttention,
  ...lessons21To24,
  ...lessons25To28,
  ...lessons29To32,
  ...refreshed06To11,
  ...refreshed12To18,
  ...refreshed19To24,
  ...refreshed25To32,
  ...lessons33To37,
  ...lessons38To42
};
