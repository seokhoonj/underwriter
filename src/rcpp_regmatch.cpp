#include <Rcpp.h>
#include <vector>
#include <regex>
using namespace Rcpp;

// [[Rcpp::export]]
StringVector regmatch(std::string m, std::vector<std::string> x) {
  std::regex re(m);
  std::smatch match;
  std::vector<std::string> res;
  res.resize(res.size());
  for (const auto &s : x) {
    if (std::regex_match(s, match, re)) {
      for (size_t i = 0; i < match.size(); i++) {
        res[i] = match[i].str();
      }
    }
  }
  return(wrap(res));
}
