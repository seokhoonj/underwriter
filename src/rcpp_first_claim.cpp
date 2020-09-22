
// UnderWriting by Seokhoon Joo 2020-09-22
#include <Rcpp.h>
#include <iostream>
#include <vector>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector rcpp_first_claim(std::vector<double> id, std::vector<double> claim) {
  // locations vector
  std::vector<double> rows(1, 0);
  std::vector<double>::iterator ip;
  double row_p = 0;
  for (ip = id.begin(); ip != id.end(); ++ip) {
    if (*ip != *(ip+1)) {
      row_p += 1;
      rows.push_back(row_p);
    } else {
      row_p += 1;
    }
  }
  // result vector
  std::vector<double> result;
  result.resize(claim.size());
  std::vector<double>::iterator iq;
  for (iq = rows.begin(); iq != rows.end(); ++iq) {
    double n1 = 0;
    for (double i = *iq; i < *(iq+1); ++i) {
      if (n1 == 0 && claim[i] == 1) {
        result[i] = 1;
        n1 = 1;
      } else {
        result[i] = 0;
      }
    }
  }
  return wrap(result);
}
