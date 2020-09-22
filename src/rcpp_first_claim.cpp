
// UnderWriting by Seokhoon Joo 2020-09-22
#include <Rcpp.h>
#include <iostream>
#include <vector>
using namespace Rcpp;

// [[Rcpp::export]]
std::vector<int> first_claim(std::vector<int> id, std::vector<int> claim) {
  // locations vector
  std::vector<int> rows(1, 0);
  rows.reserve(id.size());
  std::vector<int>::iterator ip;
  int row_p = 0;
  for (ip = id.begin(); ip != id.end(); ++ip) {
    if (*ip != *(ip+1)) {
      row_p += 1;
      rows.push_back(row_p);
    } else {
      row_p += 1;
    }
  }
  // result vector
  std::vector<int> result;
  result.resize(claim.size());
  std::vector<int>::iterator iq;
  int row_q = 0;
  for (iq = rows.begin(); iq != rows.end(); ++iq) {
    int n1 = 0;
    for (int i = *iq; i < *(iq+1); ++i) {
      if (n1 == 0 & claim[i] == 1) {
        result[i] = 1;
        n1 = 1;
      } else {
        result[i] = 0;
      }
    }
  }
  return result;
}
