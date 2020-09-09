
// UnderWriting by Seokhoon Joo 2020-09-09
#include <vector>
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector rcpp_uniq_day(StringMatrix id, NumericVector from, NumericVector to) {

  // split points
  std::vector<int> rows(1, 0);
  for (int i = 0; i < id.ncol(); ++i) {
    StringVector vec = id( _ , i);
    StringVector::iterator ip;
    int row = 0;
    for (ip = vec.begin(); ip != vec.end()-1; ++ip) {
      if (strcmp(*ip, *(ip + 1)) != 0) {
        //std::cout << *ip << *(ip + 1) << std::endl;
        row += 1;
        rows.push_back(row);
      } else {
        row += 1;
      }
    }
  }
  sort(rows.begin(), rows.end());
  //rows.resize(unique(rows.begin(), rows.end()) - rows.begin());
  rows.push_back(id.nrow());

  // calculate the length between from to end
  int nrows = rows.size();
  std::vector<int> lens;
  for (int k = 0; k < nrows-1; ++k) {
    //std::cout << rows[k] << rows[k+1] << std::endl;
    std::vector<int> s = std::vector<int>(from.begin()+rows[k], from.begin()+rows[k+1]);
    std::vector<int> e = std::vector<int>(to.begin()+rows[k], to.begin()+rows[k+1]);
    // nrow
    int n = s.size();
    std::vector<int> days;
    for (int j = 0; j < n; ++j) {
      int m = e[j] - s[j] + 1; // size of allocation
      std::vector<int> out(m);
      out[0] = s[j];
      days.push_back(out[0]);
      for (int i = 1; i < m; ++i) {
        out[i] = out[i-1] + 1;
        days.push_back(out[i]);
      }
    }
    sort(days.begin(), days.end());
    //days.resize(unique(days.begin(), days.end()) - days.begin());
    lens.push_back(days.size());
  }
  return wrap(lens);
}
