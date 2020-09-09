
// UnderWriting by Seokhoon Joo 2020-09-09
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector rcpp_uniq_day(StringMatrix id, NumericVector from, NumericVector to) {

  // split points
  std::vector<int> rows(1, 0);
  //std::cout << "rows max_size: " << rows.max_size() << std::endl;
  for (int i = 0; i < id.ncol(); ++i) {
    StringVector vec = id( _ , i);
    StringVector::iterator ip;
    int row = 0;
    for (ip = vec.begin(); ip != vec.end()-1; ++ip) {
      if (strcmp(*ip, *(ip + 1)) != 0) {
        //std::cout << *ip << " " << *(ip + 1) << std::endl;
        row += 1;
        rows.push_back(row);
      } else {
        row += 1;
      }
    }
  }
  sort(rows.begin(), rows.end());
  rows.resize(std::unique(rows.begin(), rows.end()) - rows.begin());
  rows.push_back(id.nrow());

  // calculate the length between from to end
  int nrows = rows.size();
  std::vector<int> lens(0);
  for (int k = 0; k < nrows-1; k++) {
    // length of rows
    int n = rows[k+1]-rows[k];
    std::vector<int> days(0);
    for (int j = rows[k]; j < rows[k+1]; ++j) {
      int m = to[j] - from[j] + 1;
      std::vector<int> out(m);
      out[0] = from[j];
      days.push_back(out[0]);
      for (int i = 1; i < m; i++) {
        out[i] = out[i-1] + 1;
        days.push_back(out[i]);
      }
    }
    sort(days.begin(), days.end());
    days.resize(unique(days.begin(), days.end()) - days.begin());
    lens.push_back(days.size());
  }
  return wrap(lens);
}
