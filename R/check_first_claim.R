#' Create first claim boolean matrix by insured id from claim boolean matrix
#'
#' you can create the first claim matrix considering insured id.
#' @param id is a insured id numeric vector
#' @param data is a boolean matrix (claim kcd code exists or not)
#' @keywords first claim
check_first_claim <- function(id, data) {
  if (!is.numeric(id))
    id <- as.numeric(id)
  if (!is.matrix(data))
    data <- as.matrix(data)
  m <- ncol(data)
  n <- nrow(data)
  matrix(vapply(1:m, function(x) underwriter::rcpp_first_claim(id, data[, x]),
                FUN.VALUE = numeric(n)), ncol = m)
}
