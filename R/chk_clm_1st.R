#' Create 1st claim boolean matrix by insured id from claim boolean matrix
#'
#' you can create the 1st claim matrix considering insured id.
#' @param data is a boolean matrix (claim kcd code exists or not)
#' @param id is a insured id numeric vector
#' @keywords 1st claim
chk_clm_1st <- function(data, id) {
  if (is.character(id))
    id <- as.double(id)
  if (!class(data) %in% c("matrix", "dgCMatrix"))
    data <- as.matrix(data)
  # m <- ncol(data)
  n <- nrow(data)
  as(vapply(1:m, function(x) rcpp_first_claim(data[, x], id),
            FUN.VALUE = double(n)), "CsparseMatrix")
}
