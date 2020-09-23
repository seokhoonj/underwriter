#' Create claim rider matrix
#'
#' you can create the claim rider matrix (from the claim boolean matrix included in the function).
#' @param rider is a rider vector
#' @param code is a kcd code regular expression vector
#' @param target is a claim kcd code vector
#' @keywords claim amount rider
check_claim_rider <- function(rider, code, target) {
  data <- underwriter::check_claim(code, target)
  m <- ncol(data)
  n <- nrow(data)
  z <- vapply(1:m,
              function(x) as.character(ifelse(data[, x] == 1L, rider[x], NA)),
              FUN.VALUE = character(n))
  apply(z, 1, glue_code)
}
