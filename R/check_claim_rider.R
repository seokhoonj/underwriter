check_claim_rider <- function(rider, code, target) {
  data <- check_claim(code, target)
  m <- ncol(data)
  n <- nrow(data)
  z <- vapply(1:m,
              function(x) as.character(ifelse(data[, x] == 1L, rider[x], NA)),
              FUN.VALUE = character(n))
  apply(z, 1, underwriter::glue_code)
}
