#' Create claim boolean matrix
#'
#' you can create the first claim matrix considering insured id.
#' @param code is a kcd code regular expression vector
#' @param target is a kcd code vector
#' @keywords claim boolean matrix
check_claim <- function(code, target) {
  vapply(code,
         function(x) as.integer(grepl(x, target, perl = TRUE)),
         FUN.VALUE = integer(length(target)), USE.NAMES = FALSE)
}
