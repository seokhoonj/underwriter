#' Create year matrix from claim date vector
#'
#' you can create claim year matrix from virtual underwritng date
#' @param date is a claim date vector
#' @param origin is an orgin date of virtual underwriting
#' @param m is a length of riders
#' @keywords claim year
check_claim_year <- function(date, origin, m) {
  yr <- as.integer(date - as.Date(origin)) %/% 365 + 1
  matrix(rep(yr, m), ncol = m)
}
