#' Create year matrix from claim date vector
#'
#' you can create claim year matrix from virtual underwritng date
#' @param date is a claim date vector
#' @param origin is an orgin date of virtual underwriting
#' @param m is a length of riders
#' @keywords claim year
chk_clm_yrs <- function(cdate, origin, m) {
  yrs <- as.double(cdate - as.Date(origin)) %/% 365 + 1
  matrix(rep(yrs, m), ncol = m)
}
