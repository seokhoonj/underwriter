#' Create month matrix from claim date vector
#'
#' you can create claim month matrix from virtual underwritng date
#' @param date is a claim date vector
#' @param origin is an orgin date of virtual underwriting
#' @param m is a length of riders
#' @keywords claim year
chk_clm_mon <- function(cdate, origin, m) {
  mon <- (year(cdate) - year(as.Date(origin))) * 12 + month(cdate) - month(as.Date(origin)) + 1L
  matrix(rep(mon, m), ncol = m)
}
