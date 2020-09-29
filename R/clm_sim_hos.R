#' Create hospitalization days simulation matrix
#'
#' you can create the hospitalization days simulation matrix from hospitalization simulation matrix
#' @param data is a hospitalization simulation matrix (through the clm_sim function)
#' @param target is a hospitalization days vector
#' @keywords hospitalization days simulation
clm_sim_hos <- function(data, target) {
  m <- ncol(data)
  z <- matrix(rep(target, m), ncol = m)
  data * z
}
