#' Combine claim matrix with 1st claim matrix by one time payment vector
#'
#' you can combine claim matrix with 1st claim matrix
#' @param clm is a claim matrix
#' @param clm_1st is a 1st claim matrix
#' @param otime is a one time benefit vector (one-time benefits or continuous benefits)
#' @keywords combine claim and 1st claim
cmb_clm_1st <- function(clm, clm_1st, otime) {
  loc <- which(otime == 1L)
  clm[, loc] <- clm_1st[, loc]
  return(clm)
}
