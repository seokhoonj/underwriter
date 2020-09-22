#' Create claim amount ratio matrix by claim year
#'
#' you can create the claim amount ratio matrix from the claim year matrix.
#' @param data claim year matrix
#' @param ratio claim amount ratio vector
#' @keywords claim amount ratio
check_claim_ratio <- function(data, ratio) {
  # check arguments
  if (!is.matrix(data))
    data <- as.matrix(data)
  # 1st year rows
  loc <- which(rowSums(data) == length(ratio))
  nloc <- length(loc)
  # create matrix for 1st year claim amt ratio
  data_rat <- matrix(rep(ratio, nloc), ncol = length(ratio), byrow = TRUE)
  data[loc,] <- data_rat
  data[data > 1] <- 1.
  return(data)
}
