#' Create surgery grade simulation matrix
#'
#' you can create the surgery grade simulation matrix from surgery simulation matrix
#' @param info is a claim book with surgery grade rider information.
#' @param data is a surgery simulation matrix (through the clm_sim function)
#' @param target is a surgery grade vector
#' @keywords surgery grade simulation
clm_sim_sur_grd <- function(info, data, target) {
  col <- target
  row <- seq_along(target)
  rng <- unname(table(info$rdr_cd))
  loc <- c(0, cumsum(rng))

  z = list()
  for (j in seq_along(rng)) {
    dat <- data[, (loc[j]+1):(loc[j+1])]
    vec <- vector(length = length(target))
    for (i in row) vec[i] <- dat[row[i], col[i]]
    z[[j]] <- vec
  }
  matrix(unlist(z), ncol = length(rng))
}
