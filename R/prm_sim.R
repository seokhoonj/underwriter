#' Create premium payment simulation data
#'
#' you can create number of premium payment matrix
#' @param rsk_info is a risk information file
#' @param clm_info is a claim information file
#' @param data is a claim history file
#' @param origin is an origin date
#' @param yrs is a payment duration
#' @keywords premium payment simulation
prm_sim <- function(rsk_info, clm_info, data, origin, yrs) {
  rdr <- unique(clm_info$rdr_kr) # rider
  prf <- unique(data[, .(id, age, gnd)]) # profile
  pop <- prf[, .(.N), .(age, gnd, grd)][order(age, gnd, grd)] # population
  npm <- cbind(prf, chk_prm_cnt(clm_info, data, origin, yrs)) # number of payment

  prm_vec <- as.list(vector(length = nrow(pop)))
  for (j in 1:nrow(pop)) {
    age_var <- pop$age[j]
    gnd_var <- pop$gnd[j]
    grd_var <- pop$grd[j]
    prm <- chk_prm(rsk_info, clm_info, age_var, gnd_var, grd_var, yrs)
    tmp <- npm[age == age_var & gnd == gnd_var]
    id <- tmp$id
    tmp <- as.matrix(tmp[, -c(1:3)])

    tmp_vec <- as.list(vector(length = nrow(tmp)))
    for (i in 1:nrow(tmp)) {
      tmp_vec[[i]] <- vapply(seq_along(tmp[i,]), function(x) sum(prm[1:tmp[i, x], x]),
                             FUN.VALUE = as.double(length(tmp[i,])))
    }
    prm_vec[[j]] <- data.table(id, do.call('rbind', tmp_vec))
  }
  z <- do.call('rbind', prm_vec)
  names(z)[-1] <- rdr
  z
}
