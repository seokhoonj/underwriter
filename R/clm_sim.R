#' Claim simulation
#'
#' Claim simulation by specific files (info, data)
#' @param info is a claim information file
#' @param data is a claim history file
#' @param origin is an origin date
#' @keywords claim simulation
clm_sim <- function(info, data, origin) {

  # set info arguments
  code   <- info$kcd
  otime  <- info$otime
  rat    <- info$ratio
  amt    <- info$amt
  mul    <- info$multiple

  # set data arguments
  id     <- data$id
  target <- data$kcd
  edate  <- data$edate

  # length
  m <- nrow(info)
  n <- nrow(data)

  # check_claim
  clm <- chk_clm(code, target)

  # check_claim_1st
  clm_1st <- chk_clm_1st(clm, id)

  # combine claim and 1st claim
  loc <- which(otime == 1)
  clm[, loc] <- clm_1st[, loc]

  # claim year
  clm_yrs <- chk_clm_yrs(edate, origin = origin, m = m)

  # clm ratio
  clm_rat <- chk_clm_rat(clm_yrs, rat)

  # clm amount
  clm_amt <- matrix(rep(amt, n), ncol = m, byrow = TRUE)

  # clm multiple
  clm_mul <- matrix(rep(mul, n), ncol = m, byrow = TRUE)

  # final
  clm_fin <- clm * clm_rat * clm_amt * clm_mul

  return(clm_fin)
}
