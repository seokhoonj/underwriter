#' Create premium matrix by gender, age, grade
#'
#' you can create premium matrix by gender, age, grade
#' @param rsk_info is a risk information file
#' @param clm_info is a claim information file
#' @param old is an age
#' @param gnd is an gender
#' @param grd is an grade
#' @param yrs is a duration of observation
#' @keywords premium matrix

chk_prm <- function(rsk_info, clm_info, old, gnd, grd, yrs) {
  # join infos
  clm_info <- unique(clm_info[, .(cate, method, rdr_cd, rdr_us, rdr_kr, weight, amt, ratio, otime)])
  info <- rsk_info[clm_info, on = .(rdr_kr)]
  nrdr <- length(clm_info$rdr_kr)

  # set vars
  old <- old + seq(0, yrs - 1)
  gnd <- c('male', 'female')[gnd]
  grd <- c(0, grd)

  # set rsk book as tmp
  tmp <- info[age %in% old & grade %in% grd]

  # set rsk, amt, rat
  rsk <- split(tmp[[gnd]], tmp$age)
  amt <- split(tmp$amt, tmp$age)
  rat <- split(tmp$rat, tmp$age)

  # set ratio as 1 from 2nd year
  for (i in 1:(yrs - 1)) {
    rat[-1][[i]] <- rep(1, nrdr)
  }

  # prm
  prm <- lapply(1:yrs, function(x) round(rsk[[x]] * rat[[x]] * amt[[x]] * 1e+06 / 12))
  z <- do.call("rbind", lapply(prm, function(x) matrix(rep(x, 12), nrow = 12, byrow = TRUE)))
  return(z)
}
