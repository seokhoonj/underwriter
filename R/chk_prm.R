#' Create premium matrix by gender, age, grade
#'
#' you can create premium matrix by gender, age, grade
#' @param info is a risk information file
#' @param rider is a rider set
#' @param old is an age
#' @param gnd is an gender
#' @param grd is an grade
#' @param yrs is a duration of observation
#' @keywords premium matrix
chk_prm <- function(info, rider, old, gnd, grd, yrs) {
  # set variables
  old <- old + seq(0, yrs-1)
  gnd <- if (gnd) 'male' else 'female'
  grd <- c(0, grd)

  # set rsk book as tmp
  tmp <- info[rdr_kr %in% rider & age %in% old & grade %in% grd]

  # set rsk, amt, rat
  rsk <- split(tmp[[gnd]], tmp$age)
  amt <- split(tmp$amt, tmp$age)
  rat <- split(tmp$rat, tmp$age)

  # set ratio as 1 from 2nd year
  for (i in 1:(yrs-1)) {
    rat[-1][[i]] <- rep(1., length(rider))
  }

  # prm
  prm <- lapply(1:yrs, function(x) round(rsk[[x]] * rat[[x]] * amt[[x]] * 1000000 / 12))
  z <- do.call('rbind', lapply(prm, function(x) matrix(rep(x, 12), nrow = 12, byrow = TRUE)))

  return(z)
}