#' Split hospitalization days by origin date (virtual uw date)
#'
#' you can create new hospitalization days data frame divided by origin date.
#' @param data or data.table with hospitalization data
#' @param var_from start date column variable
#' @param var_to end date column variable
#' @keywords split hospitalization days by origin date
spl_hos_day <- function(data, var_from, var_to) {
  str_from <- deparse(substitute(var_from))
  str_to   <- deparse(substitute(var_to))

  s <- as.integer(substr(data[[str_from]], 1, 4))
  e <- as.integer(substr(data[[str_to]]  , 1, 4))
  years <- unique(c(s[which(e-s > 0)], e[which(e-s > 0)]))
  dates <- as.Date(paste0(as.character(years), "0101"), "%Y%m%d")

  for (i in seq_along(dates)) {
    from <- eval(substitute(var_from), data)
    to   <- eval(substitute(var_to), data)

    tmp_e <- data[from >= dates[i] | to <  dates[i]]
    tmp_b <- data[from <  dates[i] & to >= dates[i]]
    tmp_a <- copy(tmp_b)

    if (nrow(tmp_a) > 0) {
      tmp_b[[str_to]]   <- dates[i]-1L
      tmp_a[[str_from]] <- dates[i]
    }

    data <- rbind(tmp_e, tmp_b, tmp_a)
  }
  cat("Please check hospitalization days or claim year, you may have to recalculate!\n")
  setorderv(data, names(data))
  return(data)
}
