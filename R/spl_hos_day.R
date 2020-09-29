#' Split hospitalization days by origin date (virtual uw date)
#'
#' you can create new hospitalization days data frame divided by origin date.
#' @param data or data.table with hospitalization data
#' @param var_from start date column variable
#' @param var_to end date column variable
#' @keywords split hospitalization days by origin date
spl_hos_day <- function(data, var_from, var_to, origin, all = TRUE) {
  from  <- eval(substitute(var_from), data)
  to    <- eval(substitute(var_to), data)
  tmp_e <- data[from >= as.Date(origin) | to <= as.Date(origin),]
  tmp_b <- data[from < as.Date(origin) & to > as.Date(origin),]
  tmp_a <- copy(tmp_b)
  tmp_b[[deparse(substitute(var_to))]] <- as.Date(origin) - 1L
  tmp_a[[deparse(substitute(var_from))]] <- as.Date(origin)
  if (all) return(rbind(tmp_e, tmp_b, tmp_a)) else return(rbind(tmp_b, tmp_a))
  cat('please check hospitalization days or claim year, you may have to recalculate!')
}
