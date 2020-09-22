#' Calculate unique hospitalization days
#'
#' you can calculate the hospitalization days by several groups like id, icd code
#' considering duplicated days.
#' @param data or data.table with hospitalization data like id, icd code, start date, end date.
#' @param var_id grouping column string
#' @param var_from start date column string
#' @param var_to end date column string
#' @keywords hospitalization
uniq_hos_day <- function(data, var_id, var_from, var_to) {

  # check arguments missing
  if (missing(data) | missing(var_id) | missing(var_from) | missing(var_to))
    stop('Please check input variables.')

  # transfrom to strings
  var_id   <- vapply(substitute(var_id)[-1L], deparse, FUN.VALUE = 'character')
  var_from <- deparse(substitute(var_from))
  var_to   <- deparse(substitute(var_to))

  # check arguments type
  if (class(data[[var_from]]) != 'Date') stop("Please check the 'var_from' column's data type.")
  if (class(data[[var_to]]) != 'Date') stop("Please check the 'var_to' column's data type.")

  # check arguments' missing data
  if (any(is.na(data[[var_from]]))) stop("Please check the 'missing data' in 'var_from' column.")
  if (any(is.na(data[[var_to]]))) stop("Please check the 'missing data' in 'var_to' column.")

  # transfrom to data.table
  data <- as.data.table(data)
  setorderv(data, c(var_id, var_from, var_to))

  # set arguments
  id   <- as.matrix(data[, ..var_id])
  from <- data[[var_from]]
  to   <- data[[var_to]]
  if (any(to - from < 0)) stop("Some 'var_from' data are greater than 'var_to' data.")

  # cpp code
  hos <- underwriter::rcpp_uniq_day(id, from ,to)

  # binding with unique ids
  data_uniq <- unique(data[, ..var_id])
  z <- cbind(data_uniq, hos = hos)

  return(z)
}
