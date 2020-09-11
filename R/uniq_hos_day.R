#' uniq_hos_day Function
#'
#' you can calculate the hospitalization days by several groups like id, icd code
#' considering duplicated days.
#' @param data.frame or data.table with hospitalization data like id, icd code, start date, end date.
#' @keywords hospitalization
#'
uniq_hos_day <- function(data, var_id, var_from, var_to) {
  # check arguments
  if (missing(data)) stop('Please check input variables.')
  if (class(data[[var_from]]) != 'Date') stop("Please check the 'var_from' column's data type.")
  if (class(data[[var_to]]) != 'Date') stop("Please check the 'var_to' column's data type.")
  if (any(is.na(data[[var_from]]))) stop("Please check the 'missing data' in 'var_from' column.")
  if (any(is.na(data[[var_to]]))) stop("Please check the 'missing data' in 'var_to' column.")

  # transfrom to data.table
  data <- as.data.table(data)
  setorderv(data, var_id)

  # set arguments
  id = as.matrix(data[, ..var_id])
  from = data[[var_from]]
  to = data[[var_to]]
  if (any(to - from < 0)) stop("Some 'var_from' data are greater than 'var_to' data.")

  # cpp code
  hos <- rcpp_uniq_day(id, from ,to)

  # binding with unique ids
  data_uniq <- unique(data[, ..var_id])
  z <- cbind(data_uniq, hos = hos)

  return(z)
}
