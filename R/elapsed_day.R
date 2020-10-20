#' Calculate elapsed days
#'
#' you can calculate the minimum elapsed days by several groups like id, icd code
#' considering duplicated days.
#' @param data is claim data with id, icd code, start date, end date.
#' @param var_id is the grouping column variables
#' @param var_to is the end date column variable
#' @param origin is an origin date
#' @keywords elapsed days
elapsed_day <- function(data, var_id, var_to, origin) {
  var_id <- vapply(substitute(var_id), deparse, FUN.VALUE = "character")
  var_id <- names(data)[match(var_id, names(data), 0L)]
  var_to <- parse(text = deparse(substitute(var_to)))
  data[, .(elapsed = as.numeric(as.Date(origin) - max(eval(var_to)))), var_id]
}
