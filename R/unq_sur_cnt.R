#' Calculate unique surgery counts
#'
#' you can calculate the number of surgery by several groups like id, icd code
#' considering duplicated days.
#' @param data is claim data with id, icd code, start date, end date.
#' @param var_id is the grouping column variables
#' @param var_to is the surgery date column variable
#' @keywords surgery counts
unq_sur_cnt <- function(data, var_id, var_to) {
  var_id <- vapply(substitute(var_id), deparse, FUN.VALUE = "character")
  var_id <- names(data)[match(var_id, names(data), 0L)]
  var_to <- parse(text = deparse(substitute(var_to)))
  data[, .(sur = uniqueN(eval(var_to))), var_id]
}
