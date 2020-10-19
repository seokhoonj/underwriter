#' Calculate elapsed days
#'
#' you can calculate the minimum elapsed days by several groups like id, icd code
#' considering duplicated days.
#' @param ... claim data with id, icd code, start date, end date.
#' @param var_id grouping column variables
#' @param var_to end date column variable
#' @param origin is an origin date
#' @keywords elapsed days
elapsed_day <- function(..., var_id, var_to, origin) {
  l <- list(...)
  var_id <- vapply(substitute(var_id), deparse, FUN.VALUE = "character")
  var_id <- names(l[[1L]])[match(var_id, names(l[[1L]]), 0L)]
  var_to <- deparse(substitute(var_to))
  v <- as.list(vector(length = length(l)))
  for (i in seq_along(l)) {
    v[[i]] <- l[[i]][, .(elapsed = vapply(.SD, function(x) as.numeric(as.Date(origin) - max(x)), FUN.VALUE = 1)), var_id, .SDcols = var_to]
  }
  z <- do.call('rbind', v)
  z[order(id, kcd), .(elapsed = vapply(.SD, min, FUN.VALUE = 1)), var_id, .SDcols = 'elapsed']
}
