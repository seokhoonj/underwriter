#' Get ids with specific kcd codes
#'
#' you can get ids with specific kcd (Korean Standard Classification of Diseases) codes
#' considering duplicated days.
#' @param data is a claim history file
#' @param var_id is an id column
#' @param var_kcd is a kcd column
#' @param code is kcd codes (regular expressions are also ok)
#' @param only if 'only' variable is TRUE, it means 'having only that kcd code'
#' @keywords id with kcd code
get_id_with_kcd <- function(data, var_id, var_kcd, code, only) {

  if (missing(only)) only <- FALSE

  var_id <- deparse(substitute(var_id))
  var_kcd <- deparse(substitute(var_kcd))
  ids <- unique(data[grepl(code, data[[var_kcd]])][[var_id]])
  z <- data[data[[var_id]] %in% ids]

  if (only) {
    ids_excl <- unique(z[!grepl(code, z[[var_kcd]])]$id)
    z <- z[!z[[var_id]] %in% ids_excl]
  }

  return(z)
}
