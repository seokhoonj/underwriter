#' Split each kcd codes and increase the data by the number of each kcd code
#'
#' you can create increased data splitted by each unique kcd code
#' @param data is a claim data
#' @param var is a kcd column variable
#' @keywords split each kcd codes
spl_unq_kcd <- function(data, var) {
  # save data columns names
  old_names <- names(data)


  # split kcd codes uniquely
  kcd <- eval(substitute(var), data)
  kcd <- lapply(kcd, splt_code)
  len <- vapply(kcd, length, FUN.VALUE = 1)

  # expand dataset
  var_kcd <- deparse(substitute(var))
  var_ids <- getcol(data, var_kcd, contain = FALSE)
  z <- as.data.table(lapply(var_ids, function(x) rep(data[[x]], times = len)))
  setnames(z, var_ids)

  # add kcd
  z[[var_kcd]] <- unlist(kcd)
  setcolorder(z, old_names)

  return(z)
}
