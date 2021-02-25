#' Reshape kcd code column long-to-wide version by each var_id
#'
#' you can transfrom kcd code columns long-to-wide version by each var_id
#' @param data is a claim data
#' @param var_id is var_id
#' @param target is a kcd column variable
#' @param prefix is a prefix for new kcd columns
#' @param glue is to glue all new kcd columns
#' @keywords reshape kcd code column long-to-wide
cst_unq_kcd <- function(data, var_id, target, prefix = "kcd", glue = TRUE) {
  var_id <- vapply(substitute(var_id), deparse, FUN.VALUE = "character")
  var_id <- names(data)[match(var_id, names(data), 0L)]
  target <- deparse(substitute(target))
  z <- copy(data)
  z[, rank := rank(get(target), ties.method = "first"), by = var_id]
  f <- formula(paste(paste(var_id, collapse = " + "), " ~ rank"))
  z <- dcast.data.table(z, formula = f, value.var = target)
  var_cst <- paste0(prefix, str_pad(names(z)[-match(var_id, names(z), 0L)],
                                    width = nchar(length(names(z))-length(var_id)),
                                    pad = "0"))
  vars <- c(var_id, var_cst)
  setnames(z, vars)
  if (glue) {
    z <- data.table(z[, ..var_id], kcd = apply(z[, ..var_cst], 1, function(x) glue_code(sort(unique(x)))))
    setnames(z, c(var_id, prefix))
  }
  return(z)
}
