#' Calculate the basic kcd code statistics from the data including specific kcd code
#'
#' you can calculate the basic kcd code statistics
#' @param data is claim data with id, kcd code
#' @param var_id is the id variable for the unique count
#' @param var_kcd is the kcd code column variable
#' @param code is a specific kcd code
#' @keywords kcd statistics
stat_kcd_rat <- function(data, var_id, var_kcd, code) {
  # variables
  var_kcd   <- deparse(substitute(var_kcd))
  var_kcd_p <- parse(text = var_kcd)
  var_id   <- deparse(substitute(var_id))
  var_id_p <- parse(text = var_id)

  # total summary
  ntot <- unilen(data[[var_id]])
  dtot <- data[, .(tot = uniqueN(eval(var_id_p))), var_kcd]
  dtot[, rat_tot := round(tot / ntot * 100, 2)]

  # observation
  obs_id <- unique(data[grepl(code, data[[var_kcd]])][[var_id]])
  dobs <- data[data[[var_id]] %in% obs_id]

  # observation summary
  nobs <- unilen(dobs[[var_id]])
  dobs <- dobs[, .(obs = uniqueN(eval(var_id_p))), var_kcd]
  dobs[, rat_obs := round(obs / nobs * 100, 2)]

  # merge
  z <- merge(dtot, dobs, by = 'kcd', all = TRUE)
  z[, obs := ifelse(is.na(obs), 0, obs)]
  z[, rat_obs := ifelse(is.na(rat_obs), 0, rat_obs)]
  z[, diff := rat_obs - rat_tot]

  # return
  get_kcd(paste0(code, '$'))
  z[order(obs, decreasing = TRUE)]
}
