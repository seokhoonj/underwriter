#' Calculate the basic kcd code statistics from the data including specific kcd code
#'
#' you can calculate the basic kcd code statistics
#' @param data is claim data with id, kcd code
#' @param var_id is the id variable for the unique count
#' @param var_kcd is the kcd code column variable
#' @param var_clm is the claim amount column variable
#' @param var_prm is the risk premium amount column variable
#' @param code is a specific kcd code
#' @keywords kcd statistics
stat_kcd <- function(data, var_id, var_kcd, var_clm, var_prm, code) {

  # variables
  var_kcd  <- deparse(substitute(var_kcd))
  var_kcd_p <- parse(text = var_kcd)
  var_id  <- deparse(substitute(var_id))
  var_id_p <- parse(text = var_id)

  if (missing(var_clm) | missing(var_prm)) {

    # total summary
    ntot <- unilen(data[[var_id]])
    dtot <- data[, .(tot_pop = uniqueN(eval(var_id_p))), var_kcd]
    dtot[, `:=`(tot_rat, round(tot_pop/ntot * 100, 2))]

    # observation
    obs_id <- unique(data[grepl(code, data[[var_kcd]])][[var_id]])
    dobs <- data[data[[var_id]] %in% obs_id]

    # observation summary
    nobs <- unilen(dobs[[var_id]])
    dobs <- dobs[, .(obs_pop = uniqueN(eval(var_id_p))), var_kcd]
    dobs[, `:=`(obs_rat, round(obs_pop/nobs * 100, 2))]

    # merge
    z <- merge(dtot, dobs, by = 'kcd', all = TRUE)
    z[, `:=`(obs_pop, ifelse(is.na(obs_pop), 0, obs_pop))]
    z[, `:=`(obs_rat, ifelse(is.na(obs_rat), 0, obs_rat))]

  } else {

    # variables
    var_clm <- deparse(substitute(var_clm))
    var_clm_p <- parse(text = var_clm)
    var_prm <- deparse(substitute(var_prm))
    var_prm_p <- parse(text = var_prm)

    # total summary
    ntot <- unilen(data[[var_id]])
    dtot <- data[, .(tot_pop = uniqueN(eval(var_id_p)),
                     tot_clm = sum(eval(var_clm_p)),
                     tot_prm = sum(eval(var_prm_p))), var_kcd]
    dtot[, `:=`(tot_rat, round(tot_pop/ntot * 100, 2))]

    # observation
    obs_id <- unique(data[grepl(code, data[[var_kcd]])][[var_id]])
    dobs <- data[data[[var_id]] %in% obs_id]

    # observation summary
    nobs <- unilen(dobs[[var_id]])
    dobs <- dobs[, .(obs_pop = uniqueN(eval(var_id_p)),
                     obs_clm = sum(eval(var_clm_p)),
                     obs_prm = sum(eval(var_prm_p))), var_kcd]
    dobs[, `:=`(obs_rat, round(obs_pop/nobs * 100, 2))]

    # merge
    z <- merge(dtot, dobs, by = "kcd", all = TRUE)
    z[, `:=`(obs_pop, ifelse(is.na(obs_pop), 0, obs_pop))]
    z[, `:=`(obs_clm, ifelse(is.na(obs_clm), 0, obs_clm))]
    z[, `:=`(obs_prm, ifelse(is.na(obs_prm), 0, obs_prm))]
    z[, `:=`(obs_rat, ifelse(is.na(obs_rat), 0, obs_rat))]

  }


  # return
  get_kcd(paste0(code, "$"))
  z[order(obs_pop, decreasing = TRUE)]
}
