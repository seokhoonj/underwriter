#' Create number of premium payment matrix by id
#'
#' you can create number of premium payment matrix
#' @param info is a claim information file
#' @param data is a claim history file
#' @param origin is an origin date
#' @param yrs is a payment duration
#' @keywords number of premium payment matrix
chk_prm_cnt <- function(info, data, origin, yrs) {
  # set info variables
  rdr <- info$rdr_kr
  code <- info$kcd
  otime <- info$otime

  # set data variables
  id <- data$id
  cdate <- data$sdate
  target <- data$kcd

  # m, n
  m <- nrow(info)
  n <- nrow(data)

  # clm_cmb & clm_mon
  clm <- chk_clm(code, target)
  clm_1st <- chk_clm_1st(clm, id)
  clm_cmb <- cmb_clm_1st(clm, clm_1st, otime)
  clm_mon <- chk_clm_mon(cdate, origin, m)

  # clm_mon
  clm_mon[clm_cmb == 0] <- 0L
  rownames(clm_mon) <- id
  tmp <- as.data.table(clm_mon, keep.rownames = 'id')[, lapply(.SD, max), by = .(id)]
  names(tmp)[-1] <- rdr
  id <- tmp$id

  # set column variables
  col_ot <- c('id', names(tmp[,-1])[which(otime == 1)])
  col_re <- c('id', names(tmp[,-1])[which(otime == 0)])

  # transform onetime matrix
  mat_ot <- as.matrix(tmp[, ..col_ot][,-1])
  mat_ot[mat_ot == 0] <- yrs * 12

  # transform repeat matrix
  mat_re <- as.matrix(tmp[, ..col_re][,-1])
  mat_re[] <- yrs * 12

  # combine
  z <- cbind(mat_ot, mat_re)
  rownames(z) <- id
  col <- unique(rdr)
  z[, col]
}
