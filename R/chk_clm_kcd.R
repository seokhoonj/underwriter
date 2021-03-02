#' Create each rider's claim kcd code matrix
#'
#' you can create each rider's claim kcd code matrix (from the kcd code matrix).
#' @param code is a kcd code regular expression vector
#' @param target is a claim kcd code vector
#' @param cores is a number of cores for parallel processing
#' @keywords claim kcd code
chk_clm_kcd <- function(code, target, cores) {
  if (missing(cores))
    cores <- detectCores()
  if (Sys.info()["sysname"] == "Linux") {
    z <- matrix(unlist(mclapply(code, pull_clm_kcd, target = target, mc.cores = cores)),
                ncol = length(code), byrow = FALSE)
  } else {
    cl <- makeCluster(cores)
    z <- as(matrix(unlist(parLapply(cl, code, pull_clm_kcd, target = target)),
                   ncol = length(code), byrow = FALSE), "CsparseMatrix")
    stopCluster(cl)
  }
  return(z)
}
