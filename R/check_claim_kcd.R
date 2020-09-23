#' Create each rider's claim kcd code matrix
#'
#' you can create each rider's claim kcd code matrix (from the kcd code matrix).
#' @param code is a kcd code regular expression vector
#' @param target is a claim kcd code vector
#' @param cores is a number of cores for parallel processing
#' @keywords claim kcd code
check_claim_kcd <- function(code, target, cores) {
  if (missing(cores))
    cores <- detectCores()
  if (Sys.info()["sysname"] == "Linux") {
    data <- rbindlist(list(mclapply(keyword, pull_claim_kcd, target = target, mc.cores = cores)))
  } else {
    cl <- makeCluster(cores)
    data <- rbindlist(list(parLapply(cluster, keyword, pull_claim_kcd, target = target)))
    stopCluster(cl)
  }
  return(data)
}
