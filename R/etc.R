
# base functions ----------------------------------------------------------

dolock <- function(obj, key) aes_cbc_encrypt(serialize(obj, NULL), key = sha256(charToRaw(key)))
unlock <- function(obj, key) unserialize(aes_cbc_decrypt(obj, key = sha256(charToRaw(key))))
unilen <- function(x) length(unique(x))
getsiz <- function(x) format(object.size(x), unit = 'Mb')
getcol <- function(data, str, contain = TRUE) if (contain) names(data)[ grepl(str, names(data))] else names(data)[!grepl(str, names(data))]
varstr <- function(data, x) {
  x <- vapply(substitute(x), deparse, FUN.VALUE = "character")
  names(data)[match(x, names(data), 0L)]
}
join <- function(..., by, all = FALSE, all.x = all, all.y = all, sort = TRUE) {
  l <- list(...)
  Reduce(function(...) merge(..., by = by, all = all, all.x = all.x, all.y = all.y, sort = sort), l)
}


# matrix functions --------------------------------------------------------

dup_col_min <- function(data) {
  coln <- colnames(data)
  colt <- table(coln)
  dups <- names(colt[colt > 1])
  if (length(dups) > 0) {
    loc <- as.list(vector(length = length(dups)))
    tmp <- as.list(vector(length = length(dups)))
    for (i in seq_along(dups)) {
      loc[[i]] <- which(coln == dups[i])
      tmp[[i]] <- apply(data[, loc[[i]], drop = FALSE], 1, min)
    }
    dat <- data[, -unlist(loc), drop = FALSE]
    tmp <- do.call('cbind', tmp)
    colnames(tmp) <- dups
    z <- cbind(dat, tmp)
    data <- z[, unique(coln), drop = FALSE]
  }
  return(data)
}


# kcd code functions ------------------------------------------------------

glue_code <- function(code) paste0(code[!is.na(code)], collapse = '|')
splt_code <- function(code) strsplit(code, split = '\\|')[[1]]
excl_code <- function(code) paste0('^((?!', code, ').)*$')
pull_code <- function(code, string) unique(unlist(regmatches(string, gregexpr(code, string))))
pull_clm_kcd <- function(code, target) unlist(lapply(regmatches(target, gregexpr(code, target, perl = TRUE)), paste0, collapse = '|'))


# date functions ----------------------------------------------------------

diff_mon <- function(date, origin) (year(date) - year(as.Date(origin))) * 12 + month(date) - month(as.Date(origin)) + 1L


# model functions ---------------------------------------------------------

k_fold <- function(data, k) {
  n <- nrow(data)
  r <- round(n / k)
  l <- n - (r * (k-1))
  t <- c(rep(r, k-1), l)
  l <- rep(1:k, times = t)
  s <- sample(n, size = n)
  z <- split(s, l)
  names(z) <- paste0('fold.', str_pad(names(z), width = 2, pad = '0'))
  z
}

k_spl <- function(x, k) {
  v <- unique(x)
  n <- length(v)
  s <- ceiling(n / k)
  r <- n + s * (1-k)
  q <- c(rep(1:(k-1), each = s), rep(k, times = r))
  z <- split(v, q)
  names(z) <- paste0("spl.", str_pad(names(z), width = 2,
                                     pad = "0"))
  return(z)
}
