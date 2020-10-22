
# base functions ----------------------------------------------------------

dolock <- function(obj, key) aes_cbc_encrypt(serialize(obj, NULL), key = sha256(charToRaw(key)))
unlock <- function(obj, key) unserialize(aes_cbc_decrypt(obj, key = sha256(charToRaw(key))))
unilen <- function(x) length(unique(x))
getsiz <- function(x) format(object.size(x), unit = 'Mb')
getcol <- function(data, str, contain = TRUE) if (contain) names(data)[ grepl(str, names(data))] else names(data)[!grepl(str, names(data))]
join <- function(..., by, all = FALSE, all.x = all, all.y = all, sort = TRUE) {
  l <- list(...)
  Reduce(function(...) merge(..., by = by, all = all, all.x = all.x, all.y = all.y, sort = sort), l)
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
