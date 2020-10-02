
# base functions ----------------------------------------------------------

unlock <- function(obj, key) unserialize(aes_cbc_decrypt(obj, key = sha256(charToRaw(key))))
unilen <- function(x) length(unique(x))
getsiz <- function(x) format(object.size(x), unit = 'Mb')
getcol <- function(data, str, contain = TRUE) if (contain) names(data)[ grepl(str, names(data))] else names(data)[!grepl(str, names(data))]

# kcd code functions ------------------------------------------------------

glue_code <- function(code) paste0(code[!is.na(code)], collapse = '|')
splt_code <- function(code) strsplit(code, split = '\\|')[[1]]
excl_code <- function(code) paste0('^((?!', code, ').)*$')
pull_code <- function(code, string) unique(unlist(regmatches(string, gregexpr(code, string))))
pull_clm_kcd <- function(code, target) unlist(lapply(regmatches(target, gregexpr(code, target, perl = TRUE)), paste0, collapse = '|'))
