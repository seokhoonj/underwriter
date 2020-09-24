glue_code <- function(code) paste0(code[!is.na(code)], collapse = '|')
splt_code  = function(code) strsplit(code, split = '\\|')[[1]]
pull_claim_kcd <- function(code, target) unlist(lapply(regmatches(target, gregexpr(code, target, perl = TRUE)), paste0, collapse = '|'))
