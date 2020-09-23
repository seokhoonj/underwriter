glue_code <- function(code) paste0(code[!is.na(code)], collapse = '|')
pull_claim_kcd <- function(code, target) unlist(lapply(regmatches(target, gregexpr(code, target, perl = TRUE)), paste0, collapse = '|'))
