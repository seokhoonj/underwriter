
# base functions ----------------------------------------------------------

sizeof = function(x) format(object.size(x), unit = 'Mb')
unilen = function(x) length(unique(x))
getcol = function(df, str, contain = TRUE) if (contain) names(df)[ grepl(str, names(df))] else names(df)[!grepl(str, names(df))]

# kcd code functions ------------------------------------------------------

glue_code <- function(code) paste0(code[!is.na(code)], collapse = '|')
splt_code <- function(code) strsplit(code, split = '\\|')[[1]]
excl_code <- function(code) paste0('^((?!', code, ').)*$')
pull_code <- function(code, string) unique(unlist(regmatches(string, gregexpr(code, string))))
pull_clm_kcd <- function(code, target) unlist(lapply(regmatches(target, gregexpr(code, target, perl = TRUE)), paste0, collapse = '|'))
