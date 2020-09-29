#' Get descriptions of kcd codes
#'
#' you can get kcd and find its description in the kcd book.
#' @param string A kcd code or regular expression
#' @examples
#' get_kcd('J00')
#' get_kcd('^M51$')
get_kcd <- function (kcd, lang = "kr") {
  if (missing(kcd))
    stop('Please insert kcd code string or regular expression.')

  if (any(grepl(toupper(kcd), kcd_book$code))) {
    if (lang == "kr") {
      df <- kcd_book[grepl(kcd, kcd_book$code, ignore.case = TRUE), c("code", "kr")]
      nc <- max(nchar(df$code))
      rc <- max(nchar(df$kr))
      iter <- nc + nchar(" | ") + ceiling(rc * 1.6)
      line <- paste0(rep("=", times = iter), collapse = "")
      result <- paste0(paste0(stri_pad_right(df$code, width = nc),
                              " | ", df$kr), collapse = "\n")
    }
    else {
      df <- kcd_book[grepl(kcd, kcd_book$code, ignore.case = TRUE), c("code", "us")]
      nc <- max(nchar(df$code))
      rc <- max(nchar(df$us))
      iter <- nc + nchar(" | ") + rc
      line <- paste0(rep("=", times = iter), collapse = "")
      result <- paste0(paste0(stri_pad_right(df$code, width = nc),
                              " | ", df$us), collapse = "\n")
    }
    cat(line,   "\n")
    cat(result, "\n")
    cat(line,   "\n")
    invisible(df)
  } else {
    cat("No kcd code is found in the kcd book.\n")
  }
}
