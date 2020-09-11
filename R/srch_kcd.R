#' Search description of kcd
#'
#' you can search kcd and find its description in the kcd book.
#' @param string A kcd code
#' @examples
#' srch_kcd('J00')
srch_kcd <- function (kcd, lang = "kr") {
  if (missing(kcd))
    stop('Please insert kcd code string or regular expression.')

  if (any(grepl(kcd, kcd_book$code))) {
    if (lang == "kr") {
      df <- kcd_book[grepl(kcd, code), c("code", "kr")]
      nc <- max(nchar(df$code))
      rc <- max(nchar(df$kr))
      line <- nc + 2 + ceiling(rc * 1.5)
      result <- paste0(paste0(stri_pad_right(df$code, width = nc),
                              ": ", df$kr), collapse = "\n")
    }
    else {
      df <- kcd_book[grepl(kcd, code), c("code", "us")]
      nc <- max(nchar(df$code))
      rc <- max(nchar(df$us))
      line <- nc + 2 + rc
      result <- paste0(paste0(stri_pad_right(df$code, width = nc),
                              ": ", df$us), collapse = "\n")
    }
    cat(paste0(rep("=", times = line), collapse = ""), "\n")
    cat(result, "\n")
    cat(paste0(rep("=", times = line), collapse = ""), "\n")
    df
  } else {
    cat("No kcd code is found in the kcd book.")
  }
}
