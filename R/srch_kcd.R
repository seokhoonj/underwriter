#' Search description of kcd
#'
#' you can search kcd and find its description in the kcd book.
#' @param string A kcd code
#' @examples
#' srch_kcd('J00')
srch_kcd = function(kcd, lang = 'kr') {
  df <- kcd_book[grepl(kcd, code)]
  nc <- max(nchar(df$code))
  if (lang == 'kr') {
    rc <- max(nchar(df$kr))
    result <- paste0(paste0(stri_pad_right(df$code, width = nc), ': ', df$kr), collapse = '\n')
  } else {
    rc <- max(nchar(df$us))
    result <- paste0(paste0(stri_pad_right(df$code, width = nc), ': ', df$us), collapse = '\n')
  }
  cat(paste0(rep('=', times = nc + 2 + rc), collapse=''), '\n')
  cat(result, '\n')
  cat(paste0(rep('=', times = nc + 2 + rc), collapse=''), '\n')
}
