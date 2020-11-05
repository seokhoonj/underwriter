#' Categorize the age variable
#'
#' you can categorize the age variable.
#' @param data is a claim data
#' @param var is a age column variables
#' @param interval is an interval number
#' @param right is a limit direction (default FALSE)
#' @keywords hospitalization
cut_age <- function(data, var, interval, right = FALSE) {
  age <- eval(substitute(var), data)
  min <- floor(min(age) / interval) * interval
  max <- ceiling(max(age) / interval) * interval
  if (max(age) == max) max <- ceiling(max(age) / interval + 1) * interval
  cut(age, breaks = seq(min, max, interval), right = right)
}
