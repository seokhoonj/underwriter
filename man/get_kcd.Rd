% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/get_kcd.R
\name{get_kcd}
\alias{get_kcd}
\title{Get descriptions of kcd codes}
\usage{
get_kcd(kcd, lang = "kr")
}
\arguments{
\item{string}{A kcd code or regular expression}
}
\description{
you can get kcd and find its description in the kcd book.
}
\examples{
get_kcd('J00')
get_kcd('^M51$')
}
