% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/elapsed_day.R
\name{elapsed_day}
\alias{elapsed_day}
\title{Calculate elapsed days}
\usage{
elapsed_day(data, var_id, var_to, origin)
}
\arguments{
\item{data}{is claim data with id, icd code, start date, end date.}

\item{var_id}{is the grouping column variables}

\item{var_to}{is the end date column variable}

\item{origin}{is an origin date}
}
\description{
you can calculate the minimum elapsed days by several groups like id, icd code
considering duplicated days.
}
\keyword{days}
\keyword{elapsed}
