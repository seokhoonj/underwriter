% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/get_id_with_kcd.R
\name{get_id_with_kcd}
\alias{get_id_with_kcd}
\title{Get ids with specific kcd codes}
\usage{
get_id_with_kcd(data, var_id, var_kcd, code, only)
}
\arguments{
\item{data}{is a claim history file}

\item{var_id}{is an id column}

\item{var_kcd}{is a kcd column}

\item{code}{is kcd codes (regular expressions are also ok)}

\item{only}{if 'only' variable is TRUE, it means 'having only that kcd code'}
}
\description{
you can get ids with specific kcd (Korean Standard Classification of Diseases) codes
considering duplicated days.
}
\keyword{code}
\keyword{id}
\keyword{kcd}
\keyword{with}
