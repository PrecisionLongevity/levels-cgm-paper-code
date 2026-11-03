#!/usr/bin/env Rscript
# Computes standard MAGE (moving-average algorithm; Fernandes et al., J Diabetes Sci Technol 2022;16:576-577) using
# iglu::mage(version = 'ma') at the package defaults (short_ma = 5, long_ma = 32, direction = 'avg', inter_gap = 45,
# max_gap = 180), one value per participant over that participant's dense-day trace.
#
# Called by analysis/clinical/mage_standard.py (stage 'iglu'); not a stand-alone producer.
#
# Usage: Rscript analysis/clinical/mage_standard_iglu.R <cache_dir> <workers> <chunk file> [<chunk file> ...]
#        (chunk files are relative to <cache_dir>/traces)
#
# Inputs
#   <cache_dir>/traces/<chunk>.parquet  (USER_ID, TS [UTC], VALUE; accepted readings on the dense days)
#
# Outputs
#   <cache_dir>/iglu/<chunk>.parquet  (USER_ID, mage_iglu, n_input_rows, seconds) - participant level, restricted
#   <cache_dir>/iglu/session.json     (R and iglu versions, the mage() formals actually used)
#
# The process time zone is UTC, so iglu's 5-minute interpolation grid (CGMS2DayByDay) is built in UTC; with whole-hour or
# quarter-hour local offsets the grid points are the same instants as in local time, and only segment durations enter the
# result, so the value does not depend on the display time zone (and no daylight-saving discontinuity can enter the grid).
Sys.setenv(TZ = 'UTC')
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) >= 3)
cache <- args[1]; workers <- as.integer(args[2]); chunks <- args[-(1:2)]
.libPaths(c(file.path(cache, 'r_lib'), .libPaths()))
suppressPackageStartupMessages({library(iglu); library(arrow); library(parallel); library(jsonlite)})
stopifnot(as.character(packageVersion('iglu')) == '4.2.2')
f <- formals(iglu::mage)
used <- list(version = eval(f$version)[1], short_ma = f$short_ma, long_ma = f$long_ma, direction = eval(f$direction)[1],
             inter_gap = f$inter_gap, max_gap = f$max_gap, sd_multiplier = f$sd_multiplier)
stopifnot(identical(used$version, 'ma'), used$short_ma == 5, used$long_ma == 32, identical(used$direction, 'avg'),
          used$inter_gap == 45, used$max_gap == 180)
dir.create(file.path(cache, 'iglu'), showWarnings = FALSE)
write_json(list(R = R.version.string, iglu = as.character(packageVersion('iglu')), arrow = as.character(packageVersion('arrow')),
                call = "iglu::mage(data.frame(id, time, gl), version = 'ma')  # all other arguments at their defaults",
                defaults = used, TZ = Sys.getenv('TZ')),
           file.path(cache, 'iglu', 'session.json'), auto_unbox = TRUE, pretty = TRUE, digits = NA)
one <- function(s) {
  t0 <- proc.time()[['elapsed']]
  x <- data.frame(id = s$USER_ID, time = s$TS, gl = s$VALUE)
  m <- suppressMessages(suppressWarnings(iglu::mage(x, version = 'ma')))
  stopifnot(nrow(m) == 1)
  data.frame(USER_ID = s$USER_ID[1], mage_iglu = as.numeric(m$MAGE[1]), n_input_rows = nrow(s),
             seconds = proc.time()[['elapsed']] - t0, stringsAsFactors = FALSE)
}
for (ch in chunks) {
  t0 <- Sys.time()
  d <- as.data.frame(read_parquet(file.path(cache, 'traces', ch)))
  stopifnot(identical(attr(d$TS, 'tzone'), 'UTC'), !anyNA(d$VALUE), !anyNA(d$TS))
  parts <- split(d[, c('USER_ID', 'TS', 'VALUE')], d$USER_ID)
  parts <- parts[order(-vapply(parts, nrow, 1L))]   # longest first: dynamic scheduling keeps all workers busy
  res <- mclapply(parts, one, mc.cores = workers, mc.preschedule = FALSE)
  bad <- vapply(res, function(r) !is.data.frame(r), TRUE)
  if (any(bad)) stop(sprintf('%s: %d participants failed: %s', ch, sum(bad), paste(head(names(res)[bad], 3), collapse = ', ')))
  out <- do.call(rbind, res); out <- out[order(out$USER_ID), ]; rownames(out) <- NULL
  stopifnot(nrow(out) == length(unique(d$USER_ID)))
  tmp <- file.path(cache, 'iglu', paste0(ch, '.tmp'))
  write_parquet(out, tmp); file.rename(tmp, file.path(cache, 'iglu', ch))
  cat(sprintf('%s: %d participants, %.0f s\n', ch, nrow(out), as.numeric(Sys.time() - t0, units = 'secs')))
}
