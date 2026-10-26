# Independent R replication of the baselines and scoring, in base R only.
#   Rscript r/replicate.R
# Reads the splits written by `python -m projections.export_for_r`, recomputes the
# Marcel-style projection, checks it against the Python result, fits an
# empirical-Bayes beta-binomial model, and scores everything on the same metrics.

in_dir <- "results/r_input"
weights <- c(5, 4, 3)
regression <- 1200

log_loss <- function(p, k, n) {
  p <- pmin(pmax(p, 1e-9), 1 - 1e-9)
  -sum(k * log(p) + (n - k) * log1p(-p)) / sum(n)
}
brier <- function(p, k, n) sum(k * (1 - p)^2 + (n - k) * p^2) / sum(n)
wrmse <- function(p, k, n) sqrt(sum(n * (p - k / n)^2) / sum(n))

marcel <- function(hist, target_year) {
  w <- weights[target_year - hist$yearID]
  w[is.na(w)] <- 0
  lg <- hist$league_rate[1]
  wk <- tapply(w * hist$k, hist$playerID, sum)
  wn <- tapply(w * hist$n, hist$playerID, sum)
  (wk + regression * lg) / (wn + regression)
}

# Beta-binomial prior fit by maximum likelihood on players' pooled history,
# then each player's posterior mean rate.
eb_beta_binomial <- function(hist) {
  k <- tapply(hist$k, hist$playerID, sum)
  n <- tapply(hist$n, hist$playerID, sum)
  nll <- function(par) {
    a <- exp(par[1]); b <- exp(par[2])
    -sum(lbeta(k + a, n - k + b) - lbeta(a, b))
  }
  m <- sum(k) / sum(n)
  fit <- optim(c(log(m * 100), log((1 - m) * 100)), nll)
  a <- exp(fit$par[1]); b <- exp(fit$par[2])
  list(proj = (k + a) / (n + a + b), prior_strength = a + b)
}

rows <- list()
for (f in list.files(in_dir, pattern = "_target.csv$", full.names = TRUE)) {
  tgt <- read.csv(f)
  hist <- read.csv(sub("_target.csv$", "_history.csv", f))
  parts <- strsplit(basename(f), "_")[[1]]
  stat <- paste(parts[1], parts[2], sep = "_")
  year <- as.integer(parts[3])
  m <- marcel(hist, year)[tgt$playerID]
  eb <- eb_beta_binomial(hist)
  pe <- eb$proj[tgt$playerID]
  lg <- rep(hist$league_rate[1], nrow(tgt))
  rows[[length(rows) + 1]] <- data.frame(
    stat = stat, year = year, players = nrow(tgt),
    marcel_max_diff_vs_python = max(abs(m - tgt$marcel_py)),
    league_logloss = log_loss(lg, tgt$k, tgt$n),
    marcel_logloss = log_loss(m, tgt$k, tgt$n),
    eb_logloss = log_loss(pe, tgt$k, tgt$n),
    marcel_rmse = wrmse(m, tgt$k, tgt$n),
    eb_rmse = wrmse(pe, tgt$k, tgt$n),
    marcel_brier = brier(m, tgt$k, tgt$n),
    eb_prior_strength = eb$prior_strength
  )
}
out <- do.call(rbind, rows)
out <- out[order(out$stat, out$year), ]
stopifnot(all(out$marcel_max_diff_vs_python < 1e-9))
write.csv(out, "results/r_replication.csv", row.names = FALSE)
print(out, digits = 5)
