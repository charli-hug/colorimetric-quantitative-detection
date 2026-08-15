#  sixpanel_tiers.R
#  -----------------------------------------------------------------------------
#  Six-panel box plot: each colour channel (a, red_dominance, L, inv_mgv, b, S)
#  split by concentration TIER (high / low / negative). Shows why a* separates
#  the tiers best. Reads the ingest concentrations file directly.
#
#  Tiers:  high = molar >= 1e-7 (100 nM);  low = 0 < molar < 1e-7 (100 pM-10 nM);
#          negative = molar == 0.
#
#  Usage:
#    Rscript sixpanel_tiers.R measurements.csv concentrations.csv [outdir]
# -----------------------------------------------------------------------------

args   <- commandArgs(trailingOnly = TRUE)
mcsv   <- if (length(args) >= 1) args[1] else "measurements.csv"
ccsv   <- if (length(args) >= 2) args[2] else "concentrations.csv"
outdir <- if (length(args) >= 3) args[3] else "."
if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

m <- read.csv(mcsv, stringsAsFactors = FALSE)
c <- read.csv(ccsv, stringsAsFactors = FALSE)
m$key <- basename(m$image); c$key <- basename(c$image)
df <- merge(m, c, by = "key")
if (nrow(df) == 0) stop("No rows matched between measurements and concentrations.")

df$molar <- suppressWarnings(as.numeric(df$concentration_molar))
df <- df[!is.na(df$molar), ]              # drop blank/unparsed concentrations
df$tier <- ifelse(df$molar == 0, "negative",
           ifelse(df$molar >= 1e-7, "high", "low"))
df$tier <- factor(df$tier, levels = c("negative", "low", "high"))

cat(sprintf("Matched %d tubes: %d negative, %d low, %d high\n\n",
            nrow(df), sum(df$tier=="negative"), sum(df$tier=="low"), sum(df$tier=="high")))

channels <- c("a", "red_dominance", "L", "inv_mgv", "b", "S")
channels <- channels[channels %in% names(df)]
nice <- c(a="CIELAB a* (green-red)", red_dominance="Red dominance R-(G+B)/2",
          L="CIELAB L* (lightness)", inv_mgv="Inverted MGV (grayscale)",
          b="CIELAB b* (blue-yellow)", S="HSV saturation")

# one-way ANOVA per channel (how strongly does the channel separate the tiers?)
cat("Per-channel tier separation (one-way ANOVA):\n")
pvals <- c()
for (ch in channels) {
  p <- tryCatch(summary(aov(df[[ch]] ~ df$tier))[[1]][["Pr(>F)"]][1],
                error = function(e) NA)
  pvals[ch] <- p
  cat(sprintf("  %-14s ANOVA p = %.2e\n", ch, p))
}
best <- names(which.min(pvals))
cat(sprintf("\nStrongest tier separation: %s (smallest ANOVA p)\n", best))

# ---- six-panel figure -----------------------------------------------------
tier_cols <- c(negative="#2471A3", low="#7F77DD", high="#C0392B")
png(file.path(outdir, "sixpanel_tiers.png"), width = 1600, height = 1050, res = 130)
op <- par(mfrow = c(2, 3), mar = c(3.5, 4.2, 3, 1), oma = c(0, 0, 2.5, 0))
for (ch in channels) {
  boxplot(df[[ch]] ~ df$tier, col = adjustcolor(tier_cols, 0.35),
          border = tier_cols, outline = FALSE,
          xlab = "", ylab = ch, main = sprintf("%s\n(ANOVA p = %.1e)", nice[[ch]], pvals[ch]),
          cex.main = 0.95, ylim = range(df[[ch]], na.rm = TRUE))
  set.seed(1)
  for (i in seq_along(levels(df$tier))) {
    lev <- levels(df$tier)[i]
    v <- df[[ch]][df$tier == lev]
    points(jitter(rep(i, length(v)), 0.7), v, pch = 19,
           col = tier_cols[lev], cex = 0.7)
  }
}
mtext("Colour channels by concentration tier — a* separates tiers most cleanly",
      outer = TRUE, cex = 1.05, font = 2, line = 0.3)
par(op); dev.off()

cat("\nSaved: ", file.path(normalizePath(outdir), "sixpanel_tiers.png"), "\n", sep = "")
