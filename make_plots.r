#  make_plots.R  (single-tube version)
#  -----------------------------------------------------------------------------
#  Base-R graphs from the pipeline's measurements.csv when there is ONE object
#  per image (the clinical single-tube case). No ggplot2.
#
#  No concentration labels yet, so these show: (1) each colour channel's value
#  per image, and (2) how the channels co-vary -- which is what tells you which
#  channel best separates samples and how stable it is across lighting.
#
#  Usage:
#    Rscript make_plots.R measurements.csv            # writes PNGs to cwd
#    Rscript make_plots.R measurements.csv outdir     # writes them to outdir
# -----------------------------------------------------------------------------

args   <- commandArgs(trailingOnly = TRUE)
csv    <- if (length(args) >= 1) args[1] else "measurements.csv"
outdir <- if (length(args) >= 2) args[2] else "."
if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

df <- read.csv(csv, stringsAsFactors = FALSE)
df <- df[order(df$image), ]
n  <- nrow(df)
labs <- sub("\\.(jpe?g|png)$", "", df$image)

# ---------------------------------------------------------------------------
# FIGURE 1: every channel's value across the images (sorted by a*).
# Sorting by a* lets you see whether the other channels track the redness order.
# ---------------------------------------------------------------------------
ord  <- order(df$a)
d    <- df[ord, ]
dl   <- labs[ord]

png(file.path(outdir, "fig1_channels_by_image.png"),
    width = 1500, height = 1000, res = 130)
op <- par(mfrow = c(2, 2), mar = c(7, 4.5, 3, 1), oma = c(0, 0, 3, 0))

panels <- list(
  c("a",             "CIE a* (green-red)",        "a*"),
  c("red_dominance", "Red dominance R-(G+B)/2",   "red dom."),
  c("L",             "CIE L* (lightness)",        "L*"),
  c("inv_mgv",       "Inverted MGV (paper)",      "inv MGV")
)
for (p in panels) {
  col <- p[1]
  barplot(d[[col]], names.arg = dl, las = 2, cex.names = 0.6,
          col = "#7F77DD", border = NA,
          main = p[2], ylab = p[3], cex.main = 1.0)
  abline(h = 0, col = "gray40", lwd = 0.6)
}
mtext("Channel values across images (sorted by a*)",
      outer = TRUE, cex = 1.1, font = 2)
par(op); dev.off()

# ---------------------------------------------------------------------------
# FIGURE 2: a* vs the other channels -- scatter, to see which co-vary with a*.
# (a* is the lighting-stable redness axis; this shows what tracks it.)
# ---------------------------------------------------------------------------
png(file.path(outdir, "fig2_a_correlations.png"),
    width = 1500, height = 500, res = 130)
op <- par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3, 1))
pairs_to_plot <- list(
  c("red_dominance", "red dominance"),
  c("L", "L*"),
  c("inv_mgv", "inverted MGV")
)
for (pr in pairs_to_plot) {
  col <- pr[1]
  plot(df$a, df[[col]], pch = 19, col = "#1D9E75",
       xlab = "a*", ylab = pr[2],
       main = sprintf("a* vs %s  (r = %.2f)", pr[2], cor(df$a, df[[col]])))
  grid(col = "gray85")
  abline(lm(df[[col]] ~ df$a), col = "#D85A30", lwd = 1.5)
}
par(op); dev.off()

# ---------------------------------------------------------------------------
# FIGURE 3: ROI size sanity check -- flags images whose ROI was too small
# (a small ROI means the pool wasn't captured well, so trust its colour less).
# ---------------------------------------------------------------------------
png(file.path(outdir, "fig3_roi_size.png"),
    width = 1500, height = 500, res = 130)
op <- par(mar = c(7, 4.5, 3, 1))
cols <- ifelse(df$reliable %in% c(TRUE, "True", "true"), "#1D9E75", "#D85A30")
barplot(df$roi_px, names.arg = labs, las = 2, cex.names = 0.6,
        col = cols, border = NA,
        main = "ROI pixel count per image (orange = flagged unreliable)",
        ylab = "ROI pixels")
par(op); dev.off()

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
cat(sprintf("Images: %d\n\n", n))
cat("Channel spread across images (max - min):\n")
for (col in c("a","b","L","red_dominance","inv_mgv","S")) {
  if (col %in% names(df)) {
    v <- df[[col]]
    cat(sprintf("  %-14s: %7.2f .. %7.2f   spread = %6.2f\n",
                col, min(v), max(v), max(v)-min(v)))
  }
}
cat("\nCorrelation of each channel with a*:\n")
for (col in c("red_dominance","L","inv_mgv","b","S")) {
  if (col %in% names(df)) cat(sprintf("  %-14s: r = %+.2f\n", col, cor(df$a, df[[col]])))
}
small <- df[df$roi_px < quantile(df$roi_px, 0.15), c("image","roi_px")]
if (nrow(small)) {
  cat("\nSmallest ROIs (check these figures -- pool may be under-captured):\n")
  for (i in seq_len(nrow(small))) cat(sprintf("  %-18s roi_px = %d\n", small$image[i], small$roi_px[i]))
}
cat("\nSaved 3 figures to: ", normalizePath(outdir), "\n", sep = "")