#  analyze_labeled.R
#  -----------------------------------------------------------------------------
#  Joins measurements.csv to labels.csv (positive/negative) and tests how well
#  each colour channel separates the two classes. For a binary 100nM vs 0nM
#  design this is the real question: which channel classifies correctly, and at
#  what threshold.
#
#  Outputs:
#    - console: per-channel class means, separation, best threshold + accuracy
#    - fig_class_separation.png : strip/box plot of each channel by class
#    - fig_best_channel_roc.png : threshold sweep for the best channel
#
#  Usage:
#    Rscript analyze_labeled.R measurements.csv labels.csv [outdir]
# -----------------------------------------------------------------------------

args   <- commandArgs(trailingOnly = TRUE)
mcsv   <- if (length(args) >= 1) args[1] else "measurements.csv"
lcsv   <- if (length(args) >= 2) args[2] else "labels.csv"
outdir <- if (length(args) >= 3) args[3] else "."
if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

m <- read.csv(mcsv, stringsAsFactors = FALSE)
l <- read.csv(lcsv, stringsAsFactors = FALSE)

# normalise image keys (strip path, lowercase ext differences)
m$key <- basename(m$image)
l$key <- basename(l$image)
df <- merge(m, l, by = "key")
if (nrow(df) == 0) stop("No rows matched between measurements and labels -- check image-name formats.")

df$label <- tolower(df$label)
cat(sprintf("Matched %d images: %d positive, %d negative\n\n",
            nrow(df), sum(df$label == "positive"), sum(df$label == "negative")))

channels <- c("a", "red_dominance", "L", "inv_mgv", "b", "S")
channels <- channels[channels %in% names(df)]

# ---- per-channel separation + best threshold ------------------------------
# For each channel, sweep every midpoint between sorted values as a threshold,
# pick the one that maximises balanced accuracy. Records direction too.
best_acc <- function(x, y) {            # y = 1 for positive, 0 for negative
  cand <- sort(unique(x))
  ths  <- c(cand[1] - 1, (head(cand, -1) + tail(cand, -1)) / 2, tail(cand, 1) + 1)
  best <- list(acc = 0, th = NA, dir = NA)
  for (th in ths) {
    for (dir in c("greater", "less")) {
      pred <- if (dir == "greater") x >= th else x <= th
      acc  <- mean(pred == (y == 1))
      if (acc > best$acc) best <- list(acc = acc, th = th, dir = dir)
    }
  }
  best
}

y <- as.integer(df$label == "positive")
cat("Channel separation (positive vs negative):\n")
cat(sprintf("  %-14s %10s %10s %9s  %s\n", "channel", "pos mean", "neg mean", "best acc", "rule"))
results <- list()
for (ch in channels) {
  x <- df[[ch]]
  pm <- mean(x[y == 1]); nm <- mean(x[y == 0])
  b  <- best_acc(x, y)
  results[[ch]] <- c(acc = b$acc, th = b$th, posmean = pm, negmean = nm)
  rule <- sprintf("%s %s %.2f -> positive", ch, if (b$dir=="greater") ">=" else "<=", b$th)
  cat(sprintf("  %-14s %10.2f %10.2f %8.0f%%  %s\n", ch, pm, nm, 100*b$acc, rule))
}

best_ch <- names(which.max(sapply(results, function(r) r["acc"])))
cat(sprintf("\nBest single channel: %s (%.0f%% accuracy)\n",
            best_ch, 100 * results[[best_ch]]["acc"]))

# ---- report which images were misclassified by the best channel -----------
bx  <- df[[best_ch]]
bth <- as.numeric(results[[best_ch]]["th"])
# direction: are positives higher than negatives on this channel?
dir_greater <- mean(bx[y == 1]) >= mean(bx[y == 0])
pred_pos <- if (isTRUE(dir_greater)) bx >= bth else bx <= bth
wrong <- which(pred_pos != (y == 1))
if (length(wrong) == 0) {
  cat("All images classified correctly by", best_ch, "\n")
} else {
  cat(sprintf("\nMisclassified by %s (%d):\n", best_ch, length(wrong)))
  for (i in wrong) {
    cat(sprintf("  %-18s true=%s  %s=%.2f  roi_px=%s\n",
                df$key[i], df$label[i], best_ch, bx[i],
                if ("roi_px" %in% names(df)) as.character(df$roi_px[i]) else "NA"))
  }
}

# ---- FIGURE 1: each channel by class --------------------------------------
png(file.path(outdir, "fig_class_separation.png"),
    width = 1500, height = 950, res = 130)
op <- par(mfrow = c(2, 3), mar = c(4, 4.5, 3, 1))
for (ch in channels) {
  pos <- df[[ch]][y == 1]; neg <- df[[ch]][y == 0]
  boxplot(list(negative = neg, positive = pos),
          col = c("#B5D4F4", "#F4C0D1"), border = c("#185FA5", "#993556"),
          main = sprintf("%s  (%.0f%% acc)", ch, 100*results[[ch]]["acc"]),
          ylab = ch, outline = FALSE, ylim = range(df[[ch]]))
  set.seed(1)
  points(jitter(rep(1, length(neg)), 0.6), neg, pch = 19, col = "#185FA5", cex = 0.8)
  points(jitter(rep(2, length(pos)), 0.6), pos, pch = 19, col = "#993556", cex = 0.8)
  abline(h = results[[ch]]["th"], lty = 2, col = "gray40")
}
mtext("Colour channels by class (dashed line = best threshold)",
      outer = TRUE, cex = 1.1, font = 2, line = -1.2)
par(op); dev.off()

# ---- FIGURE 2: threshold sweep for the best channel -----------------------
x <- df[[best_ch]]
x <- x[is.finite(x)]
if (length(unique(x)) >= 2) {
  ths <- seq(min(x), max(x), length.out = 100)
  acc <- sapply(ths, function(th)
    max(mean((x >= th) == (y == 1)), mean((x <= th) == (y == 1))))
  png(file.path(outdir, "fig_best_channel_threshold.png"),
      width = 1100, height = 600, res = 130)
  op <- par(mar = c(4.5, 4.5, 3, 1))
  plot(ths, 100 * acc, type = "l", lwd = 2, col = "#1D9E75",
       xlab = best_ch, ylab = "classification accuracy (%)",
       main = sprintf("Threshold sweep: %s", best_ch))
  grid(col = "gray85")
  abline(v = results[[best_ch]]["th"], lty = 2, col = "#D85A30")
  par(op); dev.off()
} else {
  cat("(threshold-sweep figure skipped: best channel has no spread)\n")
}

cat("\nSaved figures to: ", normalizePath(outdir), "\n", sep = "")