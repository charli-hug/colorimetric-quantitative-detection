#  stress_test.R
#  -----------------------------------------------------------------------------
#  Pressure-tests colorspace robustness by applying SIMULATED lighting transforms
#  to the measured per-tube RGB, recomputing each channel, and measuring how
#  classification accuracy holds up -- RAW vs BACKGROUND-NORMALIZED.
#
#  Hypothesis (from the project): a* is robust to lighting; inv_mgv is not.
#  Background normalization should make a SAME concentration read the SAME a*
#  across lighting (collapse within-class lighting variance) WITHOUT erasing the
#  between-class signal.
#
#  Transforms applied to RGB:
#    brightness : multiply R,G,B by k  (exposure / lighting intensity)
#    warmth     : multiply R by k, B by 1/k (colour temperature / white balance)
#
#  Requires the CSV to contain R,G,B and bg_R,bg_G,bg_B (background illuminant).
#
#  Usage:
#    Rscript stress_test.R measurements.csv labels.csv [outdir]
# -----------------------------------------------------------------------------

args   <- commandArgs(trailingOnly = TRUE)
mcsv   <- if (length(args) >= 1) args[1] else "measurements.csv"
lcsv   <- if (length(args) >= 2) args[2] else "labels.csv"
outdir <- if (length(args) >= 3) args[3] else "."
if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

m <- read.csv(mcsv, stringsAsFactors = FALSE)
l <- read.csv(lcsv, stringsAsFactors = FALSE)
m$key <- basename(m$image); l$key <- basename(l$image)
df <- merge(m, l, by = "key")

# Accept either a binary 'label' column (positive/negative) OR a concentration
# file (concentration_molar). For the robustness test we need a binary split:
# any target present (molar > 0) = positive, zero/control = negative.
if ("label" %in% names(df)) {
  df$label <- tolower(df$label)
  y <- as.integer(df$label == "positive")
} else if ("concentration_molar" %in% names(df)) {
  molar <- suppressWarnings(as.numeric(df$concentration_molar))
  # drop rows with no parseable concentration (blank captions etc.)
  keep <- !is.na(molar)
  df <- df[keep, ]; molar <- molar[keep]
  y <- as.integer(molar > 0)
  cat("Using concentration file: molar>0 -> positive, molar==0 -> negative.\n")
} else {
  stop("Label file needs either a 'label' or 'concentration_molar' column.")
}
if (nrow(df) == 0) stop("No matched rows.")
has_bg <- all(c("bg_R","bg_G","bg_B") %in% names(df)) && !any(is.na(df$bg_R))
cat(sprintf("Matched %d images (%d pos, %d neg). Background columns present: %s\n\n",
            nrow(df), sum(y), sum(1-y), has_bg))

# ---- colour-space conversions from RGB (0-255) ----------------------------
# sRGB -> CIE Lab (D65). Vectorised over rows.
rgb2lab <- function(R, G, B) {
  f <- function(t) ifelse(t > 0.04045, ((t + 0.055)/1.055)^2.4, t/12.92)
  r <- f(R/255); g <- f(G/255); b <- f(B/255)
  X <- r*0.4124 + g*0.3576 + b*0.1805
  Y <- r*0.2126 + g*0.7152 + b*0.0722
  Z <- r*0.0193 + g*0.1192 + b*0.9505
  Xn <- 0.95047; Yn <- 1.0; Zn <- 1.08883
  fx <- function(t) ifelse(t > 0.008856, t^(1/3), 7.787*t + 16/116)
  L <- 116*fx(Y/Yn) - 16
  a <- 500*(fx(X/Xn) - fx(Y/Yn))
  bb <- 200*(fx(Y/Yn) - fx(Z/Zn))
  data.frame(L = L, a = a, b = bb)
}
channels_from_rgb <- function(R, G, B) {
  lab <- rgb2lab(R, G, B)
  gray <- 0.299*R + 0.587*G + 0.114*B
  data.frame(
    a = lab$a, L = lab$L, b = lab$b,
    red_dominance = R - (G + B)/2,
    inv_mgv = 255 - gray
  )
}

# best balanced single-threshold accuracy for a vector x vs labels y
best_acc <- function(x, y) {
  if (length(unique(x)) < 2) return(0.5)
  cand <- sort(unique(x)); ths <- (head(cand,-1)+tail(cand,-1))/2
  best <- 0.5
  for (th in ths) best <- max(best, mean((x>=th)==(y==1)), mean((x<=th)==(y==1)))
  best
}

# ---- transform sweep ------------------------------------------------------
ks <- seq(0.6, 1.4, by = 0.1)        # transform intensities
chans <- c("a","red_dominance","L","inv_mgv")
pal <- c(a="#1D9E75", red_dominance="#7F77DD", L="#E08A2B", inv_mgv="#D85A30")

run_sweep <- function(kind, normalize) {
  out <- matrix(NA, length(ks), length(chans), dimnames = list(NULL, chans))
  for (i in seq_along(ks)) {
    k <- ks[i]
    R <- df$R; G <- df$G; B <- df$B
    bR <- df$bg_R; bG <- df$bg_G; bB <- df$bg_B
    if (kind == "brightness") { R<-R*k; G<-G*k; B<-B*k; bR<-bR*k; bG<-bG*k; bB<-bB*k }
    if (kind == "warmth")     { R<-R*k; B<-B/k; bR<-bR*k; bB<-bB/k }
    if (normalize && has_bg) {     # divide tube RGB by background illuminant, rescale to mid-gray
      R <- R/bR*128; G <- G/bG*128; B <- B/bB*128
    }
    R<-pmin(pmax(R,0),255); G<-pmin(pmax(G,0),255); B<-pmin(pmax(B,0),255)
    ch <- channels_from_rgb(R, G, B)
    for (cn in chans) out[i, cn] <- best_acc(ch[[cn]], y)
  }
  out
}

plot_sweep <- function(kind, normalize, fname, subtitle) {
  S <- run_sweep(kind, normalize)
  png(file.path(outdir, fname), width = 1000, height = 650, res = 130)
  op <- par(mar = c(4.5, 4.5, 4, 1))
  plot(NA, xlim = range(ks), ylim = c(0.45, 1.0),
       xlab = sprintf("%s transform factor (1.0 = original)", kind),
       ylab = "classification accuracy", main = subtitle)
  grid(col = "gray85"); abline(v = 1.0, lty = 3, col = "gray50")
  for (cn in chans) { lines(ks, S[,cn], col = pal[cn], lwd = 2); points(ks, S[,cn], col = pal[cn], pch = 19, cex = 0.8) }
  legend("bottomleft", legend = chans, col = pal[chans], lwd = 2, pch = 19, cex = 0.8, bg = "white")
  par(op); dev.off()
}

# RAW (no normalization): shows which channels break under the transform
plot_sweep("brightness", FALSE, "stress_brightness_raw.png",
           "Brightness robustness (raw)")
plot_sweep("warmth", FALSE, "stress_warmth_raw.png",
           "White-balance robustness (raw)")

# NORMALIZED: shows background normalization restoring robustness
if (has_bg) {
  plot_sweep("brightness", TRUE, "stress_brightness_normalized.png",
             "Brightness robustness (background-normalized)")
  plot_sweep("warmth", TRUE, "stress_warmth_normalized.png",
             "White-balance robustness (background-normalized)")
}

# ---- within-class variance: does normalization collapse lighting noise? ----
# For the BEST channel (a*), compare the spread of values WITHIN each class
# before and after normalization. Lower within-class spread = lighting removed.
within_sd <- function(vals) mean(c(sd(vals[y==1]), sd(vals[y==0])))
raw_a <- channels_from_rgb(df$R, df$G, df$B)$a
if (has_bg) {
  nR <- df$R/df$bg_R*128; nG <- df$G/df$bg_G*128; nB <- df$B/df$bg_B*128
  nrm_a <- channels_from_rgb(pmin(pmax(nR,0),255), pmin(pmax(nG,0),255), pmin(pmax(nB,0),255))$a
  cat("Within-class a* spread (mean of per-class SD):\n")
  cat(sprintf("  raw         : %.3f\n", within_sd(raw_a)))
  cat(sprintf("  normalized  : %.3f   (lower = lighting noise removed)\n", within_sd(nrm_a)))
  cat(sprintf("\nBetween-class a* gap (pos mean - neg mean):\n"))
  cat(sprintf("  raw         : %.3f\n", mean(raw_a[y==1]) - mean(raw_a[y==0])))
  cat(sprintf("  normalized  : %.3f   (should stay large = signal preserved)\n",
              mean(nrm_a[y==1]) - mean(nrm_a[y==0])))
}

cat("\nSaved stress-test figures to: ", normalizePath(outdir), "\n", sep = "")