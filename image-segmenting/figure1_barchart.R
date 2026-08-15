# figure_single_tubes.R
# ----------------------------------------------------------
# Plots one positive tube and one negative tube from ImageJ.
# No statistics or error bars (n = 1 each).
#
# CSV format:
#
# group,value
# Positive,108.494
# Negative,97.548
#
# Usage:
# Rscript figure_single_tubes.R imagej_data.csv single_tubes.png
# ----------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

csv <- if(length(args) >= 1) args[1] else "imagej_data.csv"
out <- if(length(args) >= 2) args[2] else "single_tubes.png"

df <- read.csv(csv, stringsAsFactors = FALSE)

names(df) <- tolower(names(df))
colnames(df)[colnames(df)=="value"] <- "signal"

df$group <- factor(df$group,
                   levels=c("Positive","Negative"))

cols <- c("#D9531E","#2E75C6")

ymax <- max(df$signal)*1.20

png(out,
    width=700,
    height=700,
    res=150)

par(mar = c(5, 5, 4, 1))

bp <- barplot(df$signal,
              names.arg=df$group,
              col=cols,
              border=NA,
              ylim=c(0,ymax),
              ylab="Inverted Signal Mean Value",
              main = "Positive vs. Negative Control\nwith Inverted Signal",
              cex.main = 1.2)

## Print exact values above each bar
text(bp,
     df$signal + ymax*0.03,
     labels=round(df$signal,2),
     cex=1)

dev.off()

cat("Saved:",out,"\n")