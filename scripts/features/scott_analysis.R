library(ScottKnottESD)

par(mar=c(12, 10, 1, 10))
par(cex.axis=3.5, lwd=2)

# Size 12 x 7

BW <- read.csv("~/GitHub/ICSE-2023/results/feature_analysis/BW.csv", row.names=1)
sk_BW <- sk_esd(BW)
plot(sk_BW, col=c("#0F9D58", "#4285F4", "#a36a00", "#DB4437", "#AE00FF"), pch = 16, ylab="", xlab="", title="", cex=4.5, lwd = 9, padj=1.7, las=1)
mtext("Means", side=2, line=3.5, cex=4.5, padj=-1)
mtext("Top features grouped by color", side=1, line=3.5, cex=4.5, padj=2.5)

# CF - Contribution Flow
CF <- read.csv("~/GitHub/ICSE-2023/results/feature_analysis/CF.csv", row.names=1)
sk_CF <- sk_esd(CF)
plot(sk_CF, col=c("#0F9D58", "#4285F4", "#a36a00", "#DB4437", "#AE00FF"), pch = 16, ylab="", xlab="", title="", cex=3, lwd = 9, padj=1.7)
mtext("Means", side=2, line=3.5, cex=4.5, padj=-1)
mtext("Top features grouped by color", side=1, line=3.5, cex=4.5, padj=2.5)

# CT - Choose the task
CT <- read.csv("~/GitHub/ICSE-2023/results/feature_analysis/CT.csv", row.names=1)
sk_CT <- sk_esd(CT)
plot(sk_CT, col=c("#0F9D58", "#4285F4", "#a36a00", "#DB4437", "#AE00FF"), pch = 16, ylab="", xlab="", title="", cex=3.5, lwd = 9, padj=1.3)
mtext("Means", side=2, line=3.5, cex=3.5)
mtext("Top features grouped by color", side=1, line=3.5, cex=3.5, padj=1.7)

# TC - Talk to the community
TC <- read.csv("~/GitHub/ICSE-2023/results/feature_analysis/TC.csv", row.names=1)
sk_TC <- sk_esd(TC)
plot(sk_TC, col=c("#0F9D58", "#4285F4", "#a36a00", "#DB4437", "#AE00FF"), pch = 16, ylab="", xlab="", title="", cex=3.5, lwd = 9, padj=1.3)
mtext("Means", side=2, line=3.5, cex=3.5)
mtext("Top features grouped by color", side=1, line=3.5, cex=3.5, padj=1.7)

# DC - Deal with the code
DC <- read.csv("~/GitHub/ICSE-2023/results/feature_analysis/DC.csv", row.names=1)
sk_DC <- sk_esd(DC)
plot(sk_DC, col=c("#0F9D58", "#4285F4", "#a36a00", "#DB4437", "#AE00FF"), pch = 16, ylab="", xlab="", title="", cex=3.5, lwd = 9, padj=1.3)
mtext("Means", side=2, line=3.5, cex=3.5)
mtext("Top features grouped by color", side=1, line=3.5, cex=3.5, padj=1.7)

# NC - No categories identified
NC <- read.csv("~/GitHub/ICSE-2023/results/feature_analysis/NC.csv", row.names=1)
sk_NC <- sk_esd(NC)
plot(sk_NC, col=c("#0F9D58", "#4285F4", "#a36a00", "#DB4437", "#AE00FF"), pch = 16, ylab="", xlab="", title="", cex=3.5, lwd = 9, padj=1.3)
mtext("Means", side=2, line=3.5, cex=3.5)
mtext("Top features grouped by color", side=1, line=3.5, cex=3.5, padj=1.7)

# SC - Submit the changes
SC <- read.csv("~/GitHub/ICSE-2023/results/feature_analysis/SC.csv", row.names=1)
sk_SC <- sk_esd(SC)
plot(sk_SC, col=c("#0F9D58", "#4285F4", "#a36a00", "#DB4437", "#AE00FF"), pch = 16, ylab="", xlab="", title="", cex=3.5, lwd = 9, padj=1.3)
mtext("Means", side=2, line=3.5, cex=3.5)
mtext("Top features grouped by color", side=1, line=3.5, cex=3.5, padj=1.7)
