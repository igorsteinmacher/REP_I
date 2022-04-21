library(ScottKnottESD)

par(mar=c(5, 7, 4, 10))
par(cex.axis=1.5)

BW <- read.csv("~/GitHub/USP-2020/results/feature_analysis/BW.csv", row.names=1)
sk_BW <- sk_esd(BW)
pdf(file = "~/GitHub/USP-2020/results/feature_analysis/BW.pdf", width=10, height=5)
mtext("Means", side=2, line=3.5, cex=2)
mtext("Top features grouped by color", side=1, line=3.5, cex=2)

# CF - Contribution Flow
CF <- read.csv("~/GitHub/USP-2020/results/feature_analysis/CF.csv", row.names=1)
sk_CF <- sk_esd(CF)
plot(sk_CF, col=c("#0F9D58", "#4285F4", "#F4B400", "#DB4437", "#AE00FF"), lwd=2, pch = 16, ylab="", xlab="", title="", cex=1.5)
mtext("Means", side=2, line=3.5, cex=2)
mtext("Top features grouped by color", side=1, line=3.5, cex=2)

# CT - Choose the task
CT <- read.csv("~/GitHub/USP-2020/results/feature_analysis/CT.csv", row.names=1)
sk_CT <- sk_esd(CT)
plot(sk_CT, col=c("#0F9D58", "#4285F4", "#F4B400", "#DB4437", "#AE00FF"), lwd=2, pch = 16, ylab="", xlab="", title="", cex=1.5)
mtext("Means", side=2, line=3.5, cex=2)
mtext("Top features grouped by color", side=1, line=3.5, cex=2)

# TC - Talk to the community
TC <- read.csv("~/GitHub/USP-2020/results/feature_analysis/TC.csv", row.names=1)
sk_TC <- sk_esd(TC)
plot(sk_TC, col=c("#0F9D58", "#4285F4", "#F4B400", "#DB4437", "#AE00FF"), lwd=2, pch = 16, ylab="", xlab="", title="", cex=1.5)
mtext("Means", side=2, line=3.5, cex=2)
mtext("Top features grouped by color", side=1, line=3.5, cex=2)

# DC - Deal with the code
DC <- read.csv("~/GitHub/USP-2020/results/feature_analysis/DC.csv", row.names=1)
sk_DC <- sk_esd(DC)
plot(sk_DC, col=c("#0F9D58", "#4285F4", "#F4B400", "#DB4437", "#AE00FF"), lwd=2, pch = 16, ylab="", xlab="", title="", cex=1.5)
mtext("Means", side=2, line=3.5, cex=2)
mtext("Top features grouped by color", side=1, line=3.5, cex=2)

# NC - No categories identified
NC <- read.csv("~/GitHub/USP-2020/results/feature_analysis/NC.csv", row.names=1)
sk_NC <- sk_esd(NC)
plot(sk_NC, col=c("#0F9D58", "#4285F4", "#F4B400", "#DB4437", "#AE00FF"), lwd=2, pch = 16, ylab="", xlab="", title="", cex=1.5)
mtext("Means", side=2, line=3.5, cex=2)
mtext("Top features grouped by color", side=1, line=3.5, cex=2)

# SC - Submit the changes
SC <- read.csv("~/GitHub/USP-2020/results/feature_analysis/SC.csv", row.names=1)
sk_SC <- sk_esd(SC)
plot(sk_SC, col=c("#0F9D58", "#4285F4", "#F4B400", "#DB4437", "#AE00FF"), lwd=2, pch = 16, ylab="", xlab="", title="", cex=1.5)
mtext("Means", side=2, line=3.5, cex=2)
mtext("Top features grouped by color", side=1, line=3.5, cex=2)

