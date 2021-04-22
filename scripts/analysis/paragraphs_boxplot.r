paragraphs <- read.table("~/GitHub/usp/scripts/analysis/paragraphs.csv", quote="\"")
summary(paragraphs)
boxplot(paragraphs, horizontal=TRUE,outline=FALSE, lwd=3, cex.axis=2, cex.lab=2, xlab="# Paragraphs", width=0.2)

