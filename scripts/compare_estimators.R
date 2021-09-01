install.packages("ScottKnottESD")
library(ScottKnottESD)
library(readxl)
performances <- read_excel("GitHub/nemenyi/performances.xlsx")

sk <- sk_esd(performances, version="np")
plot(sk)
