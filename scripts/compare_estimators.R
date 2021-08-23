install.packages("ScottKnottESD")
library(ScottKnottESD)

help(sk_esd)
sk <- sk_esd(estimators_performance, version="np")
plot(sk)
