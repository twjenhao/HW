###########資料整理###########
file_path <- "/Users/liurenhao/Documents/TaipeiuniversityNote/QuantitiveMethod/2022HW/for_R_Labor Supply Data From Mroz (1987).txt"
LaborSupplyData <- read.table(file_path, header = TRUE, stringsAsFactors = FALSE)
dat <- LaborSupplyData[LaborSupplyData$LFP != 0, ]
class(dat)
library(dplyr)
dat %>% mutate(dat,K18=KL6+K618) -> dat
dat$DK18 <- ifelse(dat$K18 > 0, 1, 0)
dat %>% mutate(WA2=WA*WA) -> dat
dat %>% mutate(Earnings=WHRS*WW) -> dat
# View(dat)
dat1 <- dat

###########課本方法###########
###########回歸###########
Earnings <- dat$Earnings
wages <- dat$WW
Age <- dat$WA
Age2 <- dat$WA2
Kids <- dat$DK18
Education <- dat1$WE
lm <-lm(log(Earnings) ~Age+Age2+Education+Kids)
summary(lm)
sink("regression_summary.txt")
summary(lm)
sink()
############Covariance Matrix###########
y <- as.matrix(log(Earnings))
X <- cbind(1,Age,Age2,Education,Kids)
b <- crossprod(solve(crossprod(X,X)),crossprod(X,y))
e <- y-X%*%b
n <- nrow(X)
i <- as.matrix(X[,1])
m0 <- diag(n)-i%*%solve(crossprod(i,i))%*%t(i)
rss <- crossprod(e,e)
k <- ncol(X)
nmk <- n-k
s2 <- as.numeric(rss/nmk)
s <- s2^0.5
VCOV <- s2*solve(crossprod(X,X))
sink("Covariance Matrix.txt")
VCOV
sink()
###########課本方法(男性)###########
###########資料整理###########
LaborSupplyData -> dat2
dat2 %>% mutate(dat2,K18=KL6+K618) -> dat2
dat2$DK18 <- ifelse(dat2$K18 > 0, 1, 0)
dat2 %>% mutate(HA2=HA*HA) -> dat2
dat2 %>% mutate(HEarnings=HHRS*HW) -> dat2
###########回歸###########
HEarnings <- dat2$HEarnings
Hwages <- dat2$HW
HAge <- dat2$HA
HAge2 <- dat2$HA2
HKids <- dat2$DK18
HEducation <- dat2$HE
Hlm <-lm(log(HEarnings) ~HAge+HAge2+HEducation+HKids)
summary(Hlm)







