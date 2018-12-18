

usage <- as.data.frame(X2356304_usage_data_excluding_dru_and_internal_users)

dim(usage)

head(usage)

usage$Username <- as.factor(sub('.*@', '', usage$Username))
usage$Number_of_Accounts <- 1

usage <- usage[,c(1,5,6,7)]

head(usage)

all <- aggregate(. ~ Username, data = usage, sum)

all$School <- c("app.state", "colorado", "harvard",
                "harvard", "harvard", "northwestern", 
                "harvard", "harvard", "harvard", 
                "bi.no", "harvard", "nus", "ucsb")

all2 <- all[,c(5,2,3,4)]

aggregate(. ~ School, data = all2, sum)

