require(data.table)
require(QuantPsyc)
library(lmSupport)

flickrcar <- fread("D:/Flickr/flickrexport_cars_oid_coded.csv", sep=",", header=TRUE)

flickr_groups_cars <- read.csv("C:/Users/jwulf/OneDrive/Research/IWI Forschung/Flickr/flickr_groups_cars.csv", sep=";")
flickrcar[flickr_groups_cars, on = 'group_url', Class:=i.Class]
nrow(flickrcar)
flickrcar_wo_country_1<- flickrcar[Class!='country',]
nrow(flickrcar_wo_country_1)
rm(flickrcar)

#separate oids and rest
cols <- grep("oid.", names(flickrcar_wo_country), value=T)
flickrcar_nonoids <- flickrcar_wo_country[, setdiff(names(flickrcar_wo_country),cols), with=FALSE]
flickrcar_wo_country[, setdiff(names(flickrcar_wo_country),cols):=NULL]

ncol(flickrcar_wo_country)

#NAs to zeros
f_dowle2 = function(DT) {
  for (i in names(DT))
    DT[is.na(get(i)), (i):=0]
}

f_dowle2(flickrcar_wo_country)

#one percent filter

mask2 <- flickrcar_wo_country[,lapply(.SD, function(x) sum(x!=0)>(nrow(flickrcar_wo_country)/100)) ]
flickrcar_onepercent <- flickrcar_wo_country[, names(which(unlist(mask2))), with=FALSE]
rm(flickrcar_wo_country)
ncol(flickrcar_onepercent)


###low variance filter
#kick out the zero variances
mask = flickrcar_onepercent[,lapply(.SD, function(x) var(x, na.rm = TRUE) > 0)]
flickrcar_nozerovar <- flickrcar_onepercent[, names(which(unlist(mask))), with=FALSE]
rm(flickrcar_onepercent)
ncol(flickrcar_nozerovar)


###high correlation filter
corMat <- cor(flickrcar_nozerovar)
#variance cutoff is set very low , which we legitimate with score structure 
highlyCor <- caret::findCorrelation(corMat, cutoff=0.70)
flickrcar_nozerovar[,unlist(highlyCor):=NULL]
ncol(flickrcar_nozerovar)

##pca
pca <- princomp(flickrcar_nozerovar, cor = FALSE, scores = TRUE)

summary(pca) #to see the cumulative proportion of the individual components

theloadings <- unclass(loadings(pca))
write.csv(theloadings,file="D:/Flickr/flickrexport_cars_oid_pca_loadings_coded_wocountry_wopol.csv")

write.csv(corMat,file="D:/Flickr/flickrexport_cars_oid_corMat_wopol.csv")

#######add to other variables
flickr_modeldata <- cbind(flickrcar_nonoids,as.data.table(pca$scores[,1:4]))

###check the components
flickr_modeldata[flickr_modeldata$Comp.1<quantile(flickr_modeldata$Comp.1,.001),]$url_t
flickr_modeldata[flickr_modeldata$Comp.1>quantile(flickr_modeldata$Comp.1,.999),]$url_t
flickr_modeldata[flickr_modeldata$Comp.1>quantile(flickr_modeldata$Comp.1,.4) & flickr_modeldata$Comp.1<quantile(flickr_modeldata$Comp.1,.6),]$url_t

flickr_modeldata[flickr_modeldata$Comp.2<quantile(flickr_modeldata$Comp.2,.001),]$url_t
flickr_modeldata[flickr_modeldata$Comp.2>quantile(flickr_modeldata$Comp.2,.999),]$url_t
flickr_modeldata[flickr_modeldata$Comp.2>quantile(flickr_modeldata$Comp.2,.4) & flickr_modeldata$Comp.2<quantile(flickr_modeldata$Comp.2,.6),]$url_t

flickr_modeldata[flickr_modeldata$Comp.4<quantile(flickr_modeldata$Comp.4,.001),]$url_t
flickr_modeldata[flickr_modeldata$Comp.4>quantile(flickr_modeldata$Comp.4,.999),]$url_t
flickr_modeldata[flickr_modeldata$Comp.4>quantile(flickr_modeldata$Comp.4,.4) & flickr_modeldata$Comp.4<quantile(flickr_modeldata$Comp.4,.6),]$url_t


flickr_modeldata$conversion <- ifelse(flickr_modeldata$views==0,0,flickr_modeldata$count_faves/flickr_modeldata$views)

flickr_modeldata$group_url <- as.factor(flickr_modeldata$group_url)


flickr_modeldata$Class <- as.factor(as.character(flickr_modeldata$Class))
flickr_modeldata$Class <- relevel(flickr_modeldata$Class,"general")

#clean up
rm(flickrcar_nonoids)
rm(flickrcar_nozerovar)
rm(pca)
rm(mask)
rm(mask2)
rm(flickr_groups_cars)
rm(corMat)
rm(theloadings)
rm(cols)
rm(highlyCor)


###OLS regressions
#we include Class everywhere, even though it is linear combination of groups
lm_pure <- lm(conversion~dateadded+group_url+license+precontext.nextphoto.views+precontext.prevphoto.views+Comp.3+Class,data=flickr_modeldata)
summary(lm_pure)
lm.beta(lm_pure)

#lm_carclass <- lm(conversion~dateadded+group_url+license+precontext.nextphoto.views+precontext.prevphoto.views+Comp.1+Comp.6+Comp.12,data=flickr_modeldata)
lm_carclass <- lm(conversion~dateadded+group_url+license+precontext.nextphoto.views+precontext.prevphoto.views+Comp.3+Comp.1+Comp.2+Comp.4+Class,data=flickr_modeldata)


summary(lm_carclass)
lm.beta(lm_carclass)
modelCompare(lm_pure, lm_carclass)

#lm_carclass_mod <- lm(conversion~dateadded+group_url+license+precontext.nextphoto.views+precontext.prevphoto.views+Comp.1*Class+Comp.4*Class+Comp.2+Comp.3,data=flickr_modeldata)
#lm_carclass_mod <- lm(log(conversion+1)~dateadded+group_url+license+precontext.nextphoto.views+precontext.prevphoto.views+Comp.1*Class+Comp.4*Class+Comp.2+Comp.3,data=flickr_modeldata)
lm_carclass_mod <- lm(log(count_faves+1)~log(views+1)+dateadded+group_url+license+log(precontext.nextphoto.views+1)+log(precontext.prevphoto.views+1)+scale(Comp.1)*Class+scale(Comp.4)*Class+scale(Comp.2)+scale(Comp.3),data=flickr_modeldata)
confint(lm_carclass_mod,level=0.95)

summary(lm_carclass_mod)
modelCompare(lm_carclass, lm_carclass_mod)

lm.beta(lm_carclass_mod)

###now without zero faves
lm_carclass_mod_no0 <- lm(log(count_faves)~log(views+1)+dateadded+group_url+license+log(precontext.nextphoto.views+1)+log(precontext.prevphoto.views+1)+scale(Comp.1)*Class+scale(Comp.4)*Class+scale(Comp.2)+scale(Comp.3),data=flickr_modeldata[count_faves>0,])

###and check poisson
fit_poisson <- glm(count_faves~views+dateadded+group_url+license+precontext.nextphoto.views+precontext.prevphoto.views+Comp.3+Comp.1*Class+Comp.2+Comp.4*Class,data=flickr_modeldata, family=poisson())
