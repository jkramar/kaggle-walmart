# Data desc:
# TripType: the answer.
# VisitNumber: ID.
# Weekday.
# UPC number.
# ScanCount: how many were bought.
# DepartmentDesc: rough prod cat.
# FinelineNumber: fine prod cat. (vaguely orthogonal to DepartmentDesc)
require(data.table)
sample_sub = read.csv("sample_submission.csv")
test = data.table(read.csv("test.csv"))
train = data.table(read.csv("train.csv"))
set.seed(3L)
train[, VisitNumber := c(factor(VisitNumber,levels=sample(levels(factor(VisitNumber)))))]
setkey(train, VisitNumber)
train[, Upc := factor(Upc,exclude=NULL,levels=names(sort(table(Upc,exclude=NULL),decreasing = TRUE)))]
test[, Upc := factor(Upc,exclude=NULL,levels=levels(train$Upc))]
train[, FinelineNumber := factor(FinelineNumber,exclude=NULL,levels=names(sort(table(FinelineNumber,exclude=NULL),decreasing = TRUE)))]
test[, FinelineNumber := factor(FinelineNumber,exclude=NULL,levels=levels(train$FinelineNumber))]
test[, DepartmentDescription := factor(DepartmentDescription,exclude=NULL,levels=levels(train$DepartmentDescription))]
# we'll see if the NA's get us in trouble. they're just labels ....
#train[, Upc := ifelse(is.na(Upc),-1,Upc)]
#test[, Upc := ifelse(is.na(Upc),-1,Upc)]
#train[, FinelineNumber := ifelse(is.na(FinelineNumber),-1,FinelineNumber)]
#test[, FinelineNumber := ifelse(is.na(FinelineNumber),-1,FinelineNumber)]

# there is much cruft. let's do the simplest good thing: select the most popular "words" of each kind,
# fit a logistic regression on whether they're there, plot learning curves.
# can then fool around with caret, mlr, the rtexttools training suggestions, or whatever.


# what are the NULLs like? well, Upc is null whenever DepartmentDescription is, and usually when PharmacyRx is.
# eonly with null/PharmacyRx DepartmentDescription.
dummify = function(x,y=1,by=1:length(x)) {
  difs=which(tail(by,-1)!=head(by,-1))
  new("matrix.csr",
      ra=rep_len(y,length(x))[!is.na(x)],
      ia=cumsum(c(1L,!is.na(x)))[1+c(0,which(tail(by,-1)!=head(by,-1)),length(by))],
      ja=c(x[!is.na(x)]),
      dimension=c(sum(tail(by,-1)!=head(by,-1))+1L,length(levels(x))))
}
  #require(SparseM)
  #SparseM::model.matrix(~x-1,list(x=x))
require(SparseM)
#require(glmnet)
Upcs = with(train[,.N,by=.(VisitNumber,Upc)],dummify(Upc,N,VisitNumber))
Finelines = with(train[,.N,by=.(VisitNumber,FinelineNumber)],dummify(FinelineNumber,N,VisitNumber))
Departments = with(train[,.N,by=.(VisitNumber,DepartmentDescription)],dummify(DepartmentDescription,N,VisitNumber))
UpcS = with(train[,sum(pmax(ScanCount,0)),by=.(VisitNumber,Upc)],dummify(Upc,V1,VisitNumber))
FinelinesS = with(train[,sum(pmax(ScanCount,0)),by=.(VisitNumber,FinelineNumber)],dummify(FinelineNumber,V1,VisitNumber))
DepartmentsS = with(train[,sum(pmax(ScanCount,0)),by=.(VisitNumber,DepartmentDescription)],dummify(DepartmentDescription,V1,VisitNumber))
UpcA = with(train[,sum(pmax(-ScanCount,0)),by=.(VisitNumber,Upc)],dummify(Upc,V1,VisitNumber))
FinelinesA = with(train[,sum(pmax(-ScanCount,0)),by=.(VisitNumber,FinelineNumber)],dummify(FinelineNumber,V1,VisitNumber))
DepartmentsA = with(train[,sum(pmax(-ScanCount,0)),by=.(VisitNumber,DepartmentDescription)],dummify(DepartmentDescription,V1,VisitNumber))
{
#Upcs = train[
#  ,.(VisitNumber,Upc=factor(ifelse(Upc%in%topUpc,Upc,NA),exclude=NULL),ScanCount)][
#    ,data.table(cbind(VisitNumber,dummify(Upc)))][
#      ,lapply(.SD,sum),by=VisitNumber][
#        ,.SD[,2:(length(.SD)-1),with=FALSE]]
#Finelines = train[
#  ,.(VisitNumber,Fineline=factor(ifelse(FinelineNumber%in%topFineline,FinelineNumber,NA),exclude=NULL),ScanCount)][
#    ,data.table(cbind(VisitNumber,dummify(Fineline)))][
#      ,lapply(.SD,sum),by=VisitNumber][
#        ,.SD[,2:(length(.SD)-1),with=FALSE]]
#Finelines = as.data.frame(matrix(0,nrow(Finelines),0))
#Departments = train[
#  ,data.table(cbind(VisitNumber,model.matrix(~DepartmentDescription-1)))][
#    ,lapply(.SD,sum),by=VisitNumber][
#      ,.SD[,2:(length(.SD)-1),with=FALSE]]
#Departments = train[
#  ,cbind.matrix.csr(as.matrix.csr(VisitNumber),as.matrix.csr(dummify(DepartmentDescription)))]
#DepartmentsS = train[
#  ,data.table(cbind(VisitNumber,pmax(ScanCount,0)*model.matrix(~DepartmentDescription-1)))][
#    ,lapply(.SD,sum),by=VisitNumber][
#      ,.SD[,2:(length(.SD)-1),with=FALSE]]
#DepartmentsA = train[
#  ,data.table(cbind(VisitNumber,pmax(-ScanCount,0)*model.matrix(~DepartmentDescription-1)))][
#    ,lapply(.SD,sum),by=VisitNumber][
#      ,.SD[,2:(length(.SD)-1),with=FALSE]]
}
Weekdays = dummify(train[,.(Weekday=Weekday[1]),by=VisitNumber]$Weekday)



# TripType 14 sucks, we have very few of it. main predictor is FinelineNumber 7949
raw.TripTypes=train[,.(TripType=TripType[1]),by=VisitNumber]
TripTypes = factor(raw.TripTypes[,ifelse(TripType==14,NA,TripType)])
Y.wide = dummify(TripTypes)
#Y.wide = subset(model.matrix(~x-1,list(x=factor(raw.TripTypes$TripType))),select=-x14)

validIndices=75000+which(!is.na(tail(TripTypes,-75000)))


rowSums.csr = function(A) {
  cs = c(0,cumsum(A@ra))
  cs[tail(A@ia,-1)]-cs[head(A@ia,-1)]
}
colSums.csr = function(A) rowSums.csr(t(A))
as.pos = function(x) pmax(x,.Machine$double.eps)
tfidfy = function(A,B=A) t(t(A * (1/as.pos(rowSums.csr(A)))) * log2(nrow(B)/as.pos(colSums.csr(B>0))))
DepartmentsTI = tfidfy(Departments)
DepartmentsSTI = tfidfy(DepartmentsS)
DepartmentsATI = tfidfy(DepartmentsA)
FinelinesTI = tfidfy(Finelines)
FinelinesSTI = tfidfy(FinelinesS)
FinelinesATI = tfidfy(FinelinesA)
UpcsTI = tfidfy(Upcs)


# this is now a new thing, let's see:
train[,fine1:=as.integer(as.character(FinelineNumber))%/%1000]
train[,fine2:=(as.integer(as.character(FinelineNumber))%/%100)%%10]
train[,fine3:=(as.integer(as.character(FinelineNumber))%/%10)%%10]
train[,fine4:=as.integer(as.character(FinelineNumber))%%10]
Fine1 = with(train[,.N,by=.(VisitNumber,fine1)],dummify(factor(fine1,0:9),N,VisitNumber))
Fine2 = with(train[,.N,by=.(VisitNumber,fine2)],dummify(factor(fine2,0:9),N,VisitNumber))
Fine3 = with(train[,.N,by=.(VisitNumber,fine3)],dummify(factor(fine3,0:9),N,VisitNumber))
Fine4 = with(train[,.N,by=.(VisitNumber,fine4)],dummify(factor(fine4,0:9),N,VisitNumber))
Fine12 = with(train[,.N,by=.(VisitNumber,f=fine1*10+fine2)],dummify(factor(f,0:99),N,VisitNumber))
Fine13 = with(train[,.N,by=.(VisitNumber,f=fine1*10+fine3)],dummify(factor(f,0:99),N,VisitNumber))
Fine14 = with(train[,.N,by=.(VisitNumber,f=fine1*10+fine4)],dummify(factor(f,0:99),N,VisitNumber))
Fine23 = with(train[,.N,by=.(VisitNumber,f=fine2*10+fine3)],dummify(factor(f,0:99),N,VisitNumber))
Fine24 = with(train[,.N,by=.(VisitNumber,f=fine2*10+fine4)],dummify(factor(f,0:99),N,VisitNumber))
Fine34 = with(train[,.N,by=.(VisitNumber,f=fine3*10+fine4)],dummify(factor(f,0:99),N,VisitNumber))

X=cbind(Departments,DepartmentsTI,DepartmentsS,DepartmentsSTI,Finelines,Upcs,
        Departments>0,Finelines>0,Upcs>0,Fine12,
        t(as.matrix.csr(apply(cbind(log(rowSums.csr(Departments)),log1p(rowSums.csr(DepartmentsS)),log1p(rowSums.csr(DepartmentsA))),
                              1,function(x){c(x,x^2/2,x^3/6,x^4/24)}))))
# this was great! and it's huge. try smaller?
{
# Depts, Weekdays, 1500: -1.21, -1.82
# Depts, Weekdays, 10k: -1.36, -1.53
# Depts, 1500: -1.22, -1.78
# Depts, 10k: -1.36, -1.53
# Depts, Finelines, 1500: -1.12, -1.76
# Depts, Finelines, 10k: -1.28, -1.51
# Finelines, 1500: -2.45, -2.96
# Depts, Upcs, 1500: -1.19, -1.98
# Depts, Upcs, 10k: -1.30, -1.56

# so maybe Finelines help a bit, but probably Depts carry most of the weight
# Depts, DeptS, 1500: -1.05, -1.70
# Depts, DeptS, 10k: -1.20, -1.42
# DeptS, 1500: -1.22, -1.76
# DeptS, 10k: -1.42, -1.55
# Depts, DeptS, DeptA, 1500: -1.01, -1.72

# with L1:

# Depts, DeptS, 1500: -1.10, -1.66
# Depts, DeptS, 10k: -1.22, -1.41

# so L1 isn't really worthwhile.
# "normalizing" by just dividing by sd or mean and having no bias term is horrible.
# but dividing by sd and then leaving the bias term gives:
# Depts, DeptS, 1500: -1.05, -1.74
# Depts, DeptS, 2500: -1.08, -1.63
# Depts, DeptS, 10000: -1.20, -1.42
# Depts, DeptS, DeptSTI, 300Finelines, 1500: -0.73, -1.53
# with totals of Depts and DeptS as well: -0.71, -1.52
# and now logging those: -0.66, -1.46
# with 10k: -0.77, -1.12
# sweet. now back to 1500. let's try 400Finelines: -0.64, -1.45
# 1000Finelines? : -0.53, -1.43
# ok that's cool. now let's cool off, take away the Finelines, add 300 Upcs : -0.74, -1.45
# and then adding weekdays : -0.75, -1.45 does nothing.
# 300Fin,300Upc: -0.63, -1.45
# 1000Fin,1000Upc: -0.45, -1.43
# ok so validation performance is barely budging. let's try N=5000 with no Fin/Upc: -0.88, -1.21
# with 300Fin: -0.74, -1.21
# with 600Fin: -0.65, -1.20
# with 2000Fin, 2000Upc: -0.42, -1.16
# now with "cost=0.84" to save time, and allFin,5000Upc: -0.37, -1.15
# 8kUPC: -0.36, -1.15
# 20kUPC: -0.34, -1.14
# 40kUPC: -0.32, -1.14
# allUPC: -0.31, -1.14
# 10ksamples: -0.30, -1.05
# 75ksamples: -0.32, -0.85
# now I could add more (S/A) variants of UPC&Fin, try tfidfing, add more variants of the raw bulk thing, tweak the cost,
# try lasso, ...
# ok. trying lasso. 10ksamples, with cost=0.84: -0.62, -1.07
# refitting cost, got cost=0.9, with the same results. fitting is slightly (not noticeably) faster. screw it.
# now adding DeptsSTI: -0.28, -1.03.
# and DeptsA: -0.28, -1.02.
# and DeptsATI: -0.27, -1.02.
# and FineS: -0.22, -1.05 (or if refitting cost, -0.29, -1.04)
# ok, taking that back. add log1p(sum(DeptA)): -0.27, -1.02
# fine. now adding quadratic terms: -0.26, -1.00
# and cubic: -0.27, -0.99
# and quartic fucks things up (??), so screw that.
# replacing Fines with FinesTI: -0.17, -1.02
#  ... let's not do that. re quartic, we could normalize (x^k/k!): -0.26, -1.00 . great.
# ok. now adding FinesTI: -0.15, -1.04. maybe it can work with larger N, but meh.
# trying instead FinesA: -0.26, -1.00 . meh.
# replacing Upcs with UpcsTI: -0.10, -1.04.
# adding UpcsTI: -0.09, -1.04.
# so let's not do that either. instead let's go for a submission.
# with 75k and cost=0.54235: -0.35, -0.81
# hmm. let's backtrack and add binaries for Depts: -0.25, -0.97
# and for Fines: -0.21, -0.97
# and (sillily) for Upcs: -0.18, -0.97
}
X.means=colMeans(X)
X.sds=apply(X,2,sd)
X.sds=ifelse(X.sds==0,1,X.sds)
#X=t((t(X))/X.sds)
predict.liblin = function(model, data) {
  ret = predict(model, data, proba=TRUE)$probabilities
  as.pos(ret[,order(factor(colnames(ret),levels(model$ClassNames)))])
}
predict.liblin.dec = function(model, data) {
  ret = predict(model, data, decisionValues=TRUE)$decisionValues
  ret[,order(factor(colnames(ret),levels(model$ClassNames)))]
}
predict.libsvm = function(model, data) {
  ret = predict(model, data, probability=TRUE)
  p = attr(ret, "probabilities")
  as.pos(p[,order(factor(colnames(p),levels(ret)))])
}
# probs = plogis(dec)/sum(plogis(dec))
require(LiblineaR)
require(optimx)
# X=X[,1:100]
read.fit = function(file) {
  fit = read.table(file,head=TRUE)
  fit = fit[,2:ncol(fit)]
  colnames(fit) = substring(colnames(fit),2)
  as.matrix(fit[,order(factor(colnames(fit),levels(TripTypes)))])
}
weirdvalid = factor(as.character(read.matrix.csr("valid.dat")$y),levels(TripTypes))
weirdvalid.wide = dummify(weirdvalid)
howgood.withsmoo = function(fit, truth)
  optimize(function(lam) {
    smoo = t((as.numeric(table(TripTypes))/sum(table(TripTypes))*lam+(1-lam)*t(fit)))
    mean(rowSums.csr(log(smoo)*truth))
  },c(0,1),maximum=TRUE)$objective
for(N in c(1500,2750,5000,9000)) {
  trainIndices=which(!is.na(TripTypes[1:N]))
  cost = optimize(function(cost){
    system.time(fitted <- LiblineaR(X[trainIndices,],TripTypes[trainIndices],cost=cost,bias=TRUE))
    valid.fit <- predict.liblin(fitted,X[validIndices,])
    lam=optimize(function(lam) {
      valid.smoo <- t((as.numeric(table(TripTypes))/sum(table(TripTypes))*lam+(1-lam)*t(valid.fit)))
      valid.howgood <- mean(rowSums.csr(log(valid.smoo)*Y.wide[validIndices,]))
      -valid.howgood
    },c(0,1))$minimum
    valid.smoo <- t((as.numeric(table(TripTypes))/sum(table(TripTypes))*lam+(1-lam)*t(valid.fit)))
    valid.howgood <- mean(rowSums.csr(log(valid.smoo)*Y.wide[validIndices,]))
    train.fit <- predict.liblin(fitted,X[trainIndices,])
    train.smoo <- t((as.numeric(table(TripTypes))/sum(table(TripTypes))*lam+(1-lam)*t(train.fit)))
    train.howgood=mean(rowSums.csr(log(train.smoo)*Y.wide[trainIndices,]))
    print(c(cost=cost,lam=lam,train.howgood=train.howgood,valid.howgood=valid.howgood))
    -valid.howgood
  },c(0,2))$minimum
  cost.svm = 12 # 12 for radial, 0.4 for linear
  optim(c())
  # ok this is being horrible, how can I make it less horrible?
  # - try insisting on linear (with enough features it's just as good)
  # - liblinear is so much faster... really can't get probs?
  # - increase cache size
  # - try not shrinking
  # - increase tol
  # note, svm-scale is a thing that can probably be told to scale to [0,1]. though actually svm() is smart, if(sparse) makes all the "scale"s sparse.
  # in terms of gridding, on cmdline, there's some "grid.py" thing to handle that
  # can try different ECCs and compare performance :)
  # and multicore liblinear exists.
  par = optimx(c(log(.002),log(12)), function(par){
    gamma = exp(par[1])
    cost.svm = exp(par[2])
    system.time(fitted <- svm(X[trainIndices,],TripTypes[trainIndices],cost=cost.svm,probability=TRUE,kernel="radial",gamma=gamma))
    valid.fit <- predict.libsvm(fitted,X[validIndices,])
    lam=optimize(function(lam) {
      valid.smoo <- t((as.numeric(table(TripTypes))/sum(table(TripTypes))*lam+(1-lam)*t(valid.fit)))
      valid.howgood <- mean(rowSums.csr(log(valid.smoo)*Y.wide[validIndices,]))
      -valid.howgood
    },c(0,1))$minimum
    valid.smoo <- t((as.numeric(table(TripTypes))/sum(table(TripTypes))*lam+(1-lam)*t(valid.fit)))
    valid.howgood <- mean(rowSums.csr(log(valid.smoo)*Y.wide[validIndices,]))
    train.fit <- predict.libsvm(fitted,X[trainIndices,])
    train.smoo <- t((as.numeric(table(TripTypes))/sum(table(TripTypes))*lam+(1-lam)*t(train.fit)))
    train.howgood=mean(rowSums.csr(log(train.smoo)*Y.wide[trainIndices,]))
    print(c(cost.svm=cost.svm,lam=lam,train.howgood=train.howgood,valid.howgood=valid.howgood))
    -valid.howgood
  })
  cost.svm = optimize(function(cost.svm){
    system.time(fitted <- svm(X[trainIndices,],TripTypes[trainIndices],cost=cost.svm,probability=TRUE,kernel="linear"))
    valid.fit <- predict.libsvm(fitted,X[validIndices,])
    lam=optimize(function(lam) {
      valid.smoo <- t((as.numeric(table(TripTypes))/sum(table(TripTypes))*lam+(1-lam)*t(valid.fit)))
      valid.howgood <- mean(rowSums.csr(log(valid.smoo)*Y.wide[validIndices,]))
      -valid.howgood
    },c(0,1))$minimum
    valid.smoo <- t((as.numeric(table(TripTypes))/sum(table(TripTypes))*lam+(1-lam)*t(valid.fit)))
    valid.howgood <- mean(rowSums.csr(log(valid.smoo)*Y.wide[validIndices,]))
    train.fit <- predict.libsvm(fitted,X[trainIndices,])
    train.smoo <- t((as.numeric(table(TripTypes))/sum(table(TripTypes))*lam+(1-lam)*t(train.fit)))
    train.howgood=mean(rowSums.csr(log(train.smoo)*Y.wide[trainIndices,]))
    print(c(cost.svm=cost.svm,lam=lam,train.howgood=train.howgood,valid.howgood=valid.howgood))
    -valid.howgood
  },c(0,20))$minimum
  
}
# boosting with 1500, default settings, gives -0.19 (!!!!) -2.32
#                               eta=0.1 gives -0.23, -2.08
#                         subsample=0.5 gives -0.24, -2.63
#                              eta=0.03 gives -0.33, -1.84
#                             eta=0.003 gives -1.16, and then who cares
#                           max.depth=3 gives -0.23, -2.31

# having more fun now. 5k samples, 4 threads, L1 regularization beats L2, alpha=1 or 3 decent (.3,10 bad)
# 5k, alpha=1: 0.36, 1.06 after 41 rounds.
#  & max_depth=2: 0.46, 1.02 after 202 rounds.
#  & max_depth=1: 0.55, 1.02 after 544 rounds.
# 5k, alpha=3: 0.48, 1.06 after 119 rounds.
# 5k, alpha=2: 0.38, 1.05 after 49 rounds.
# 5k, alpha=2, subsample=colsample_bytree=0.5, num_parallel_tree=10: 0.46, 1.02 after 71 rounds. (but they're slow rounds.)
# 5k, alpha=2, subsample=colsample_bytree=0.5: 0.45, 1.07 after 75 rounds.
# 5k, alpha=2, subsample=colsample_bytree=0.5, eta=.1: 0.46 (or 0.45), 1.02 after 214 rounds.
# 5k, alpha=2, subsample=0.5, eta=.1: 0.46, 1.03 after 181 rounds.
# have lots of params to play with: alpha, eta, gamma, colsample_bytree, subsample, lambda
trainIndices=which(!is.na(TripTypes[1:75000]))
dm=xgb.DMatrix(as(X[trainIndices],"Matrix"),label=c(TripTypes[trainIndices])-1)
dm5k=xgb.DMatrix(as(X[trainIndices[1:5000]],"Matrix"),label=c(TripTypes[trainIndices[1:5000]])-1)
dm10k=xgb.DMatrix(as(X[trainIndices[1:10000]],"Matrix"),label=c(TripTypes[trainIndices[1:10000]])-1)
dmval=xgb.DMatrix(as(X[validIndices],"Matrix"),label=c(TripTypes[validIndices])-1)
stuffs2=data.frame(alpha=c(),eta=c(),gamma=c(),lambda=c(),colsample_bytree=c(),subsample=c(),max_depth=c(),sc=c(),time.taken=c())
stuffs3=data.frame(alpha=c(),eta=c(),gamma=c(),lambda=c(),colsample_bytree=c(),subsample=c(),max_depth=c(),sc=c(),time.taken=c())

cleanerFeatures=c(1:(69*4+5196),(69*4+5196+97715+1):(69*4+5196+97715+69+5196),(69*4+5196+97715+69+5196+97715+1):(69*4+5196+97715+69+5196+97715+112))
dm.t=xgb.DMatrix(as(X[trainIndices,cleanerFeatures],"Matrix"),label=c(TripTypes[trainIndices])-1)
dmval.t=xgb.DMatrix(as(X[validIndices,cleanerFeatures],"Matrix"),label=c(TripTypes[validIndices])-1)
# max_depth and eta should be handled separately. eta generally improves quality but has lots of timewasting potential.
# so let's try sticking with eta=.4 or something.
# hm, finding that subsample isn't helpful. I wonder whether this is related to eta being high and N being 5k ...
# also this is getting boring, everything's close to 1.02 (which is what I had found manually too). maybe it's time to use a bigger dataset...
# or actually let's try increasing tree depth again to 2...
# okay, tree depth 2 looks way faster/better. 3? less good with same params, let's see how it tunes. ah, pretty well.
# can depth-6 match it? unclear, let's see it tune.
# meh. not that fruitful. let's stick with 2 for now, try 10k samples. ... okay that's slow.
# subsampling doesn't actually produce a speedup, which is disappointing.

# optim / optimx seem horrible at par scaling with nelder-mead, should try the nloptr version
#optimx(c(alpha=log(.63),gamma=log(.75),cols=qnorm(.6),subs=qnorm(.9),lambd=log(1.5)),function(par){
#  alpha=exp(par[1])
#  gamma=exp(par[2])
#  colsample_bytree=pnorm(par[3])
#  subsample=pnorm(par[4])
#  lambda=exp(par[5])
  
for(fno in 43:1000) {
  alpha=.4 #.63
  gamma=0 #.75
  colsample_bytree=.6
  subsample=.9
  lambda=1.5
  eta=.1
  max_depth=3
#shuf=sample(trainIndices)
#dmshuf=xgb.DMatrix(as(X[shuf],"Matrix"),label=c(TripTypes[shuf])-1)

# dm.t is interesting! try weakening the regularization, say to .4.
  fitted=xgb.train(list(),dm.t,10000,watchlist=list(valid=dmval.t,train=dm.t),objective="multi:softprob",num_class=nlevels(TripTypes),eval_metric="mlogloss",nthread=4,early.stop.round=200,
                                    alpha=alpha, eta=eta, gamma=gamma, lambda=lambda, colsample_bytree=colsample_bytree, subsample=subsample,
                                    max_depth=max_depth, print.every.n=10, round=1, num_parallel_trees=1)
  dims=list(train=ncdim_def("itrain","",1:length(trainIndices)),valid=ncdim_def("ivalid","",1:length(validIndices)),test=ncdim_def("itest","",1:95674),preds=ncdim_def("ipreds","",1:37))
  vars=list(train=ncvar_def("train","",list(dims$train,dims$preds)),valid=ncvar_def("valid","",list(dims$valid,dims$preds)),test=ncvar_def("test","",list(dims$test,dims$preds)))
  fname=paste("a_fitted_xgb_",fno,".nc",sep="")
  stopifnot(!file.exists(fname))
  f=nc_create(fname,vars)
  ncvar_put(f,vars$train,predict.xgb(fitted,dm.t))
  ncvar_put(f,vars$valid,predict.xgb(fitted,dmval.t))
  ncvar_put(f,vars$test,predict.xgb(fitted,dmtest.t))
  nc_close(f)
}

  print("")
  stuffs3<<-rbind(stuffs3,data.frame(alpha=alpha,eta=eta,gamma=gamma,lambda=lambda,colsample_bytree,subsample=subsample,max_depth=max_depth,
          sc=sc$bestScore,time.taken=time.taken))
  print(stuffs3)
#  sc+time.taken*(.01/300)
#},control=list(parscale=rep(100,5)))

alpha=1 #.63
gamma=0 #.75
colsample_bytree=.6
subsample=.9
lambda=1.5
eta=.12
max_depth=2
time.taken = system.time(
  bestscore <- xgb.train(list(),dm5k,10000,watchlist=list(valid=dmval,train=dm5k),objective="multi:softprob",num_class=nlevels(TripTypes),eval_metric="mlogloss",nthread=4,early.stop.round=20,
                  alpha=alpha, eta=eta, gamma=gamma, lambda=lambda, colsample_bytree=colsample_bytree, subsample=subsample,
                  max_depth=max_depth, print.every.n=10, num_parallel_trees=1, round=1)$bestScore)["elapsed"]
print("")
print(bestscore)
print(time.taken)

predict.xgb=function(fit,x) t(matrix(predict(fit,x),37))
get_nc=function(filename) {
  f=nc_open(filename)
  train.fit=ncvar_get(f,"train")
  valid.fit=ncvar_get(f,"valid")
  test.fit=ncvar_get(f,"test")
  if(dim(train.fit)[1]==37) {
    train.fit=t(train.fit)
    valid.fit=t(valid.fit)
    test.fit=t(test.fit)
  }
  list(train=train.fit,valid=valid.fit,test=test.fit)
}
valid.howgood=function(preds) {
  mean(rowSums(log(preds+.Machine$double.eps)*as.matrix(Y.wide)[validIndices,]))-
    mean(log(rowSums(preds+.Machine$double.eps)))
}

pps=list()
for(filename in list.files(".",".*[^x].nc")){
  tryCatch({
    g=valid.howgood(get_nc(filename)$valid)
    if(g>-0.8) {
      show(filename)
      show(g)
      pps[[filename]]=get_nc(filename)$valid
    }
  },error=function(e)e)
}

wts=exp(optim(c(log(wts),rep(0,length(pps)-length(wts))),function(lws){
  p=0
  for(i in 1:length(pps)) {
    p=p+exp(lws[i])*pps[[i]]
  }
  g=valid.howgood(p)
  show(exp(lws))
  show(g)
  -g
})$par)
wts=wts/sum(wts)

p=0
for(i in 1:length(pps)) {
  p=p+wts[i]*get_nc(names(pps)[i])$test
}
test.VisitNumber=test[,VisitNumber,by=VisitNumber]$VisitNumber
colnames(p)<-paste("TripType",levels(TripTypes),sep="_")
write.csv(data.frame(VisitNumber=test.VisitNumber,p*0.99996,TripType_14=0.00004),file="submission.csv",row.names=FALSE)


#... and now I'm looking at where the actual losses are.
# by true label: 12% 39, 7% 8, 6% 9, 5% 7/38/999/42, ...
# and of these, 39, 38, 42 are the ones we're predicting poorly.
# how poorly? when we're underestimating 39, we're (on weighted average) predicting 39 32% of the time, and then ~5-7% for
# 36, 7, 38, 35, 40, 37. so what's the deal there?
sample(validIndices[TripTypes[validIndices]=="39"],5,prob=(valid.fit[,"36"]*-log(valid.fit[,"39"]))[TripTypes[validIndices]=="39"])

# some examples of trips that were 39 but were ranked as probably-not-39 and rather probably-36: 92091,91678,88680,92233,91021
# and some vice-versa examples: 78086, 90092, 91423, 78427, 75742

# train[VisitNumber%in%c(78086,90092,91423,78427,75742,92091,91678,88680,92233,91021)]
# what do I see? I see 36's being more random/mixed trips, with a mix of household and personal, and 39 having more sorts of beauty product.
# this suggests I might benefit from chopping up FinelineNumbers into their constituent digits, digit-pairs, and digit-triples. (well, probably not triples :)

# tried that. now let's look again!
# loss allocation very similar; if anything, more extreme (14% 39, 8% 8, 7% 9, ...)
# in fact when predicting 39, there's essentially no false positives, lots of false negatives.
# ... ok, breaking down the logloss down more directly. biggest things: truth 9, prediction 8 or 9 or (less likely) 42 or 43; truth 5, prediction 8 or 9
# data.table(cbind(valid.fit,y=TripTypes[validIndices]))[,lapply(.SD,function(p)sum(log(p))),by=y][,.SD/sum(.SD)]
# some samples of losing by predicting 8 with truth 9
# 94094 85609 89773 88706 78371
# some samples of losing by underpredicting 9 with truth 9 (????)
# 94406 91848 93325 91484 80723
# some samples of losing on predicting 8 with truth 5
# 86000 92828 77240 84532 93292
# some samples of losing on predicting 9 with truth 5
# 82915 95008 82235 89233 81114

# what could go better? transf'd svm? transf'd NB? kknn/fnn::knn? gbm? adabag? nn? randomForest/randomForestSRC/ranger? 
# xgboost?
test.Upcs = with(test[,.N,by=.(VisitNumber,Upc)],dummify(Upc,N,VisitNumber))
test.Finelines = with(test[,.N,by=.(VisitNumber,FinelineNumber)],dummify(FinelineNumber,N,VisitNumber))
test.Departments = with(test[,.N,by=.(VisitNumber,DepartmentDescription)],dummify(DepartmentDescription,N,VisitNumber))
test.UpcS = with(test[,sum(pmax(ScanCount,0)),by=.(VisitNumber,Upc)],dummify(Upc,V1,VisitNumber))
test.FinelinesS = with(test[,sum(pmax(ScanCount,0)),by=.(VisitNumber,FinelineNumber)],dummify(FinelineNumber,V1,VisitNumber))
test.DepartmentsS = with(test[,sum(pmax(ScanCount,0)),by=.(VisitNumber,DepartmentDescription)],dummify(DepartmentDescription,V1,VisitNumber))
test.UpcA = with(test[,sum(pmax(-ScanCount,0)),by=.(VisitNumber,Upc)],dummify(Upc,V1,VisitNumber))
test.FinelinesA = with(test[,sum(pmax(-ScanCount,0)),by=.(VisitNumber,FinelineNumber)],dummify(FinelineNumber,V1,VisitNumber))
test.DepartmentsA = with(test[,sum(pmax(-ScanCount,0)),by=.(VisitNumber,DepartmentDescription)],dummify(DepartmentDescription,V1,VisitNumber))
test.Weekdays = dummify(test[,.(Weekday=Weekday[1]),by=VisitNumber]$Weekday)
test.DepartmentsTI = tfidfy(test.Departments,Departments)
test.DepartmentsSTI = tfidfy(test.DepartmentsS,DepartmentsS)
test.DepartmentsATI = tfidfy(test.DepartmentsA,DepartmentsA)
test.FinelinesTI = tfidfy(test.Finelines,Finelines)
test.FinelinesSTI = tfidfy(test.FinelinesS,FinelinesS)
test.FinelinesATI = tfidfy(test.FinelinesA,FinelinesA)
test.UpcsTI = tfidfy(test.Upcs,Upcs)
test[,fine1:=as.integer(as.character(FinelineNumber))%/%1000]
test[,fine2:=(as.integer(as.character(FinelineNumber))%/%100)%%10]
test[,fine3:=(as.integer(as.character(FinelineNumber))%/%10)%%10]
test[,fine4:=as.integer(as.character(FinelineNumber))%%10]
test.Fine1 = with(test[,.N,by=.(VisitNumber,fine1)],dummify(factor(fine1,0:9),N,VisitNumber))
test.Fine2 = with(test[,.N,by=.(VisitNumber,fine2)],dummify(factor(fine2,0:9),N,VisitNumber))
test.Fine3 = with(test[,.N,by=.(VisitNumber,fine3)],dummify(factor(fine3,0:9),N,VisitNumber))
test.Fine4 = with(test[,.N,by=.(VisitNumber,fine4)],dummify(factor(fine4,0:9),N,VisitNumber))
test.Fine12 = with(test[,.N,by=.(VisitNumber,f=fine1*10+fine2)],dummify(factor(f,0:99),N,VisitNumber))
test.Fine13 = with(test[,.N,by=.(VisitNumber,f=fine1*10+fine3)],dummify(factor(f,0:99),N,VisitNumber))
test.Fine14 = with(test[,.N,by=.(VisitNumber,f=fine1*10+fine4)],dummify(factor(f,0:99),N,VisitNumber))
test.Fine23 = with(test[,.N,by=.(VisitNumber,f=fine2*10+fine3)],dummify(factor(f,0:99),N,VisitNumber))
test.Fine24 = with(test[,.N,by=.(VisitNumber,f=fine2*10+fine4)],dummify(factor(f,0:99),N,VisitNumber))
test.Fine34 = with(test[,.N,by=.(VisitNumber,f=fine3*10+fine4)],dummify(factor(f,0:99),N,VisitNumber))


test.X=cbind(test.Departments,test.DepartmentsTI,test.DepartmentsS,test.DepartmentsSTI,test.Finelines,test.Upcs,
        test.Departments>0,test.Finelines>0,test.Upcs>0,test.Fine12,
        t(as.matrix.csr(apply(cbind(log(rowSums.csr(test.Departments)),log1p(rowSums.csr(test.DepartmentsS)),log1p(rowSums.csr(test.DepartmentsA))),
                              1,function(x){c(x,x^2/2,x^3/6,x^4/24)}))))

dmtest=xgb.DMatrix(as(test.X,"Matrix"))
dmtest.t=xgb.DMatrix(as(test.X[,cleanerFeatures],"Matrix"))

writeMM(as(test.X,"sparseMatrix"),"testX")


# test.X=cbind(as.matrix(test.DepartmentsTI),as.matrix(test.DepartmentsSTI))
# test.X=t(t(test.X)/X.sds)

test.fit=predict.liblin(fitted,test.X)
test.fit=t((as.numeric(table(TripTypes))/sum(table(TripTypes))*lam+(1-lam)*t(test.fit)))

# test.fit=predict(fitted.glm,as.matrix(cbind(test.Upcs,test.Finelines,test.Departments,test.Weekdays)),type="response",s=lambda)
test.VisitNumber=test[,VisitNumber,by=VisitNumber]$VisitNumber
colnames(test.fit)<-paste("TripType",colnames(test.fit),sep="_")
write.csv(data.frame(VisitNumber=test.VisitNumber,test.fit*0.99996,TripType_14=0.00004),file="submission.csv",row.names=FALSE)


# with all covars and N=1500, train:-1.16 and valid:-1.87.
# now removing Finelines...




glm.fit(as.matrix(cbind(Upcs,Finelines,Departments,Weekdays)),TripTypes==999,family="binomial")

train[Upc%in%topUpc,.(VisitNumber,Upc=factor(Upc))][,as.list(table(Upc)),by=VisitNumber]

train[,.(VisitNumber,Upc=factor(ifelse(Upc%in%topUpc,Upc,NA),exclude=NULL),ScanCount)][,data.table(cbind(VisitNumber,t(head(t(model.matrix(~Upc-1)),-1))))][,lapply(.SD,sum),by=VisitNumber]

































train[, VisitNumber := c(factor(VisitNumber))]
setkey(train, VisitNumber)
train[, FinelineNumber := factor(ifelse(is.na(FinelineNumber), -1, FinelineNumber))]
train[, DeptFine := factor(paste(DepartmentDescription, FinelineNumber))]
train[, Upc := factor(ifelse(is.na(Upc),-1,Upc))]
cvset=train[VisitNumber>75000]
train=train[VisitNumber<=75000]
#test[, Upc := ifelse(is.na(Upc),-1,Upc)]
#Upc.levels = levels(factor(c(train$Upc, test$Upc)))
#train[, Upc := factor(Upc, levels=Upc.levels)]
#test[, Upc := factor(Upc, levels=Upc.levels)]
setkey(test, VisitNumber)
test[, VisitNumber := c(factor(VisitNumber))]
test[, FinelineNumber := factor(ifelse(is.na(FinelineNumber), -1, FinelineNumber), levels=levels(train$FinelineNumber))]
ResortDtm <- function(working.dtm) {
  # sorts a sparse matrix in triplet format (i,j,v) first by i, then by j.
  # Args:
  #   working.dtm: a sparse matrix in i,j,v format using $i $j and $v respectively. Any other variables that may exist in the sparse matrix are not operated on, and will be returned as-is.
  # Returns:
  #   A sparse matrix sorted by i, then by j.
  working.df <- data.frame(i = working.dtm$i, j = working.dtm$j, v = working.dtm$v)  # create a data frame comprised of i,j,v values from the sparse matrix passed in.
  working.df <- working.df[order(working.df$i, working.df$j), ] # sort the data frame first by i, then by j.
  working.dtm$i <- working.df$i  # reassign the sparse matrix' i values with the i values from the sorted data frame.
  working.dtm$j <- working.df$j  # ditto for j values.
  working.dtm$v <- working.df$v  # ditto for v values.
  return(working.dtm) # pass back the (now sorted) data frame.
}  # end function
library(slam)
by.dept = train[,.(N=.N,Count=sum(pmax(0,ScanCount)),Antis=sum(ScanCount==-1)),by=.(VisitNumber,DepartmentDescription)]
by.fine = train[,.(N=.N,Count=sum(pmax(0,ScanCount)),Antis=sum(ScanCount==-1)),by=.(VisitNumber,FinelineNumber)]
by.dept.fine = train[,.(N=.N,Count=sum(pmax(0,ScanCount)),Antis=sum(ScanCount==-1)),by=.(VisitNumber,DeptFine)]
by.upc = train[,.(N=.N,Count=sum(pmax(0,ScanCount)),Antis=sum(ScanCount==-1)),by=.(VisitNumber,Upc)]
#triplicator = expression(train[,.(YY=Y,by=.(VisitNumber,X))][,simple_triplet_matrix(c(VisitNumber),c(X),YY,dimnames=.(levels(VisitNumber),levels(X)))])


N.by.dept = as.DocumentTermMatrix(weighting=weightTfIdf,
                                  by.dept[,simple_triplet_matrix(c(VisitNumber),c(DepartmentDescription),N,dimnames=.(levels(VisitNumber),levels(DepartmentDescription)))])
Count.by.dept = as.DocumentTermMatrix(weighting=weightTfIdf,
                                      by.dept[,simple_triplet_matrix(c(VisitNumber),c(DepartmentDescription),Count,dimnames=.(levels(VisitNumber),levels(DepartmentDescription)))])
Antis.by.dept = as.DocumentTermMatrix(weighting=weightTfIdf,
                                      by.dept[,simple_triplet_matrix(c(VisitNumber),c(DepartmentDescription),Antis,dimnames=.(levels(VisitNumber),levels(DepartmentDescription)))])
N.by.dept.fine = as.DocumentTermMatrix(weighting=weightTfIdf,
                                       by.dept.fine[,simple_triplet_matrix(c(VisitNumber),c(DeptFine),N)])
Count.by.dept.fine = as.DocumentTermMatrix(weighting=weightTfIdf,
                                           by.dept.fine[,simple_triplet_matrix(c(VisitNumber),c(DeptFine),Count)])
Antis.by.dept.fine = as.DocumentTermMatrix(weighting=weightTfIdf,
                                           by.dept.fine[,simple_triplet_matrix(c(VisitNumber),c(DeptFine),Antis)])
N.by.upc = as.DocumentTermMatrix(weighting=weightTfIdf,
                                 by.upc[,sparseMatrix(c(VisitNumber),c(Upc),x=N)])
Count.by.upc = as.DocumentTermMatrix(weighting=weightTfIdf,
                                     by.upc[,simple_triplet_matrix(c(VisitNumber),c(Upc),Count)])
Antis.by.upc = as.DocumentTermMatrix(weighting=weightTfIdf,
                                     by.upc[,simple_triplet_matrix(c(VisitNumber),c(Upc),Antis)])

library(tm)

#train[,keep.train:=rbinom(1,1,.6),by=VisitNumber]
#valid=train[!keep.train]
#train=train[!!keep.train]
library(ranger)
library(RTextTools)

m=with(train[, .N, by = list(VisitNumber, DepartmentDescription)],
       ResortDtm(as.DocumentTermMatrix(simple_triplet_matrix(
         c(VisitNumber), c(DepartmentDescription), N, dimnames = list(levels(VisitNumber), levels(DepartmentDescription))))
         mm=ResortDtm(as.DocumentTermMatrix(m,weighting = weightTfIdf))
         # Weekday, UPC Number, ScanCount, FinelineNumber
         
         # weekdaymatrix = as(train$Weekday,"sparseMatrix")
         
         mm.dims=dimnames(mm)
         mm=as.matrix.csr(as.matrix(mm))
         dimnames(mm)=mm.dims
         make.csr = function(m) {
           dims = dimnames(m)
           m = as.matrix.csr(as.matrix(m))
           if(is.list(dims)) dimnames(m)=dims
           m
         }
         # do not pass create_container a DocumentTermMatrix! it will eat you!!
         c=create_container(mm,
                            train[, TripType[1], by = VisitNumber]$V1,
                            trainSize = 1:50000, testSize=50001:95674, virgin=FALSE)
         cross_validate
         
         t = train_model(c,"BOOSTING")
         # resps=classify_models(c,ts)
         a = create_analytics(c,classify_model(c,t))
         1/mean(1/a@algorithm_summary$LOGITBOOST_FSCORE,na.rm=T)
         
         pred.goodness = function(preds, ys) {
           k=length(levels(preds))
           t=table(preds, ys)
           sum((log(t+k^-2)-log(rowSums(t)+k^-1))*(t+k^-2))/(sum(t)+1)
         }
         fit.dir = function(ns) {
           uniroot(function(x) {
             a=(1/x)-1
             mean(digamma(ns+a))+digamma(a*length(ns))-digamma(a)-digamma(a*length(ns)+sum(ns))
           }, c(1e-6,1-1e-6))
         }
         plot.dir.score = function(ns,lim=1) {
           plot(function(a)mean(digamma(ns+a))+digamma(a*length(ns))-digamma(a), c(0,lim))
         }
         ts=train_models(c,c("SVM","MAXENT","SLDA","BOOSTING","BAGGING","RF","NNET"))
         
         cs=classify_models(c,ts)
         a=create_analytics(c,cs)
         
         # in the result with 1000, nnet sucked, slda not very helpful, the others reasonable; logitboost seems strongest.
         # can probably tune more