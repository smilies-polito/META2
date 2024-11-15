
library(flacco)

#Define a function named flacco_FLA
flacco_FLA <- function(X, y){
    feat.object = createFeatureObject(X = X, y = y, blocks = 3)
    featureSet = calculateFeatureSet(feat.object, set = "ela_meta")
    for (i in c("cm_angle", "cm_conv", "cm_grad", "ela_distr", "ela_level", "basic", "disp", "limo", "nbc", "pca", "bt", "gcm", "ic")){
        featureSet = rbind(featureSet, calculateFeatureSet(feat.object, set = i))
    }
    return(featureSet)
}

flacco_FLA_cm_angle <- function(X, y){
    feat.object = createFeatureObject(X = X, y = y, blocks = 5)
    featureSet = calculateFeatureSet(feat.object, set = "cm_angle")
    return(featureSet)
}
flacco_FLA_cm_conv <- function(X, y){
    feat.object = createFeatureObject(X = X, y = y, blocks = 5)
    featureSet = calculateFeatureSet(feat.object, set = "cm_conv")
    return(featureSet)
}
flacco_FLA_cm_grad <- function(X, y){
    feat.object = createFeatureObject(X = X, y = y, blocks = 5)
    featureSet = calculateFeatureSet(feat.object, set = "cm_grad")
    return(featureSet)
}
flacco_FLA_ela_distr <- function(X, y){
    feat.object = createFeatureObject(X = X, y = y, blocks = 5)
    featureSet = calculateFeatureSet(feat.object, set = "ela_distr")
    return(featureSet)
}
flacco_FLA_ela_level <- function(X, y){
    feat.object = createFeatureObject(X = X, y = y, blocks = 5)
    featureSet = calculateFeatureSet(feat.object, set = "ela_level")
    return(featureSet)
}
flacco_FLA_basic <- function(X, y){
    feat.object = createFeatureObject(X = X, y = y, blocks = 5)
    featureSet = calculateFeatureSet(feat.object, set = "basic")
    return(featureSet)
}
flacco_FLA_disp <- function(X, y){
    feat.object = createFeatureObject(X = X, y = y, blocks = 5)
    featureSet = calculateFeatureSet(feat.object, set = "disp")
    return(featureSet)
}
flacco_FLA_limo <- function(X, y){
    feat.object = createFeatureObject(X = X, y = y, blocks = 5)
    featureSet = calculateFeatureSet(feat.object, set = "limo")
    return(featureSet)
}
flacco_FLA_nbc <- function(X, y){
    feat.object = createFeatureObject(X = X, y = y, blocks = 5)
    featureSet = calculateFeatureSet(feat.object, set = "nbc")
    return(featureSet)
}
flacco_FLA_pca <- function(X, y){
    feat.object = createFeatureObject(X = X, y = y, blocks = 5)
    featureSet = calculateFeatureSet(feat.object, set = "pca")
    return(featureSet)
}
flacco_FLA_bt <- function(X, y){
    feat.object = createFeatureObject(X = X, y = y, blocks = 5)
    featureSet = calculateFeatureSet(feat.object, set = "bt")
    return(featureSet)
}
flacco_FLA_gcm <- function(X, y){
    feat.object = createFeatureObject(X = X, y = y, blocks = 5)
    featureSet = calculateFeatureSet(feat.object, set = "gcm")
    return(featureSet)
}
flacco_FLA_ic <- function(X, y){
    feat.object = createFeatureObject(X = X, y = y, blocks = 5)
    featureSet = calculateFeatureSet(feat.object, set = "ic")
    return(featureSet)
}