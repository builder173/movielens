#Author: matt_173 at EdX
#This script develops the movie prediction model for HarvardX PH125.9x
#Objective: develop a model to minimize RMSE of predictions on out of sample validation data set
#Note: max points for RMSE < 0.86490

#Step 1 recreate data for training/testing/validation from movielens
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

#define all libraries here for transparency
library(tidyverse)
library(caret)
library(data.table)
library(caret)
library(ggrepel)
library(lubridate) 
library(gam)
library(irlba) #fast truncated singular value decomposition PCA for large dense and sparse matrices

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                            title = as.character(title),
#                                            genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Convenience save/load data to avoid reprocessing
#save(list = c("edx","validation"),file = "movies.RData")
load(file = "movies.RData")

#At this point we've created edx set of ratings for training/testing and validation set for final model validation
#Inspect the data structure
str(edx)
# Classes ‘data.table’ and 'data.frame':	9000055 obs. of  6 variables:
# $ userId   : int  1 1 1 1 1 1 1 1 1 1 ...
# $ movieId  : num  122 185 292 316 329 355 356 362 364 370 ...
# $ rating   : num  5 5 5 5 5 5 5 5 5 5 ...
# $ timestamp: int  838985046 838983525 838983421 838983392 838983392 838984474 838983653 838984885 838983707 838984596 ...
# $ title    : chr  "Boomerang (1992)" "Net, The (1995)" "Outbreak (1995)" "Stargate (1994)" ...
# $ genres   : chr  "Comedy|Romance" "Action|Crime|Thriller" "Action|Drama|Sci-Fi|Thriller" "Action|Adventure|Sci-Fi" ...
# - attr(*, ".internal.selfref")=<externalptr> 

#data validation
length(unique(edx$userId))
length(unique(validation$userId))
mean(unique(validation$userId) %in% unique(edx$userId))
length(unique(edx$movieId))
length(unique(validation$movieId))
mean(unique(validation$movieId) %in% unique(edx$movieId))
factor(unique(edx$rating))
factor(unique(validation$rating))
min(as_datetime(edx$timestamp))
max(as_datetime(edx$timestamp))
min(as_datetime(validation$timestamp))
max(as_datetime(validation$timestamp))
#note that genres and titles map to movieId
#check if there are reviews with timestampp year prior to movie year
edx %>% mutate(mov_yr = as.numeric(str_match(title, "\\((\\d{4})\\)$")[,2]), rev_yr = as.numeric(year(as_datetime(timestamp)))) %>%
  filter(rev_yr < mov_yr) %>% summarize(early_count = n()) #we see 175, so not material and usually review is marked one year earlier

#since model development also needs train/test sets we further divide edx for this purpose
set.seed(11, sample.kind="Rounding") #for reproducibility
test_index = createDataPartition(edx$rating, times=1, p=0.2, list=FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

#Define RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Outline of approach
# 1. Instructions say to "build on code already provided"
# 2. Assume that the starting point is regularized user/movie factor model
# 3. Perform exploratory analysis to identify and test additional factors/models to reduce RMSE
# 3a. There may be a time trend in average rating that we can model
# 3b. There may be genre a genre effect (e.g. lower or higher ratings for certain genres)
# 4 See if PCA on resiuduals is feasible and additive
# Stopping condition: expected RMSE below max rubric points credit. RMSE < 0.86490

# Baseline regularized user and movie effect model from Section 6 taken at starting point and repeated here
# The form of prediction is #   Y(u,i) =  mu + b(i) + b(u) where the b's are regularized parameters
# Explore potential for other effects following this same form


#Time effects
#Look at how average movie rating changes over time in the edx set, taking average by month yyyymm
#first mutate the train/test sets to assign a yearmonth column for later use

base_date = min(edx$timestamp)
#note 86,400 seconds in a day going to "bin" by ~half years
edx %>%  mutate(period = factor(round((timestamp - base_date)/(182*86400),0))) %>% group_by(period) %>%
  #summarize( n= n(), max=max(rating), min=min(rating),median=median(rating),mean=mean(rating)
  ggplot(aes(period, rating)) + geom_boxplot() + geom_smooth(aes(x=as.numeric(period)))
#The mean seems fairly constant over time so this doesn't seem an important factor


#Look at how user ratings (in aggregate) change after they rate their first movie
user_t <- edx %>% group_by(userId) %>% 
  mutate(minstamp = min(timestamp), user_age = round((timestamp - minstamp)/(182*86400),0)) %>% 
  ungroup()  %>%  group_by(user_age) %>% summarize(avg_rat = mean(rating), n = n()) 
user_t_plot <- user_t %>% ggplot(aes(user_age,avg_rat)) + geom_point() + geom_text(aes(label=n)) + 
  geom_smooth() + theme(axis.text.x = element_text(angle=90, vjust=.5, hjust=1))
user_t_plot
#The ratings only change significantly with very long tenure, but the count is so low as to be immaterial

#Look at how movie ratings (in aggregate) vary after first rating
#First look at ratings for movies released only after first rating date, base_date above. Should be largest effect
edx <- edx %>% mutate(rev_yr = year(as_datetime(timestamp)), mov_yr = as.numeric(str_match(edx$title, "\\((\\d{4})\\)$")[,2]),
               age = factor(rev_yr - mov_yr)) 
mu_t <- edx %>% group_by(age) %>% summarize(avg=mean(rating), n = n()) 
mu_t %>% ggplot(aes(age,avg)) + geom_point() + geom_text_repel(aes(label=n)) +
  theme(axis.text.x = element_text(angle=90, vjust=.5, hjust=1))
#There is pretty strong variation of the average review with elapsed time since movie release so we can consider as a 
#replacement for the overall average in our regularized model.
#Note that a loess fit to the train_set data does not converge in ~5 min so stratified averages are used

#Use the train set to create the mean as f(age) measures
mu_t <- train_set %>% group_by(age) %>% summarize(avg=mean(rating), n = n()) 

y_hat_a <-  test_set %>% left_join(mu_t, by = "age") %>% pull(avg)
head(y_hat_a)
RMSE(test_set$rating, rep(mean(train_set$rating),length.out = length(test_set$rating))) #1.060702
RMSE(test_set$rating, y_hat_a) # 1.051932, pretty minor improvement actually

#Genre effect
# Genre intuitively seems like it should matter a lot by user, but sample size (ratings/user) and many genres / film
# complicate this user approach. Explore use of the genre variable as a broad grouping
length(unique(edx$genres)) #797
edx %>% group_by(genres) %>% summarize(n=n()) %>% ggplot(aes(n)) + geom_histogram() + scale_x_log10()
edx %>% group_by(genres) %>% summarize(n=n()) %>% arrange(n) %>% head() 
#as few as 2 ratings by genre, e.g "Action|Animation|Comedy|Horror"
#Perhaps focus on the top 50 to see how impactful it might be on RMSE
edx %>% group_by(genres) %>% summarize(n=n(), rating=mean(rating)) %>% slice_max(order_by = n, n=50) %>%
      ggplot(aes(rating,genres)) + geom_point()
#Large amount of variability there, so let's add in the genre mean as the second effect
#Regularization addresses genres with very small sample size, shrinking those coefficients

#Seems logical to take away means from least specific to most so now proposing:
# Y(u,g,i) =  mu + b(g) + b(i) + b(u) with regularization. Note still static mu, will test mu_t next

#In reviewing class video professor says we shouldn't use the test set in cross validation lambda selection. 
#We'll use the train_set only to do 5-fold cross validation
#Note: bootstrapped w/replacement doesn't seem to make sense here when a user/movie review wouldn't occur >1x
set.seed(7, sample.kind="Rounding")
train_folds <- createFolds(y = train_set$rating, k=5, list = TRUE, returnTrain = FALSE)
sum(sapply(train_folds,function(x){length(x)}))
length(train_set$rating) #lengths match
#check if folds overlap
sum(train_folds[[1]] %in% train_folds[[2]])

# Looks independent, now five passes to hold out one fold for testing RMSE at each lambda
# Each pass is going to create an rmse for each lambda, which we're then going to average to final lambda/rmse metric
# Confirm we see local minima in this range and rerun if needed
lambdas <- seq(3, 7, 0.25)
#define a dataframe where cols are the lambas and rows are the fold iterations
fold_rmses <- data.frame(matrix(data=0,nrow = 0, ncol = length(lambdas))) #the zero creates numeric rather than logical
colnames(fold_rmses) <- lambdas #we'll rbind vector of rmses ordered by lambda to this dataframe

for(i in 1:5){ #sets up the iteration through folds
  #build the loop train/test sets
  l_train_set <- train_set[-train_folds[[i]]]
  temp <- train_set[train_folds[[i]]]
  l_test_set <- temp %>%
    semi_join(l_train_set, by = "age") %>%
    semi_join(l_train_set, by = "genres") %>%
    semi_join(l_train_set, by = "movieId") %>%
    semi_join(l_train_set, by = "userId")
  # Add rows removed from test set back into train set
  removed <- anti_join(temp, l_test_set)
  l_train_set <- rbind(l_train_set, removed)
  #check if these inner variables remain and need to be removed: temp, removed
  rmses <- sapply(lambdas, function(l){ 
    mu <- mean(l_train_set$rating)
    #mu_t <- l_train_set %>% group_by(age) %>% summarize(mu_t = mean(rating)) 
    
    b_g <- l_train_set %>%
      #left_join(mu_t, by = "age") %>%
      group_by(genres) %>%
      summarize(b_g = sum(rating - mu)/(n()+l))
    
    b_i <- l_train_set %>%
      #left_join(mu_t, by = "age") %>%
      left_join(b_g, by = "genres") %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu - b_g)/(n()+l))
    
    b_u <- l_train_set %>% 
      #left_join(mu_t, by = "age") %>%
      left_join(b_g, by = "genres") %>%
      left_join(b_i, by = "movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - mu - b_g - b_i)/(n()+l))
    
    predicted_ratings <- 
      l_test_set %>% 
      #left_join(mu_t, by = "age") %>%
      left_join(b_g, by = "genres") %>%
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = mu + b_g + b_i + b_u) %>%
      .$pred
    return(RMSE(predicted_ratings, l_test_set$rating))
  }) #sapply
  fold_rmses <- rbind(fold_rmses,rmses) #record the vector of rmses for lambda values in this fold
} #for i=1:5
#reassign column namnes
colnames(fold_rmses) <- lambdas
#next average the RMSEs by lambda
cv_rmses <- colMeans(fold_rmses)
#plot RMSE/lambda and select min rmse lamda value
qplot(lambdas,cv_rmses)
min(cv_rmses) #0.8664078, note this actually degrades to 0.8668541 with mu_t approach so stick static mu
cv_lambda <- lambdas[which.min(cv_rmses)] 
cv_lambda #5
# calculate the bs for entire train set and run on test for out of sample
mu <- mean(train_set$rating)
  
b_g <- train_set %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu)/(n()+cv_lambda))
  
b_i <- train_set %>%
    left_join(b_g, by = "genres") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_g)/(n()+cv_lambda))
  
b_u <- train_set %>% 
    left_join(b_g, by = "genres") %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_g - b_i)/(n()+cv_lambda))
  
#predictions on test set and rmse
predicted_ratings <- 
    test_set %>% 
    left_join(b_g, by = "genres") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_g + b_i + b_u) %>%
    .$pred

test_rmse <- RMSE(predicted_ratings, test_set$rating)
test_rmse #0.8659816 without mu_t

# The actual ratings are a "factor" and only exist in increments from 0.5 to 5.0
unique(factor(edx$rating))
# Levels: 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5
# Check the RMSE if we map the predicted rating (continuous) to the discrete available ratings
pred_rate_fact <- round(predicted_ratings * 2, 0)/2
unique(factor(pred_rate_fact))
#Levels: -0.5 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6
#Interesting we learn that predicted ratings were outside the possible bounds!
#So two things to check, the RMSE of predicted ratings when truncated to lie on allowed interval, and then rounded version
predicted_ratings <- ifelse(predicted_ratings > 5 , 5, predicted_ratings)
predicted_ratings <- ifelse(predicted_ratings < 0.5 , 0.5, predicted_ratings)
test_rmse <- RMSE(predicted_ratings, test_set$rating)
test_rmse #0.8658729
#Now look at impact of predicting only on 0.5 to 5.0 factors appearing in data set
pred_rate_fact <- round(predicted_ratings * 2, 0)/2
unique(factor(pred_rate_fact))
test_rmse <- RMSE(pred_rate_fact, test_set$rating)
test_rmse #0.8778657, so this actually makes RMSE worse and we won't use

#calculate the b's on entire edx set for this lambda (best prediction for validation set)
mu <- mean(edx$rating)

b_g <- edx %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu)/(n()+cv_lambda))

b_i <- edx %>%
  left_join(b_g, by = "genres") %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu - b_g)/(n()+cv_lambda))

b_u <- edx %>% 
  left_join(b_g, by = "genres") %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_g - b_i)/(n()+cv_lambda))

#save parameters for prediction cases
#save(list=c("mu", "b_g", "b_i", "b_u"), file = "coeff.RData")
load(file="coeff.RData")

#Now look at variation in the residuals and ability to model with SVD
#First transform edx to residuals by subtracting the predictions from each rating
edx_err <- edx %>% 
  left_join(b_g, by = "genres") %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_g + b_i + b_u, err = rating-pred) 

#Use train/set sets to check the out of sample RMSE impact of SVD fit on residuals
train_err <- edx_err[-test_index]
y_train <- train_err %>%
  select(userId, movieId, err) %>%
  spread(movieId, err) %>%
  as.matrix()
#rm(edx,edx_err,train_err)
rownames(y_train)<- y_train[,1]
y_train <- y_train[,-1]
y_train <- sweep(y_train, 1, rowMeans(y_train, na.rm=TRUE))
y_train <- sweep(y_train, 2, colMeans(y_train, na.rm=TRUE))
y_train[is.na(y_train)] <- 0
pca_train <- prcomp_irlba(y_train, n = 10, retx = TRUE)
#save(list=c("pca_train","y_train"),file="pca_train.RData")
#load(file="pca_train.RData")
#The "x" matrix is our user rows rotated into 10 PCs, assign rownames for prediction lookup
rownames(pca_train$x) <- rownames(y_train)
#The "rotation" matrix is the movie cols cast into 10 PCs, assign colnames for prediction lookup
rownames(pca_train$rotation) <- colnames(y_train)
#look at cumulative error explained and see if diminishing by 10
str(pca_train)
summary(pca_train) #10 PCs seems a reasonable cutoff
#got vector memory error trying to mutate predicted ratings to access by user/title so reform data to join
#Do we recover y_train? Or in this case the top 10 drivers of variation?
y_train_hat <- pca_train$x  %*% t(pca_train$rotation) #matrix in same shape as original errors matrix with values for each
rm(y_train)
pca_train_df <- data.frame(userId = rownames(y_train_hat), y_train_hat) #makes userId explicit column, setup for gather()  
save(list=c("pca_train_df"),file="pca_train_df.RData") # creating df takes long time, flattening matrix to rows
rm(pca_train,train_set)
colnames(pca_train_df) <- c("userId",colnames(y_train_hat))
pca_train_df[0:10,0:10]
pca_train_df <- gather(pca_train_df, key = "movieId", value = "pca_rat_adj", -userId) #many rows since values for all row x col 
pca_train_df <- pca_train_df %>% mutate(userId = as.numeric(userId), movieId = as.numeric(movieId))
save(list=c("pca_train_df"),file="pca_train_df.RData")
load(file="pca_train_df.RData")

pred_rating_pca_test <- test_set %>% 
  left_join(b_g, by = "genres") %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(pca_train_df, by = c("userId","movieId")) %>%
  mutate(pred = mu + b_g + b_i + b_u + pca_rat_adj)
#Above takes a very long time to run. Attempt to pre-filter _df with %in% on user/movieIds from test crashed
#save(list=c("pred_rating_pca_test"),file="pred_rating_pca_test.RData")

RMSE(pred_rating_pca_test$pred,test_set$rating) #0.8392949, nice improvement in RMSE!


#Full edx svd for final validation prediction
#Now transform to a matrix with user rows and movie columns where values are the errors for SVD
movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

y <- edx_err %>%
  select(userId, movieId, err) %>%
  spread(movieId, err) %>%
  as.matrix()

rownames(y)<- y[,1]
y <- y[,-1]
max(as.numeric(rownames(y)))
#[1] 71567
max(as.numeric(colnames(y)))
#[1] 65133
#colnames(y) <- with(movie_titles, title[match(colnames(y), movieId)]) #should stick with movieId here

getwd()
#save(list = "y",file = "errMatrix.RData")
load(file="errMatrix.RData")
#check row and column means. Col means should be removed for SVD/PCA else those are first PCs
max(abs(rowMeans(y, na.rm=TRUE))) #0.7705635
max(abs(colMeans(y, na.rm=TRUE))) #2.60409

y <- sweep(y, 1, rowMeans(y, na.rm=TRUE))
y <- sweep(y, 2, colMeans(y, na.rm=TRUE))
max(abs(rowMeans(y, na.rm=TRUE))) #0.1049228
max(abs(colMeans(y, na.rm=TRUE))) #1.110223e-16

#remove the many nas
y[is.na(y)] <- 0
#y <- sweep(y, 1, rowMeans(y))
#save(list = "y",file = "errMatAdj.RData")
load(file="errMatAdj.RData")

max(colMeans(y)) # 7.955555e-18
max(rowMeans(y)) # 7.191214e-18
#perform principal components
#pca <- prcomp(y, rank=5, scale=T) #does not return, assuming matrix too large. Asked on discussion
#Try the svd function instead
#svd(y) #ran for over two hours, no return

pca <- prcomp_irlba(y, n = 10, retx = TRUE)#, center = TRUE, scale. = TRUE)
#save(list = "pca",file = "pca.RData")
load(file="pca.RData")

str(pca)
dim(pca$x) #69878     10 , users
length(unique(edx$userId))#69878
dim(pca$rotation) #10677     10
length(unique(edx$movieId)) #10677

#The "x" matrix is our user rows rotated into 10 PCs, assign rownames for prediction lookup
rownames(pca$x) <- rownames(y)
#The "rotation" matrix is the movie cols cast into 10 PCs, assign colnames for prediction lookup
rownames(pca$rotation) <- colnames(y)

#Do we recover y? Or in this case the top 10 drivers of variation?
y_hat <- pca$x  %*% t(pca$rotation)
dim(y_hat) #69878 10677
str(y_hat)
sqrt(mean(y_hat^2)) #0.02282237, doesn't seem much of the overall edx_err RMSE but check impact on predict RMSE

#Recall we defined err = rating - prediction, so we would add error (or portion of error from pca)
# back to prediction to get closer to rating. Predictions are in a vector corresponding to edx data.table
#gather the matrix back to data.table format and join to existing predictions on user&movie
str(y_hat) #still matrix with userId as character for rownames and movieId as col names (char)

#got vector memory error trying to mutate predicted ratings to access by user/title so reform data to join
pca_err_tab <- data.frame(userId = rownames(y_hat), y_hat) #makes userId explicit column, setup for gather()  
#save(list = "pca_err_tab",file = "pca_err_tab.RData")

str(pca_err_tab)
colnames(pca_err_tab) <- c("userId",colnames(y_hat))
pca_err_tab[0:10,0:10]
pca_err_tab <- gather(pca_err_tab, key = "movieId", value = "pca_rat_adj", -userId) #many rows since values for all row x col 
pca_err_tab <- pca_err_tab %>% mutate(userId = as.numeric(userId), movieId = as.numeric(movieId))
#save(list = "pca_err_tab",file = "pca_err_long.RData")
load(file="pca_err_long.RData")

#######FINAL VALIDATION###############
length(unique(validation$userId)) #68534
length(unique(validation$movieId)) #9809
#Looking for > 0.86490 for maximum credit. Achieved with addition of genre, regularization
pred_rating_with_genres <- 
  validation %>% 
  left_join(b_g, by = "genres") %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_g + b_i + b_u) %>%
  .$pred

test_rmse <- RMSE(pred_rating_with_genres, validation$rating)
test_rmse #0.8648003

#Final model includes PCA on residuals and does even better on RMSE
pred_rating_pca_v <- validation %>% 
  left_join(b_g, by = "genres") %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(pca_err_tab, by = c("userId","movieId")) %>%
  mutate(pred = mu + b_g + b_i + b_u + pca_rat_adj) %>% .$pred

save(list="pred_rating_pca_v", file="pred_rating_pca_v.RData")
load(file="pred_rating_pca_v.RData")

pred_rating_pca_v <- ifelse(pred_rating_pca_v > 5 , 5, pred_rating_pca_v)
pred_rating_pca_v <- ifelse(pred_rating_pca_v < 0.5 , 0.5, pred_rating_pca_v)

RMSE(pred_rating_pca_v,validation$rating) #0.8409254, without outlier trim 0.841082

