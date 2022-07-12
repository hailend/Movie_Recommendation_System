#Movie Recommendation System
#Haile Endeshaw

##########################################################
# Create train set set, validation set (final hold-out test set)
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

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
#                                           title = as.character(title),
#                                           genres = as.character(genres))
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

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#--------------------------- Visualize data ------------------------------------
#Inspect data
movielens <- edx
movielens %>% as_tibble()
movielens %>%
  summarise(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))
#Inspect how sparse data is (take a sample of the data)
ind <- sample(nrow(movielens), 1000) #randomly sample 1000 indices
s_dat <- movielens[ind, ]
s_dat_wide <- pivot_wider(s_dat, names_from = title, values_from = rating)
as_tibble(s_dat_wide) #shows a lot of NAs -- since not every user rated every movie. 
#recommendation system is like filling those NAs
keep <- movielens %>%
  dplyr::count(movieId) %>%
  top_n(5) %>%
  pull(movieId)
tab <- movielens %>%
  filter(userId %in% c(13:20)) %>% 
  filter(movieId %in% keep) %>% 
  select(userId, title, rating) %>% 
  spread(title, rating)
tab %>% knitr::kable()

#plot to show how sparse the data is--- users vs movies --- lot of unrated movies (NAs)
users <- sample(unique(movielens$userId), 100)
rafalib::mypar()
movielens %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey") #adds grid

#histogram showing the number of ratings by movie (some movies get rated more often)
movielens %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 20, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

#histogram showing the number of ratings of each user (some users rate movies more often)
movielens %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")


#-------------------------------------------------------------------------------
#----------------------- Machine Learning ---------------------------------------
train_set <- edx
test_set <- validation
movielens %>%
  summarise(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

#Loss function: typical RSME on a test was used
#y_u_i --- actual rating (y) of user (u) of movie (i)
#y_hat_u_i ---- predicted rating (y) of user (u) of movie (i)
#RMSE = sqrt(sum((y_hat_u_i - y_u_i)^2)/N); --- N - # of movie user combinations
#RMSE -- similar to st. dev.
#if RMSE > 1; error is > one star --> not good

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# --------------------- First model (Just the Mean) ----------------------------
#recommend same rating for all users and all movies
#model -- Y_u_i = mu + e --- e= error; mu - true rating for all movies by all users
#use least squares to minimize RMSE
mu_hat <- mean(train_set$rating)
mu_hat
#RMSE
naive_RMSE <- RMSE(test_set$rating, mu_hat)
naive_RMSE
#results table
rmse_results <- tibble(method = "Just the average", RMSE = naive_RMSE)  
rmse_results %>% knitr::kable()

# --------------------  Modeling movie effects ---------------
#model -- Y_u_i = mu + bi + e ---- bi is average ranking for movie i --- bi's are called "effects"
#use least squares to find bi's (there are thousands of them)
#fit <- lm(rating ~ as.factor(movieId), data = movielens) --- don't run this, requires long time to run and big memory
#but we know that bi is the average of Y_u_i - mu_hat for each movie
mu <- mean(train_set$rating)
movie_avgs <- train_set %>%
  group_by(movieId) %>% 
  summarise(b_i = mean(rating - mu)) #mu is used in place of mu_hat
qplot(b_i, data = movie_avgs, bins = 35, color = I("black")) #since mu = 3.5, b = 1.5 implies a perfect 5 rating
#check prediction improvement: y_u_i_hat = mu_hat + b_i_hat
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i)             #This will join by movieId and adds mu to the b_i values --- since mu +  test_set
predicted_ratings %>%
  ggplot(aes(pred)) +
  geom_histogram(bins = 25, color = "black")
movie_rmse <- RMSE(test_set$rating, predicted_ratings$pred) #RMSE improved
movie_rmse
rmse_results <- bind_rows(rmse_results,
                          data.frame(method="Movie Effects",  
                                     RMSE = movie_rmse ))
rmse_results %>% knitr::kable()


# ------------------------- Modeling user effects -----------------------------
#compute the average rating for user u for those that rated 100 or more movies
train_set %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarise(y100 = mean(rating)) %>%
  ggplot(aes(y100)) + 
  geom_histogram(bins = 35, color = "black") #There is a big variability depending on the users. 
#some give good ratings to movies in general others don't
#modify model to: Y_u_i = mu + b_i + b_u + e_u_i
mu <- mean(train_set$rating)
#fit <- lm(rating ~ as.factor(movieId) + as.factor(userId), data = movielens) --- don't run this, requires long time to run and big memory
#but b_u  is the mean of Y_u_i - mu_hat - b_i
user_avgs <- train_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))
qplot(b_u, data = user_avgs, bins = 45, color = I("black"))

predicted_user_ratings <- test_set %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(userId) %>%
  mutate(pred = mu + b_u + b_i)
predicted_user_ratings %>%
  ggplot(aes(pred)) +
  geom_histogram(bins = 45, color = "Black")
user_rmse <- RMSE(predicted_user_ratings$pred, test_set$rating)  #lower RMSE --> better prediction
user_rmse
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = user_rmse ))
rmse_results %>% knitr::kable()

#------------------------------------------------
movielens <- mutate(movielens, date = as_datetime(timestamp))

movielens %>% 
  mutate(date = round_date(date, unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) + 
  geom_point() + 
  geom_smooth()

#-------------------
#effect of date parameter (the date the movie came out)
movielens %>%
  group_by(genres) %>%
  summarise(n = n(), 
            ave_rating = mean(rating), 
            se = sd(rating)/sqrt(n())) %>%
  filter(n > 50000) %>%
  arrange(ave_rating) %>%
  ggplot(aes(x = genres, y= ave_rating, ymin = ave_rating - 2*se, ymax = ave_rating + 2*se)) +
  geom_point() +
  geom_errorbar() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#-------------------------------------------------------------------------------
#-------------------------- REGULARIZATION -------------------------------------
#Regularization allows us to penalize large estimates that are formed using 
#only small sample sizes. 
#General idea of regularization - to constrain the total variability of the 
#effect sizes choosing penalty terms

#sort movies by rating 
train_set %>%
  count(movieId) %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>%
  select(title, n, b_i) %>%
  slice(1:10)

#It seems that they are not really popular movies. The reason they are at the top 
#is because they are rated by very few users who happened to give them a high rating  


lambdas <- seq(0, 10, 0.25)
mu <- mean(train_set$rating)
just_the_sum <- train_set %>%
  group_by(movieId) %>%
  summarize(s = sum(rating - mu), n_i = n())
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set %>%
    left_join(just_the_sum, by = 'movieId') %>%
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses)
lambdas[which.min(rmses)]

lambda <- 3
mu <- mean(train_set$rating)
movie_reg_avgs <- train_set %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu)/(n() + lambda), n_i = n())

#test
predicted_movie_reg_ratings <- test_set %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  group_by(movieId) %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

movie_reg_rmse <- RMSE(predicted_movie_reg_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method ="Movie Effects_regularized + User Effects",  
                                     RMSE = movie_reg_rmse))
rmse_results %>% knitr::kable()

#Top 10 best movies
movie_titles <- movielens %>%
  select(movieId, title) %>%
  distinct()

train_set %>%
  count(movieId) %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>%
  select(title, n, b_i) %>%
  slice(1:10)

#Top 10 worst movies
train_set %>%
  count(movieId) %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>%
  select(title, b_i, n) %>%
  slice(1:10)


#----add regularization for user effects -----------------------------
lambda <- 3
user_reg_avgs <- train_set %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu - b_i)/(n() + lambda), n_i = n())

#test
predicted_user_reg_ratings <- test_set %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(user_reg_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

user_reg_rmse <- RMSE(predicted_user_reg_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method ="Movie Effects_reg + User Effects reg",  
                                     RMSE = user_reg_rmse))
rmse_results %>% knitr::kable()

#-------------------------------------------------------------------------------
#---------------------------- matrix factorization ----------------------------
library(recosystem) #for faster results 
recos <- Reco()
set.seed(234, sample.kind = "Rounding")
reco_train_dat <- with(train_set, data_memory(user_index = userId, 
                                              item_index = movieId, 
                                              rating = rating))
recos$train(reco_train_dat)
