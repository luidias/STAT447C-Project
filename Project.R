## Stat 447C Final Project

# dependencies
require(data.table) # for fread
require(dplyr)      # for data wrangling
require(rstan)
require(ggplot2)
require(bayesplot)  # for trace plots

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# global declare for file paths - change these to point to the appropriate file
data_folder = "X:/Oxland/Current Members/Luis/Grad Studies/Courses/STAT 447C/STAT 447 C BAYESIAN STATS/Project/STAT447C-Project/Data/Test Data"
test_info_file_path = "X:/Oxland/Current Members/Luis/Grad Studies/Courses/STAT 447C/STAT 447 C BAYESIAN STATS/Project/STAT447C-Project/Data/Shear Intrinsic Factors (simplified).csv"
full_stan_model_path = "C:/Users/luidias/Downloads/Stat 447 project temp dir/ProjectFull.stan"
infer_stan_model_path = "C:/Users/luidias/Downloads/Stat 447 project temp dir/ProjectFullWithInference.stan"

# -------------------------- Importing test data -------------------------------
# This is slow - instead of running this section, consider loading LoadedData.Rdata into the environment.

# fetch a list of the data files
files = list.files(
  path = data_folder,
  pattern = "*.csv",
  full.names = TRUE,
  recursive = FALSE,
)

# load the data into a list. This takes a long time, only run it once.
disp_data <- list()
force_data <- list()
for (i in 1:length(files)){
  x = fread(files[i],select = "pos_x_mm",header = TRUE) # displacement
  y = fread(files[i],select = "Fx_N",header = TRUE) # force
  test_name = substring(files[i],129,145) # this is the "H1234_C67_Ant_Med" part of the filename
  
  disp_data[[test_name]] <- x
  force_data[[test_name]] <- y
  
  print(i) # to monitor progress
}




# ---------------------- Test Data wrangling and cleanup -----------------------

# convert data to list of n x 1 matrices so I can use decimate() on them (it doesn't work if they're vectors)
disp_data_matrices = lapply(disp_data,as.matrix)
force_data_matrices = lapply(force_data,as.matrix)

# low-rate data is enormous and can therefore be down-sampled for speed
downsamp_lowdata = function(lis,dsFactor){
  newList <- list()
  for (i in 1:length(lis)){
    rate = substring(names(lis[i]),15,17) # TODO: change to something that doesn't depend on char index
    
    if (toString(rate) == "Low"){
      # downsample
      ds_data = lis[[i]][seq(1,length(lis[[i]]),dsFactor)]
      newList[[names(lis[i])]]<- ds_data
      
    } else {
      # don't change sampling, just convert to vector
      newList[[names(lis[i])]]<-as.vector(lis[[i]])
    }
  }
  return(newList) # note that this is list is now a list of vectors.
}

disp_data_ds = downsamp_lowdata(disp_data_matrices,10)
force_data_ds = downsamp_lowdata(force_data_matrices,10)
  
# get lengths of all vectors as well as max length
data_lengths = lengths(disp_data_ds)
max_length = max(data_lengths)

# pad the data with infs so they are all the same length as the longest vector
pad_data_infs = function(lis, desired_length){
  for (i in 1:length(lis)){
    curr_length = length(lis[[i]])
    pad_length = desired_length-curr_length
    lis[[i]] <- c(lis[[i]],rep(Inf,pad_length))
  }
  return(lis)
}

disp_data_padded = pad_data_infs(disp_data_ds,max_length)
force_data_padded = pad_data_infs(force_data_ds,max_length)

# convert data to matrices for feeding into stan. Transpose to keep it in 'tidy' format.
disp_data_matrix = t(matrix(unlist(disp_data_padded),ncol = length(disp_data_padded)))
force_data_matrix = t(matrix(unlist(force_data_padded),ncol = length(force_data_padded)))

# normalize all of the displacements so that they start from zero
disp_data_matrix <- disp_data_matrix-disp_data_matrix[,1]

# --------------------- test parameters pre-processing -------------------------

# load the datasheet with the test parameter info
test_info_df = read.csv(test_info_file_path,)

# reorder the test parameters to match order of our data
row_indices <- c()
for (i in 1:length(files)){
  row_indices = c(row_indices,match(test_info_df$fileList[i],substring(files,129,nchar(files)-4)))
}
test_info_df <- cbind(test_info_df, data.frame(indices = row_indices))
test_info_df = arrange(test_info_df,row_indices)

# convert non-numeric variables to centered numeric values
test_info_df$direction[test_info_df$direction == "Ant"] <- 1
test_info_df$direction[test_info_df$direction == "Pos"] <- -1
test_info_df$sex[test_info_df$sex == "F"] <- 1
test_info_df$sex[test_info_df$sex == "M"] <- -1  # sex as 1, -1 to keep data centered
test_info_df[,c(3,6)] <- sapply(test_info_df[,c(3,6)],as.numeric)

# save pre-scaled dataframe 
test_info_df_raw = test_info_df

# log-transform and scale rate, since it's in orders of magnitude (1,10,100 mm/s)
test_info_df$disp_rate_mm_s <- scale(log(test_info_df$disp_rate_mm_s))
test_info_df <- rename(test_info_df,log_disp_rate = disp_rate_mm_s) # rename to make sure I don't forget

# scale everything else that we're going to use and isn't already centered
test_info_df$age_yrs <- scale(test_info_df$age_yrs)
test_info_df$avg_vert_body_treb_bv_fraction <- scale(test_info_df$avg_vert_body_treb_bv_fraction)
test_info_df$Disc_Degen_Score <- scale(test_info_df$Disc_Degen_Score)
test_info_df$FJ_Degen_Score <- scale(test_info_df$FJ_Degen_Score)


# ----------------------------Final Cleanup Steps -----------------------------
# important: flip the polarity of all posterior tests. This is needed because
# posterior data is currently all negative, which makes the overall curve look
# more like a 3rd-order than a quadratic. Flipping the polarity brings all of
# the data into the same quadrant and allows it to be modeled as a quadratic 
# curve.
for (i in 1:nrow(disp_data_matrix)){
  # since direction is encoded 1, -1, we can just multiply the data directly by the direction
  disp_data_matrix[i,] <- disp_data_matrix[i,] * test_info_df$direction[i]
  force_data_matrix[i,] <- force_data_matrix[i,] * test_info_df$direction[i]
}

# The model currently can't handle missing values (only one FSU in this dataset, luckily)
na_rows <- which(rowSums(is.na(test_info_df))>0)  # get row with missing data
test_info_df <- test_info_df[-na_rows,]           # remove from parameters dataset and from test data vectors
rownames(test_info_df) <- NULL
disp_data_matrix <- disp_data_matrix[-na_rows,]
force_data_matrix <- force_data_matrix[-na_rows,]
data_lengths <- data_lengths[-na_rows]
test_info_df_raw <- test_info_df_raw[-na_rows,]

# convert test params df to matrix and drop unwanted rows
test_info_matrix = data.matrix(test_info_df[,2:9]) # note that first column (ID) has been converted to indices


# ------------------- Model prototyping and basic LOO fitting ------------------

# prototyping (model design stage)
num_tests = 60 # only use some of the tests, for prototyping. FSUs are in multiples of 6
fit_proto = stan(
  file = full_stan_model_path,
  data = list(
    N = max_length,
    K = nrow(disp_data_matrix[1:num_tests,]),
    y = force_data_matrix[1:num_tests,], 
    x = disp_data_matrix[1:num_tests,],
    L = data_lengths[1:num_tests],
    t_pars = test_info_matrix[1:num_tests,],
    S = max(test_info_matrix[1:num_tests,1]),
    subject_ids = as.integer(test_info_matrix[1:num_tests,1])
  ), 
  chains = 1,
  iter = 1000,
  control=list(max_treedepth = 15)
)

# prior and posterior predictive check
# for prior: go into stan and comment out the y_k ~ normal(mu_k, sigma) line
# for posterior, leave the same line uncommented.
# IMPORTANT: This Posterior check is pseudo-leave-one-out (trained on whole dataset)
num_tests = 228
set.seed(2)
shear_model = stan_model(full_stan_model_path)
fit_pp = sampling(
  shear_model,
  data = list(
    N = max_length,
    K = nrow(disp_data_matrix[1:num_tests,]),
    y = force_data_matrix[1:num_tests,], 
    x = disp_data_matrix[1:num_tests,],
    L = data_lengths[1:num_tests],
    t_pars = test_info_matrix[1:num_tests,],
    S = max(test_info_matrix[1:num_tests,1]),
    subject_ids = as.integer(test_info_matrix[1:num_tests,1])
  ), 
  chains = 2,
  iter = 200,
  control=list(max_treedepth = 15)
)

samples_pp <- extract(fit_pp)
y_pp <- samples_pp$y_pp # 3D array [iteration, test number, data index]

# pseudo-LOO check using built-in methods
check_calib(unname(y_pp),force_data_matrix) # unname y_pp for compatibility with my check_calib function

# check for slow mixing
mcmc_trace(fit_pp, pars = c("a_all")) + theme_minimal()


# visualizing - avoid using this for very large numbers of iterations, it's slow
disp_plotting <- replace(disp_data_.matrix,is.infinite(disp_data_matrix),NA)     # can't have infs otherwise range() will break
force_plotting <- replace(force_data_matrix,is.infinite(force_data_matrix),NA)
plot(NA,xlab='Displacement',ylab='load',main = 'Prior Predictive Check',xlim=range(disp_plotting[1:num_tests,],na.rm = TRUE),ylim=range(force_plotting[1:num_tests,],na.rm = TRUE))
for (i in seq_len(nrow(disp_plotting[1:num_tests,]))){
  lines(disp_plotting[i,1:data_lengths[i]],force_plotting[i,1:data_lengths[i]],col=2) # actual test data
  for (j in 1:nrow(y_pp)){
    lines(disp_plotting[i,1:data_lengths[i]],y_pp[j,i,1:data_lengths[i]],col=rgb(0,0,0,0.01)) # simulated data
  }
}


# ------------------------- Calibration Check ----------------------------------

# funcion for checking calibration from posterior predictive

check_calib <- function(y_pp,true_force_data_matrix){
  # this function simply returns the calculated coverage, but can be modified to return the intermediate matrices as well.
  
  if (length(dim(y_pp))==3){
    # for multiple tests in one output
    y_pp_df <- as.data.frame.table(y_pp, responseName = "y_pred") %>%
      rename(iter = Var1, test= Var2, index = Var3) %>%
      mutate(iter = as.integer(iter),
             test = as.integer(test),
             index = as.integer(index))
    
    # generate 95% CIs for every index in each test across all iterations
    summary_df = y_pp_df %>%
      group_by(test,index) %>%
      summarise(
        y_lower = quantile(y_pred,.025),
        y_upper = quantile(y_pred,.975),
        y_median = median(y_pred),
        .groups = "drop"
      )
    
    # get original data into long form to match y_pp_df
    true_force_df <- as.data.frame.table(true_force_data_matrix,responseName="y_true") %>%
      rename(test = Var1, index = Var2) %>%
      mutate(test = as.integer(test),
             index = as.integer(index),
             y_true = ifelse(is.infinite(y_true),NA,y_true))
    
    
    # merge both dataframes
    calib_check_df <- left_join(summary_df,true_force_df,by=c("test","index"))
    
  } else if(length(dim(y_pp))==2) {
    # for single test inference
    y_pp_df <- as.data.frame.table(y_pp, responseName = "y_pred") %>%
      rename(iter = Var1,index = Var2) %>%
      mutate(iter = as.integer(iter),
             index = as.integer(index))
    
    # generate 95% CIs for every index in each test across all iterations
    summary_df = y_pp_df %>%
      group_by(index) %>%
      summarise(
        y_lower = quantile(y_pred,.025),
        y_upper = quantile(y_pred,.975),
        y_median = median(y_pred),
        .groups = "drop"
      )
    
        calib_check_df <- mutate(summary_df,y_true=as.data.frame(true_force_data_matrix))
  }
  
  
  # add a column with a logical indicating whether the true value was in the CI
  calib_check_df <- mutate(calib_check_df, in_CI = (y_true >= y_lower & y_true <= y_upper))
  #print(calib_check_df)
  # calculate the coverage
  coverage <- mean(calib_check_df$in_CI,na.rm = TRUE)
  return(coverage)
  #return(calib_check_df)
}
# --------------------------- True LOO using Stan ------------------------------
# This is extremely slow due to heavy used of generated_quantities. It's better
# to use the custom inference tools further down.

# subset the data for speed
num_tests = 60
disp_subset <- disp_data_matrix[1:num_tests,]
force_subset <- force_data_matrix[1:num_tests,]
data_len_subset <- data_lengths[1:num_tests]
test_info_subset <- test_info_matrix[1:num_tests,]


coverages_LOO <- c()

for (k in 1:nrow(disp_subset)){
  
  # setup all of the data
  train_test_data = list(
    N = max_length,
    K = nrow(disp_subset[-k,]),
    y = force_subset[-k,], 
    x = disp_subset[-k,],
    L = data_len_subset[-k],
    t_pars = test_info_subset[-k,],
    S = max(test_info_subset[-k,1]),
    subject_ids = as.integer(test_info_subset[-k,1]),
    P = data_len_subset[k],
    x_infer = disp_subset[k,1:data_len_subset[k]],
    t_pars_pred = test_info_subset[k,2:8]
  )
  
  # fit model, with prediction
  fit_LOO = stan(
    file = infer_stan_model_path,
    data = train_test_data,
    chains = 4,
    iter = 100,
    control = list(max_treedepth =15)
  )
  
  # check coverage for this loop
  samples_LOO = extract(fit_LOO)
  y_pred_LOO = samples_LOO$y_pred
  covg = check_calib(unname(y_pred_LOO),force_subset[k,1:data_len_subset[k]])
  print(covg) #for debugging
  coverages_LOO <- c(coverages_LOO,covg)
}

# overall coverage
print(mean(coverages_LOO))


# --- custom stuff (everything below this is experimental, still needs work) ---
# ------------- Custom inference engine and corridor plotter -------------------
# this is an alternative to performing prediction within stan, because it's very
# slow to do this in generated quantities. This simulator replicates the 
# prediction logic I used in the stan code, but using much faster R code.
# IF THE MODEL STRUCTURE IS CHANGED IN STAN, DON'T FORGET TO CHANGE THIS TOO.

simulate_force <- function(x_k, t_pars_k, iter_i, posterior_samples, subj_id_k=NULL){
  # simulates the load vector for one test (k) and a specific posterior sample(i)
  # If no subject ID is specified, the prediction is done for a new (unobserved) subject
  # if a subject ID is specified, we assume the subject was in the training set and re-use that intercept.
  
  # extract parameters from the specified iteration
  a_all <- posterior_samples$a_all[iter_i]
  a_dir <- posterior_samples$a_dir[iter_i]
  a_rate <- posterior_samples$a_rate[iter_i]
  a_age <- posterior_samples$a_age[iter_i]
  a_sex <- posterior_samples$a_sex[iter_i]
  a_bv <- posterior_samples$a_bv[iter_i]
  a_dd <- posterior_samples$a_dd[iter_i]
  a_fjd <- posterior_samples$a_fjd[iter_i]
  
  b_all <- posterior_samples$b_all[iter_i]
  b_dir <- posterior_samples$b_dir[iter_i]
  b_rate <- posterior_samples$b_rate[iter_i]
  b_age <- posterior_samples$b_age[iter_i]
  b_sex <- posterior_samples$b_sex[iter_i]
  b_bv <- posterior_samples$b_bv[iter_i]
  b_dd <- posterior_samples$b_dd[iter_i]
  b_fjd <- posterior_samples$b_fjd[iter_i]
  
  sigma <- posterior_samples$sigma[iter_i]
  
  # extract or generate intercepts
  if (!is.null(subj_id_k) && subj_id_k>0){
    # subject id was specified - re-use the intercepts for that subject
    a_int <- posterior_samples$a_intercepts[iter_i,subj_id_k]
    b_int <- posterior_samples$b_intercepts[iter_i,subj_id_k]
  } else {
    # simulate intercepts for an unobserved subject (we still use the SD from the posterior samples)
    a_int <- rnorm(1,mean = 0,sd = posterior_samples$sig_a[iter_i])
    b_int <- rnorm(1,mean = 0,sd = posterior_samples$sig_b[iter_i])
  }
  
  mu_k <- (a_all +
             a_dir * t_pars_k[1] + a_rate * t_pars_k[2] +
             a_age * t_pars_k[3] + a_sex * t_pars_k[4] +
             a_bv  * t_pars_k[5] + a_dd  * t_pars_k[6] + a_fjd * t_pars_k[7] +
             a_int) * x_k^2 +
          (b_all +
             b_dir * t_pars_k[1] + b_rate * t_pars_k[2] +
             b_age * t_pars_k[3] + b_sex * t_pars_k[4] +
             b_bv  * t_pars_k[5] + b_dd  * t_pars_k[6] + b_fjd * t_pars_k[7] +
             b_int) * x_k
  y_pred <- rnorm(length(x_k), mean=mu_k, sd=sigma)
  return(y_pred)
  
}

predict_test <- function(x_k, t_pars_k, posterior_samples, subj_id_k=NULL, iter_indices = NULL){
  # given a displacement vector and test parameters, loops over posterior samples
  # to generate a prediction for the specified test conditions. This can then be used 
  # to generate corridors, check calibration, etc.
  # IMPORTANT: T-PARS MUST HAVE RESCALED DATA.
  
  # t_pars must be a vector of (dir, log(rate),age,sex,bv,dd,fjd). Note that we have moved subject ID out of this vector.
  
  # by default, use all posterior samples. Can use only some of them for speed.
  if(is.null(iter_indices)){
    iter_indices <- 1:length(posterior_samples$a_all)
  }
  
  # initialize the output matrix: rows are iterations, columns are datapoints.
  n_iters <- length(iter_indices)
  n_datapoints <- length(x_k)
  y_pred_matrix <- matrix(NA,nrow=n_iters,ncol=n_datapoints)
  
  # loop over each iteration
  for (i in 1:n_iters){
    iter_i <- iter_indices[i] # note that indices don't have to start at 1
    y_pred_matrix[i,] <- simulate_force(
      x_k = x_k,
      t_pars_k = t_pars_k,
      subj_id_k = subj_id_k,
      iter_i = iter_i,
      posterior_samples = posterior_samples
    )
  }
  
  return(y_pred_matrix)
}

# plotter function
plot_prediction <- function(y_pred_matrix,x_k,y_true = NULL,title = "Posterior Force Prediction"){
  # this function plots the median response of a predicted test, along with a 95% corridor
  # y_true is an optional argument for comparing the prediction with the true vector, for LOO testing
  
  # calculate median and quantiels at each datapoint
  corridors <- apply(y_pred_matrix, 2, function(x) {
    c(median = median(x),
      lower = quantile(x, 0.025, na.rm = TRUE),
      upper = quantile(x, 0.975, na.rm = TRUE))
  })
  # convert to dataframe, tidy form
  corridors_df <- as.data.frame(t(corridors))
  colnames(corridors_df) <- c("median", "lower", "upper")
  corridors_df$x <- x_k
  
  # plot the corridors
  p<- ggplot(corridors_df,aes(x=x))+
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = 'skyblue',alpha = 0.4) +
    geom_line(aes(y=median),color = "blue",size =1) +
    labs(title = title,
         x = "Shear displacement, mm",
         y = "Shear force, N") +
    theme_minimal()
  
  # overlay true data if it was provided
  if (!is.null(y_true)){
    true_df <- data.frame(x = x_k, y = y_true)
    p <- p +geom_line(data=true_df, aes(x=x, y=y), color = "red", size = 1, linetype = "dashed")
  }
  
  return(p)
}
# ------------------------------ synthetic data --------------------------------
generate_synthetic_data <- function(K = 30, S = 10, N = 100) {
  
  subject_ids <- sample(1:S, K, replace = TRUE)
  
  # test parameters - all covars except sex and dir are sacled and centered.
  t_pars <- matrix(NA, K, 8)
  for (k in 1:K) {
    t_pars[k, ] <- c(
      k,                                # ID
      sample(c(-1, 1), 1),              # dir
      rnorm(1), rnorm(1),               # log(rate), age
      sample(c(-1, 1), 1),              # sex
      rnorm(1), rnorm(1), rnorm(1)      # BMD, DD, FJD
    )
  }
  
  # Sample parameters from priors
  a_priorMu = 40
  a_priorVar = 10
  b_priorMu = 30
  b_priorVar = 10
  a_all <- rnorm(1, a_priorMu, a_priorVar)
  a_dir <- rnorm(1, a_priorMu, a_priorVar)
  a_rate <- rnorm(1, a_priorMu, a_priorVar)
  a_age <- rnorm(1, a_priorMu, a_priorVar)
  a_sex <- rnorm(1, a_priorMu, a_priorVar)
  a_bv <- rnorm(1, a_priorMu, a_priorVar)
  a_dd <- rnorm(1, a_priorMu, a_priorVar)
  a_fjd <- rnorm(1, a_priorMu, a_priorVar)
  b_all <- rnorm(1, b_priorMu, b_priorVar)
  b_dir <- rnorm(1, b_priorMu, b_priorVar)
  b_rate <- rnorm(1, b_priorMu, b_priorVar)
  b_age <- rnorm(1, b_priorMu, b_priorVar)
  b_sex <- rnorm(1, b_priorMu, b_priorVar)
  b_bv <- rnorm(1, b_priorMu, b_priorVar)
  b_dd <- rnorm(1, b_priorMu, b_priorVar)
  b_fjd <- rnorm(1, b_priorMu, b_priorVar)
  
  sigma <- rexp(1, 1)
  sig_a <- rexp(1, 1)
  sig_b <- rexp(1, 1)
  z_a <- rnorm(S)
  z_b <- rnorm(S)
  a_intercepts <- z_a * sig_a
  b_intercepts <- z_b * sig_b
  
  x <- y <- matrix(0, K, N)
  L <- integer(K)
  
  for (k in 1:K) {
    Lk <- sample(60:100, 1)
    L[k] <- Lk
    xk <- seq(0, 2, length.out = Lk)
    subj <- subject_ids[k]
    t <- t_pars[k, ]
    
    quad_a <- a_all + a_dir*t[2] + a_rate*t[3] + a_age*t[4] + a_sex*t[5]+ a_bv*t[6]+ a_dd*t[7]+ a_fjd*t[8] + a_intercepts[subj]
    quad_b <- b_all + b_dir*t[2] + b_rate*t[3] + b_age*t[4] + b_sex*t[5]+ b_bv*t[6]+ b_dd*t[7]+ b_fjd*t[8] + b_intercepts[subj]
    
    mu <- quad_a * xk^2 + quad_b * xk
    yk <- rnorm(Lk, mu, sigma)
    
    x[k, 1:Lk] <- xk
    y[k, 1:Lk] <- yk
  }
  
  list(N = N, K = K, S = S, x = x, y = y, L = L, t_pars = t_pars, subject_ids = subject_ids)
}

synth_data = generate_synthetic_data()

# check model on synthetic data
fit_synth = stan(
  file = full_stan_model_path,
  data = synth_data, 
  chains = 1,
  iter = 200,
  control=list(max_treedepth = 15)
)

samples_synth <- extract(fit_synth)
y_synth <- samples_synth$y_pp # 3D array [iteration, test number, data index]

# pseudo-LOO check using built-in methods
check_calib(unname(y_synth),synth_data$y) # unname y_pp for compatibility with my check_calib function


# ---------------- True leave-one-subject-out testing --------------------------
# Checks coverage by leaving one subject out of the training dataset.
# This is a more thorough test than the one I implemented into the stan model.
# This is also SLOW. it refits every time, so minimize the subset size and 
# number of iterations.

loso_check <- function(model,disp_matrix,force_matrix,data_lengths,test_info,train_tests,num_loops,iters){
  test_subset_rows <- sample(1:nrow(test_info),train_tests) # random draw of subset from test data, without replacement
  unique_ids <- unique(test_info[test_subset_rows,1]) # fetch a list of the subject IDs from the sampled rows
  loop_ids <- unique_ids[1:num_loops] # IDs to do LOSO on. if num_loops=train_tests, it'll LOO over all the tests in train_tests, but it'll take a long time.
  
  # Pre-allocate predictions and true values
  max_N <- max(data_lengths[test_subset_rows])
  y_pp_array <- array(0, dim = c(iters, length(test_subset_rows), max_N))
  y_true_matrix <- matrix(NA, nrow = length(test_subset_rows), ncol = max_N)
  
  # used for indexing later
  pp_test_counter <- 1
  
  # loop through individual subjects
  for (s in loop_ids){
    test_ids <- test_subset_rows[test_info[test_subset_rows,1] ==s]  # rows that contains tests for this subject 
    train_ids <- test_subset_rows[test_info[test_subset_rows,1] !=s] # all other rows in the subset
    
    # fit the model
    fit = sampling(
      model,
      data = list(
        N = ncol(disp_matrix),
        K = length(train_ids),
        y = force_matrix[train_ids,], 
        x = disp_matrix[train_ids,],
        L = data_lengths[train_ids],
        t_pars = test_info[train_ids,],
        S = length(unique(test_info[train_ids,1])),
        subject_ids = as.integer(factor(test_info[train_ids,1]))
      ), 
      chains = 1,
      iter = iters,
      control=list(max_treedepth = 15)
    )
    
    posterior <- extract(fit)
    
    # predict on the left-out tests
    for (k in test_ids) {
      x_k <- disp_matrix[k, 1:data_lengths[k]]
      t_k <- test_info[k, 2:8]
      
      preds <- predict_test(
        x_k = x_k,
        t_pars_k = t_k,
        posterior_samples = posterior
      )
      
      y_pp_array[, pp_test_counter, 1:data_lengths[k]] <- preds
      y_true_matrix[pp_test_counter, 1:data_lengths[k]] <- force_matrix[k, 1:data_lengths[k]]
      
      pp_test_counter <- pp_test_counter + 1
    }
  }
  
  # calculate coverage using my custom function
  coverage <- check_calib(y_pp_array, y_true_matrix)
  
  # choose what you want to return - comment the other out.
  #return(list(y_pp = y_pp_array,y_true = y_true_matrix))
  return(coverage)
}

# precompile model
shear_model <- stan_model(infer_stan_model_path)

# perform loso test (be wary of large subsets and iterations!)
set.seed(1)
loso_coverage <- loso_check(shear_model,disp_data_matrix,force_data_matrix,data_lengths,test_info_matrix,228,3,200)


# -------------------------- Proof of concept inference ------------------------

# fit the model on the full dataset
fit_full = stan(
  file = full_stan_model_path,
  data = list(
    N = max_length,
    K = nrow(disp_data_matrix),
    y = force_data_matrix, 
    x = disp_data_matrix,
    L = data_lengths,
    t_pars = test_info_matrix,
    S = max(test_info_matrix[,1]),
    subject_ids = as.integer(test_info_matrix[,1])
  ), 
  chains = 1,
  iter = 1000,
  control=list(max_treedepth = 15)
)

y_full<-extract(fit_full)

# pseudo-LOO check using built-in methods (sanity check)
check_calib(unname(y_full$y_pp),force_data_matrix) 
mcmc_trace(fit_full, pars = c("a_all")) + theme_minimal() 


## subject parameters
s1_dir = -1 
s1_rate = log(10) # change number in the log only
s1_age = 70
s1_sex = 1        # M:-1, F:1
s1_bv = 0.6
s1_dd = 4  
s1_fjd = 2
s1_x = seq(from=0,to=2,by=0.01) # disp vector

s1_pars = c(s1_dir,s1_rate,s1_age,s1_sex,s1_bv,s1_dd,s1_fjd)

# rescale to match our training data. test_info_df_raw was stored during preprocessing
raw_means <- c(
  NA,
  mean(log(test_info_df_raw$disp_rate_mm_s)),
  mean(test_info_df_raw$age_yrs),
  NA,
  mean(test_info_df_raw$avg_vert_body_treb_bv_fraction),
  mean(test_info_df_raw$Disc_Degen_Score),
  mean(test_info_df_raw$FJ_Degen_Score)
)

raw_sds <- c(
  NA,
  sd(log(test_info_df_raw$disp_rate_mm_s)),
  sd(test_info_df_raw$age_yrs),
  NA,
  sd(test_info_df_raw$avg_vert_body_treb_bv_fraction),
  sd(test_info_df_raw$Disc_Degen_Score),
  sd(test_info_df_raw$FJ_Degen_Score)
)

normalize_pars<- function(t_pars_raw, means, sds, scaled_idx) {
  t_scaled <- t_pars_raw
  t_scaled[scaled_idx] <- (t_pars_raw[scaled_idx] - means[scaled_idx]) / sds[scaled_idx]
  return(t_scaled)
}

s1_pars_scaled <- normalize_pars(s1_pars,raw_means,raw_sds,c(2,3,5,6,7))

# predict at various rates (anterior shear only)
s1_1mm <- predict_test(s1_x,s1_pars_scaled,y_full)
print(plot_prediction(s1_1mm,s1_x))
