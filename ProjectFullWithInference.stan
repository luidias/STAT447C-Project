// Biomechanical Spine Segments in Shear
// Stat 447C Final Project
// Luis Dias
// April 2025

// This version of the model is functionally identical to ProjectFull.stan, but 
// also includes input variables and genereted quantities code for performing 
// inference on a  novel test. This is slow, and it's better to use the custom 
// simulator in the main project R file.

data {
  
  // raw data
  int<lower=0> N; // number of observations in each vector
  int<lower=0> K; // number of tests
  matrix[K, N] y; // Shear force data
  matrix[K, N] x; // Shear displacement data
  int<lower = 1, upper = N> L[K]; // X and Y data is ragged so I padded it for stan - this vector contains the actual data lengths for each test.
  
  // test parameters (predictors)
  matrix[K, 8] t_pars;                 // test parameter matrix. Columns are [ID, dir, log(rate), age, sex, BMD, DD, FJD]
  int<lower=0> S;                      // number of subjects
  int<lower=1,upper=S> subject_ids[K]; // corresponding subject ID for each test
  
  // // data for inference - comment out if not in use.
  // int <lower=0> P;     // length of displacement data for inference
  // vector[P] x_infer; //displacement to data to infer over
  // vector[7] t_pars_pred; // test covariates. MUST BE SCALED THE SAME AS TRAINING DATA. no ID column: [dir, log(rate), age, sex, BMD, DD, FJD]
}

parameters {
  // LMM terms for first and second-order parameters of the quadratic regression
  // 2nd-order params
  real a_all;   //"overall" second order term without interactions from covars
  real a_dir;
  real a_rate;
  real a_age;
  real a_sex;
  real a_bv;
  real a_dd;
  real a_fjd;
  // 1st-order params
  real b_all;
  real b_dir;
  real b_rate;
  real b_age;
  real b_sex;
  real b_bv;
  real b_dd;
  real b_fjd;
  
  real<lower=0> sigma;  // measurement noise. global for now, change if needed.
  
  // params for hierarchical model of random intercepts
  vector[S] z_a;       // normalized z scores (see model block)
  vector[S] z_b;
  real<lower=0> sig_a; // scaling parameters for z scores
  real<lower=0> sig_b;
  
 }

transformed parameters{
  
 vector[S] a_intercepts = z_a *sig_a; // vector of random intercepts for the second-order term, one per subject
 vector[S] b_intercepts = z_b *sig_b; // same as above, for first-order term

}

model {
  
  // prior parameters - note that these will be repeated 9 times per term
  real a_priorMu = 35;
  real a_priorVar = 15;
  real b_priorMu = 35;
  real b_priorVar = 15;
   
  // priors
  a_all ~ normal (a_priorMu,a_priorVar);
  a_dir ~ normal (a_priorMu,a_priorVar);
  a_rate ~ normal (a_priorMu,a_priorVar);
  a_age ~ normal (a_priorMu,a_priorVar);
  a_sex ~ normal (a_priorMu,a_priorVar);
  a_bv ~ normal (a_priorMu,a_priorVar);
  a_dd ~ normal (a_priorMu,a_priorVar);
  a_fjd ~ normal (a_priorMu,a_priorVar);
  b_all ~ normal (b_priorMu,b_priorVar);
  b_dir ~ normal (b_priorMu,b_priorVar);
  b_rate ~ normal (b_priorMu,b_priorVar);
  b_age ~ normal (b_priorMu,b_priorVar);
  b_sex ~ normal (b_priorMu,b_priorVar);
  b_bv ~ normal (b_priorMu,b_priorVar);
  b_dd ~ normal (b_priorMu,b_priorVar);
  b_fjd ~ normal (b_priorMu,b_priorVar);
  sigma ~ exponential(1);
  
  // priors for hierarchical subject intercepts
  z_a ~ normal(0,1);   // centered at zero and std dev of 1
  z_b ~ normal(0,1);
  sig_a ~ exponential(1); // hyperprior scaling factor for intercept
  sig_b ~ exponential(1);
  
  //attempting to speed things up
  matrix[K, 7] X_test = t_pars[, 2:8];  // predictors only
  vector[K] a_fixed = a_all + X_test * to_vector([a_dir, a_rate, a_age, a_sex, a_bv, a_dd, a_fjd]');
  vector[K] b_fixed = b_all + X_test * to_vector([b_dir, b_rate, b_age, b_sex, b_bv, b_dd, b_fjd]');


  //likelihood (vectorized in slices, one per test)
  for (k in 1:K) {
    
    // slice out the tests's x and y data, excluding any padding
    vector[L[k]] x_k = to_vector(x[k, 1:L[k]]); // displacement for test k, using only indices up to the padding (1:L[k])
    vector[L[k]] y_k = to_vector(y[k, 1:L[k]]); // force for same test
    
    // This is slower because it calculates a gajillion dot products
    // mu as a quadratic regression using all parameters. "." is element-wise multiplication
    // vector[L[k]] mu_k = (a_all 
    //                       + a_dir*t_pars[k,2] + a_rate*t_pars[k,3]                  // mechanical factors (test-specific)
    //                       + a_age*t_pars[k,4] + a_sex*t_pars[k,5]                   // demographic factors
    //                       + a_bv*t_pars[k,6] + a_dd*t_pars[k,7] + a_fjd*t_pars[k,8] // anthropometric factors
    //                       + a_intercepts[subject_ids[k]])                           // random effects for the subjects
    //                       * x_k .* x_k 
    //                   + (b_all 
    //                       + b_dir*t_pars[k,2] + b_rate*t_pars[k,3] 
    //                       + b_age*t_pars[k,4] + b_sex*t_pars[k,5]
    //                       + b_bv*t_pars[k,6] + b_dd*t_pars[k,7] + b_fjd*t_pars[k,8] 
    //                       + b_intercepts[subject_ids[k]]) 
    //                       * x_k; 
    //                       
    vector[L[k]] mu_k = (a_fixed[k] + a_intercepts[subject_ids[k]]) * x_k .* x_k
                  + (b_fixed[k] + b_intercepts[subject_ids[k]]) * x_k;

    // likelihood calculation. COmment this out when doing prior predictive, leave it in for (basic) posterior predictive
    y_k ~ normal(mu_k, sigma);
  }

}
// 
// generated quantities{
//   // comment this out if not in use, for speed.
//   
//   // prediction for a single test.
// 
//   // only use one of these pairs
//   real a_int = normal_rng(0, sig_a); // true new intercept
//   real b_int = normal_rng(0, sig_b);
//   //real a_int = a_intercepts[subject_ids[k]]; // reuse subject intercepts for leave-one-test out
//   //real b_int = b_intercepts[subject_ids[k]];
// 
//   vector[P] mu_pred = (a_all
//   + a_dir*t_pars_pred[1] + a_rate*t_pars_pred[2]
//   + a_age*t_pars_pred[3] + a_sex*t_pars_pred[4]
//   + a_bv*t_pars_pred[5] + a_dd*t_pars_pred[6] + a_fjd*t_pars_pred[7]
//   + a_int)
//   * x_infer .* x_infer
//   + (b_all
//   + b_dir*t_pars_pred[1] + b_rate*t_pars_pred[2]
//   + b_age*t_pars_pred[3] + b_sex*t_pars_pred[4]
//   + b_bv*t_pars_pred[5] + b_dd*t_pars_pred[6] + b_fjd*t_pars_pred[7]
//   + b_int)
//   * x_infer;
// 
//   // simulate y
//   vector[P] y_pred = to_vector(normal_rng(mu_pred,sigma));
// 
// }

