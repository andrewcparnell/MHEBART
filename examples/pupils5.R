# 3 different ways to fit a cross random effects model
# This model doesn't standardise the data so uses an overall mean for each tree
# This one uses more trees (set to 4 for simplicity)
# 1. Using lme4
# 2. Using JAGS
# 3. Using raw R code for MCMC
# All of these should agree (especially 2 and 3 as they're Bayesian)
# If not we're in trouble

# NOTE: still a small bug in this one

# Packages
library(tidyverse)
library(lme4)
library(R2jags)
library(mvnfast)
library(LaplacesDemon) # For half t prior
devtools::load_all(".")

load("data/pupils.Rdata")
# pupils

# 1. lmer -----------------------------------------------------------------

pupils_lmer <- lmer(
  achievement ~ 
    + (1 | primary_school_id)
    + (1 | secondary_school_id),
  data = pupils
)

pplme <- predict(pupils_lmer, pupils)
summary(pupils_lmer)

# pupils |>
#   mutate(pred = pplme) |>
#   group_by(primary_school_id, secondary_school_id) |>
#   summarise(y = mean(achievement),
#             pred = mean(pred),
#             n = n()) |>
#   View()

# 2. Using JAGS -----------------------------------------------------------

# Define the model
model_string <- "
model {
  for (i in 1:N) {
    y[i] ~ dnorm(fit[i], tau)
      fit[i] = tree1[group1[i]] + tree2[group1[i]] + tree3[group2[i]] + tree4[group2[i]]
  }
  for (j in 1:len_group1) {
    tree1[j] ~ dnorm(mu[1], sigma[1]^-2)
  }
  for (j in 1:len_group1) {
    tree2[j] ~ dnorm(mu[2], sigma[1]^-2)
  }
  for (j in 1:len_group2) {
    tree3[j] ~ dnorm(mu[3], sigma[2]^-2)
  }
  for (j in 1:len_group2) {
    tree4[j] ~ dnorm(mu[4], sigma[2]^-2)
  }
  
  for (k in 1:4) {
    mu[k] ~ dnorm(0, 10^-2) # Both scales should probably be set better
  }
  sigma[1] ~ dt(0, 10^-2, 1)T(0,)
  sigma[2] ~ dt(0, 10^-2, 1)T(0,)
  tau ~ dgamma(0.01, 0.01)

}"

# Compile the model
model_data <- list(
  y = pupils$achievement,
  group1 = pupils$primary_school_id,
  group2 = pupils$secondary_school_id,
  N = nrow(pupils),
  len_group1 = length(unique(pupils$primary_school_id)),
  len_group2 = length(unique(pupils$secondary_school_id))
)

# Run the model
model <- jags(
  model.file = textConnection(model_string),
  data = model_data,
  parameters.to.save = c(
    "mu", "sigma", "tree1", "tree2", "tree3", "tree4",  
    "fit", "tau"
  )
)
# plot(model)

# pupils |>
#   mutate(pred_JAGS = as.numeric(model$BUGSoutput$mean$fit),
#          pred_lmer = pplme) |>
#   group_by(primary_school_id, secondary_school_id) |>
#   summarise(y = mean(achievement),
#             pred_JAGS = mean(pred_JAGS),
#             pred_lmer = mean(pred_lmer),
#             n = n()) |>
#   View() # Almost identical fits
# stop()

# 3. MCMC code ------------------------------------------------------------

# Model is y_i ~ N(fit[i], tau^-1)
# fit[i] = tree1[group1[i]] + tree2[group1[i]] + tree3[group2[i]]  + tree4[group2[i]]
# tree1[j] ~ N(mu1, sigma1^2)
# tree2[k] ~ N(mu2, sigma1^2)
# tree3[k] ~ N(mu3, sigma2^2)
# tree4[k] ~ N(mu4, sigma2^2)
# mu[.] ~ N(0, a^2), a = 10
# sigma[.] ~ dt(0, 10, 1)+
# tau ~ Ga(0.01, 0.01)

# Let M1 and M2 be the group allocation matrices
# Can write as y ~ N(M1%*%tree1 + M1%*%tree2 + M2%*%tree3 + M2%*%tree4, I/tau)

# So need parameter updates for tree1, tree2,... tau (all Gibbs)
# And sigma1, sigma2 both MH

# Set up everything
set.seed(123)
num_iter <- 1000
y <- pupils$achievement
M1 <- stats::model.matrix(~ as.factor(pupils$primary_school_id) - 1)
M2 <- stats::model.matrix(~ as.factor(pupils$secondary_school_id) - 1)

# Some useful things needed later
tM1xM1 <- crossprod(M1)
tM2xM2 <- crossprod(M2)
n <- length(y)
a = 1/10

# Starting values
tau <- model$BUGSoutput$mean$tau[1] 
sigma1 <- model$BUGSoutput$mean$sigma[1]
sigma2 <- model$BUGSoutput$mean$sigma[2]
mu1 <- model$BUGSoutput$mean$mu[1]
mu2 <- model$BUGSoutput$mean$mu[2] 
mu3 <- model$BUGSoutput$mean$mu[3]
mu4 <- model$BUGSoutput$mean$mu[4]
tree1 <- model$BUGSoutput$mean$tree1 #matrix(rep(0, ncol(M1)), ncol = 1)
tree2 <- model$BUGSoutput$mean$tree2 #matrix(rep(0, ncol(M1)), ncol = 1)
tree3 <- model$BUGSoutput$mean$tree3 #matrix(rep(0, ncol(M2)), ncol = 1)
tree4 <- model$BUGSoutput$mean$tree4 #matrix(rep(0, ncol(M2)), ncol = 1)

# Storage
storage <- list(
  tree1 = matrix(NA, ncol = ncol(M1), nrow = num_iter),
  tree2 = matrix(NA, ncol = ncol(M1), nrow = num_iter),
  tree3 = matrix(NA, ncol = ncol(M2), nrow = num_iter),
  tree4 = matrix(NA, ncol = ncol(M2), nrow = num_iter),
  tau = rep(NA, num_iter),
  mu1 = rep(NA, num_iter),
  mu2 = rep(NA, num_iter),
  mu3 = rep(NA, num_iter),
  mu4 = rep(NA, num_iter),
  sigma1 = rep(NA, num_iter),
  sigma2 = rep(NA, num_iter),
  fits = matrix(NA, ncol = n, nrow = num_iter)
)

# Stuff for MH
sigma1_sd <- sigma2_sd <- 1

# Progress bar
pb <- utils::txtProgressBar(
  min = 1, max = num_iter,
  style = 3, width = 60,
  title = "Running model..."
)

for (i in 1:num_iter) {
  utils::setTxtProgressBar(pb, i)
  
  # Storage update
  storage$tree1[i,] <- tree1
  storage$tree2[i,] <- tree2
  storage$tree3[i,] <- tree3
  storage$tree4[i,] <- tree4
  storage$tau[i] <- tau
  storage$mu1[i] <- mu1
  storage$mu2[i] <- mu2
  storage$mu2[i] <- mu3
  storage$mu2[i] <- mu4
  storage$sigma1[i] <- sigma1
  storage$sigma2[i] <- sigma2
  storage$fits[i,] <- M1 %*% tree1 + M1 %*% tree2 + M2 %*% tree3 + M2 %*% tree4
  
  # Update mu1 for tree 1
  R1 <- y - M1 %*% tree2 - M2 %*% tree3  - M2 %*% tree4 # Partial residuals for tree 1
  Psi <- diag(n)/tau + (sigma1^2) * tcrossprod(M1)
  prec <- sum(solve(Psi)) + a
  mu1 <- rnorm(1, colSums(solve(Psi, R1))/prec, sd = sqrt(1 / prec))

  # Update tree1
  prec <- tau * tM1xM1 + diag(1/(sigma1^2), ncol(M1))
  tree1 <- t(mvnfast::rmvn(1, solve(prec, tau * crossprod(M1, R1)), solve(prec)))

  # # Update mu2 for tree 2
  R2 <- y - M1 %*% tree1 - M2 %*% tree3  - M2 %*% tree4
  Psi <- diag(n)/tau + (sigma1^2) * tcrossprod(M1)
  prec <- sum(solve(Psi)) + a
  mu2 <- rnorm(1, colSums(solve(Psi, R2))/prec, sd = sqrt(1 / prec))

  # Update tree2
  prec <- tau * tM1xM1 + diag(1/(sigma1^2), ncol(M1))
  tree1 <- t(mvnfast::rmvn(1, solve(prec, tau * crossprod(M1, R2)), solve(prec)))
  
  # Update mu3 for tree 3
  R3 <- y - M1 %*% tree1 - M1 %*% tree2  - M2 %*% tree4
  Psi <- diag(n)/tau + (sigma2^2) *  tcrossprod(M2)
  prec <- sum(solve(Psi)) + a
  mu3 <- rnorm(1, colSums(solve(Psi, R3))/prec, sd = sqrt(1 / prec))
  
  # Update tree3
  prec <- tau * tM2xM2 + diag(1/(sigma2^2), ncol(M2))
  tree3 <- t(mvnfast::rmvn(1, solve(prec, tau * crossprod(M2, R3)), solve(prec)))

  # Update mu4 for tree 4
  R4 <- y - M1 %*% tree1 - M1 %*% tree2  - M2 %*% tree3
  Psi <- diag(n)/tau + (sigma2^2) *  tcrossprod(M2)
  prec <- sum(solve(Psi)) + a
  mu4 <- rnorm(1, colSums(solve(Psi, R4))/prec, sd = sqrt(1 / prec))

  # Update tree4
  prec <- tau * tM2xM2 + diag(1/(sigma2^2), ncol(M2))
  tree4 <- t(mvnfast::rmvn(1, solve(prec, tau * crossprod(M2, R4)), solve(prec)))
  
  # Update tau
  S <- sum((y - M1 %*% tree1 - M1 %*% tree2 - M2 %*% tree3 - M2 %*% tree4)^2)
  tau <- rgamma(1,
    shape = 0.01 + n / 2,
    rate = 0.01 + S / 2
  )

  # Update sigma1
  repeat {
    # Proposal distribution
    new_sigma1 <- sigma1 + stats::rnorm(1, sd = sigma1_sd)
    if (new_sigma1 > 0) {
      break
    }
  }
  log_rat <- stats::pnorm(sigma1, sd = sigma1_sd, log = TRUE) -
    stats::pnorm(new_sigma1, sd = sigma1_sd, log = TRUE)

  post_new <- sum(dnorm(tree1, mu1, new_sigma1, log = TRUE)) +
    sum(dnorm(tree2, mu2, new_sigma1, log = TRUE)) +
    dhalft(new_sigma1, scale = 10, nu = 1, log = TRUE)
  post_old <- sum(dnorm(tree1, mu1, sigma1, log = TRUE)) +
    sum(dnorm(tree2, mu2, sigma1, log = TRUE)) +
    dhalft(sigma1, scale = 10, nu = 1, log = TRUE)

  log_alpha <- post_new - post_old + log_rat

  accept <- log_alpha >= 0 || log_alpha >= log(stats::runif(1))
  sigma1 <- ifelse(accept, new_sigma1, sigma1)

  # Update sigma2
  repeat {
    # Proposal distribution
    new_sigma2 <- sigma2 + stats::rnorm(1, sd = sigma2_sd)
    if (new_sigma2 > 0) {
      break
    }
  }
  log_rat <- stats::pnorm(sigma2, sd = sigma2_sd, log = TRUE) -
    stats::pnorm(new_sigma2, sd = sigma2_sd, log = TRUE)

  post_new <- sum(dnorm(tree3, mu3, new_sigma2, log = TRUE)) +
    sum(dnorm(tree4, mu4, new_sigma2, log = TRUE)) +
    LaplacesDemon::dhalft(new_sigma2, scale = 10, nu = 1, log = TRUE)
  post_old <- sum(dnorm(tree3, mu3, sigma2, log = TRUE)) +
    sum(dnorm(tree4, mu4, sigma2, log = TRUE)) +
    LaplacesDemon::dhalft(sigma2, scale = 10, nu = 1, log = TRUE)

  log_alpha <- post_new - post_old + log_rat

  accept <- log_alpha >= 0 || log_alpha >= log(stats::runif(1))
  sigma2 <- ifelse(accept, new_sigma2, sigma2)
}

# Plot some of the outputs
plot(1/sqrt(storage$tau))
plot(storage$sigma1)
plot(storage$sigma2)

# Compare models ----------------------------------------------------------

pupils |>
  mutate(pred_JAGS = as.numeric(model$BUGSoutput$mean$fit),
         pred_MCMC = colMeans(storage$fits),
         pred_lmer = pplme) |>
  group_by(primary_school_id, secondary_school_id) |>
  summarise(y = mean(achievement),
            pred_JAGS = mean(pred_JAGS),
            pred_MCMC = mean(pred_MCMC),
            pred_lmer = mean(pred_lmer),
            n = n()) |>
  View() # Almost identical fits

sqrt(mean((pplme - pupils$achievement)^2)) # 0.7978818
sqrt(mean((model$BUGSoutput$mean$fit - pupils$achievement)^2)) # 0.797759
sqrt(mean((colMeans(storage$fits) - pupils$achievement)^2)) # 0.7974181


