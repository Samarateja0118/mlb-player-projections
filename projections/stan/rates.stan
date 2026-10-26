// Same model as projections/model.py, written in Stan.
// logit p[i, s] = theta[i, s] + aging(age) + season[s]; theta follows a random walk.
data {
  int<lower=1> N;                 // player-season observations
  int<lower=1> P;                 // players
  int<lower=2> S;                 // history seasons
  array[N] int<lower=1, upper=P> player;
  array[N] int<lower=1, upper=S> season_idx;
  array[N] int<lower=0> n;
  array[N] int<lower=0> k;
  vector[N] age_c;                // age - 27
  vector[P] target_age_c;
  real league_logit;
}
parameters {
  real mu;
  real<lower=0> sigma_player;
  real<lower=0> tau;
  vector[2] b_age;
  sum_to_zero_vector[S] season;
  vector[P] z0;
  matrix[P, S - 1] steps;
}
transformed parameters {
  matrix[P, S] walk;
  walk[, 1] = mu + sigma_player * z0;
  for (s in 2:S) walk[, s] = walk[, s - 1] + tau * steps[, s - 1];
}
model {
  mu ~ normal(league_logit, 0.5);
  sigma_player ~ normal(0, 1);
  tau ~ normal(0, 0.3);
  b_age ~ normal(0, 0.05);
  season ~ normal(0, 0.2);
  z0 ~ std_normal();
  to_vector(steps) ~ std_normal();
  {
    vector[N] eta;
    for (j in 1:N)
      eta[j] = walk[player[j], season_idx[j]] + b_age[1] * age_c[j] + b_age[2] * square(age_c[j])
               + season[season_idx[j]];
    k ~ binomial_logit(n, eta);
  }
}
generated quantities {
  vector[P] p_proj;
  for (i in 1:P) {
    real theta_next = normal_rng(walk[i, S], tau);
    p_proj[i] = inv_logit(theta_next + b_age[1] * target_age_c[i] + b_age[2] * square(target_age_c[i])
                          + season[S]);
  }
}
