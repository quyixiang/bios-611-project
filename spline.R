# From: https://fromthebottomoftheheap.net/2020/06/03/extrapolating-with-gams/

## Packages
library("ggplot2")
library("tibble")
library("tidyr")
library("dplyr")
library("mgcv")
library("gratia")
library("patchwork")
library("colorblindr")
library("relayer")
try_data <- read.csv("data.csv")
gp_data <- tibble(truth = unname(try_data$y), x = try_data$x / 30) %>%
  mutate(y = try_data$y + rnorm(length(truth), 0, 0.2))

r_samp <- sample_n(gp_data, size = 28) %>%
  arrange(x) %>%
  mutate(data_set = case_when(
    x < -0.8 ~ "test",
    x > 0.8 ~ "test",
    x > -0.45 & x < -0.36 ~ "test",
    x > -0.05 & x < 0.05 ~ "test",
    x > 0.45 & x < 0.6 ~ "test",
    TRUE ~ "train"
  ))

ggplot(r_samp, aes(x = x, y = y, colour = data_set)) +
  geom_line(aes(y = truth, colour = NULL), show.legend = FALSE, alpha = 0.5) +
  geom_point() +
  scale_colour_brewer(palette = "Set1", name = "Data set")

# thin plate regression spline
m_tprs2 <- gam(y ~ s(x, k = 10, bs = "tp", m = 2),
  data = filter(r_samp, data_set == "train"), method = "REML"
)
## first order penalty
m_tprs1 <- gam(y ~ s(x, k = 10, bs = "tp", m = 1),
  data = filter(r_samp, data_set == "train"), method = "REML"
)

new_data <- tibble(x = seq(-0.1, 1.1, by = 0.002))
p_tprs2 <- as_tibble(predict(m_tprs2, new_data, se.fit = TRUE)) %>%
  rename(fit_tprs_2 = fit, se_tprs_2 = se.fit)
p_tprs1 <- as_tibble(predict(m_tprs1, new_data, se.fit = TRUE)) %>%
  rename(fit_tprs_1 = fit, se_tprs_1 = se.fit)
crit <- qnorm((1 - 0.89) / 2, lower.tail = FALSE)
new_data_tprs <- bind_cols(new_data, p_tprs2, p_tprs1) %>%
  pivot_longer(fit_tprs_2:se_tprs_1,
    names_sep = "_",
    names_to = c("variable", "spline", "order")
  ) %>%
  pivot_wider(names_from = variable, values_from = value) %>%
  mutate(upr_ci = fit + (crit * se), lwr_ci = fit - (crit * se))

ggplot(mapping = aes(x = x, y = y)) +
  geom_ribbon(
    data = new_data_tprs,
    mapping = aes(
      ymin = lwr_ci, ymax = upr_ci, x = x,
      fill = order
    ),
    inherit.aes = FALSE, alpha = 0.2
  ) +
  geom_point(data = r_samp, aes(colour = data_set)) +
  geom_line(
    data = new_data_tprs, aes(y = fit, x = x, colour2 = order),
    size = 1
  ) %>%
  rename_geom_aes(new_aes = c("colour" = "colour2")) +
  scale_colour_brewer(
    palette = "Set1", aesthetics = "colour",
    name = "Data set"
  ) +
  scale_colour_OkabeIto(aesthetics = "colour2", name = "Penalty") +
  scale_fill_OkabeIto(name = "Penalty") +
  # coord_cartesian(ylim = c(-2, 2)) +
  labs(
    title = "Extrapolating with thin plate splines",
    subtitle = "How behaviour varies with derivative penalties of different order"
  )
ggsave("figure/tprs.pdf", width = 10, height = 6)

# B spline
m_bs_default <- gam(y ~ s(x, k = 10, bs = "bs", m = c(3, 2)),
  data = filter(r_samp, data_set == "train"), method = "REML"
)

knots <- list(x = c(-2, -0.9, 0.9, 2))

m_bs_extrap <- gam(y ~ s(x, k = 10, bs = "bs", m = c(3, 2)),
  method = "REML",
  data = filter(r_samp, data_set == "train"), knots = knots
)

p_bs_default <- as_tibble(predict(m_bs_default, new_data, se.fit = TRUE)) %>%
  rename(fit_bs_default = fit, se_bs_default = se.fit)
p_bs_extrap <- as_tibble(predict(m_bs_extrap, new_data, se.fit = TRUE)) %>%
  rename(fit_bs_extrap = fit, se_bs_extrap = se.fit)

new_data_bs_eg <- bind_cols(new_data, p_bs_default, p_bs_extrap) %>%
  pivot_longer(fit_bs_default:se_bs_extrap,
    names_sep = "_",
    names_to = c("variable", "spline", "penalty")
  ) %>%
  pivot_wider(names_from = variable, values_from = value) %>%
  mutate(upr_ci = fit + (crit * se), lwr_ci = fit - (crit * se))

ggplot(mapping = aes(x = x, y = y)) +
  geom_ribbon(
    data = new_data_bs_eg,
    mapping = aes(ymin = lwr_ci, ymax = upr_ci, x = x, fill = penalty),
    inherit.aes = FALSE, alpha = 0.2
  ) +
  geom_point(data = r_samp, aes(colour = data_set)) +
  geom_line(
    data = new_data_bs_eg, aes(y = fit, x = x, colour2 = penalty),
    size = 1
  ) %>%
  rename_geom_aes(new_aes = c("colour" = "colour2")) +
  scale_colour_brewer(palette = "Set1", aesthetics = "colour", name = "Data set") +
  scale_colour_OkabeIto(aesthetics = "colour2", name = "Penalty") +
  scale_fill_OkabeIto(name = "Penalty") +
  # coord_cartesian(ylim = c(-2, 2)) +
  labs(
    title = "Extrapolating with B splines",
    subtitle = "How behaviour varies when the penalty extends beyond the data"
  )
ggsave("figure/B_spline.pdf", width = 10, height = 6)