library(ggplot2)
library(RColorBrewer)
mytheme <- theme(
  plot.title = element_text(
    face = "bold.italic", hjust = 0.5,
    size = "18", color = "brown"
  ),
  axis.title = element_text(
    face = "bold.italic", size = 12,
    color = "brown"
  ),
  axis.text = element_text(
    face = "bold", size = 11,
    color = "black"
  ),
  panel.background = element_rect(
    fill = "white",
    color = "black"
  ),
  panel.grid.major.y = element_line(
    color = "grey",
    linetype = 2
  ),
  panel.grid.minor.y = element_line(
    color = "grey",
    linetype = 2
  ),
  panel.grid.minor.x = element_blank()
)

scale_colour_discrete <- function(...) {
  scale_colour_manual(..., values = brewer.pal(n = 8, name = "Dark2"))
}
df <- read.csv("real_data/all_interpolation.csv")
subject_list <- sample(max(df$subject_ID), 5)

df_new <- df[which(df$subject_ID %in% subject_list), ]

ggplot(df_new) +
    geom_line(aes(y = bs_interpolation, x = time_point, color = factor(subject_ID)), size = 1.5) +
    geom_point(aes(y = real_data_without_mask, x = time_point, color = factor(subject_ID)), shape = 18, size = 5) +
    geom_point(aes(y = real_data_with_mask, x = time_point, color = factor(subject_ID)), shape = 9, size = 7.5) +
    facet_wrap("species_ID", ncol = 5, scales = "free_y") +
    xlab("Months") +
    ylab("Values") +
    scale_colour_discrete(
        name = "Different subjects",
        breaks = subject_list,
        labels = c("Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5")
    ) +
    ggtitle("Changes in microbial abundance of different subjects over time") +
    mytheme

ggsave("figure/B_spline.pdf", width = 50, height = 6, limitsize = FALSE)

ggplot(df_new) +
    geom_line(aes(y = dl_interpolation, x = time_point, color = factor(subject_ID)), size = 1.5) +
    geom_point(aes(y = real_data_without_mask, x = time_point, color = factor(subject_ID)), shape = 18, size = 5) +
    geom_point(aes(y = real_data_with_mask, x = time_point, color = factor(subject_ID)), shape = 9, size = 7.5) +
    facet_wrap("species_ID", ncol = 5, scales = "free_y") +
    xlab("Months") +
    ylab("Values") +
    scale_colour_discrete(
        name = "Different subjects",
        breaks = subject_list,
        labels = c("Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5")
    ) +
    ggtitle("Changes in microbial abundance of different subjects over time") +
    mytheme

ggsave("figure/DL_spline.pdf", width = 50, height = 6, limitsize = FALSE)
