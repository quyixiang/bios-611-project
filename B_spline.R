library("ggplot2")
library("tibble")
library("tidyr")
library("dplyr")
library("mgcv")

file_list <- list.files("split_data")

B_spline <- function(df, species_ID, subject_ID) {
    info_df <- df %>% filter(real_data_without_mask > 0)
    gp_data <- tibble(y = unname(info_df$real_data_without_mask), x = info_df$time)
    m_bs_default <- gam(y ~ s(x, k = 6, bs = "bs", m = c(3, 2)),
        method = "REML",
        data = gp_data
    )
    new_data <- tibble(x = unname(df$time))
    p_bs <- as_tibble(predict(m_bs_default, new_data))
    real_data <- df %>%
        replace_na(list(real_data_unavailable = 0, real_data_undetectable = 0)) %>%
        mutate(real_data_with_mask = real_data_unavailable + real_data_undetectable) %>%
        select(real_data_with_mask, real_data_without_mask) %>%
        mutate(real_data_with_mask = na_if(real_data_with_mask, 0))
    all_data <- bind_cols(new_data, real_data, p_bs, tibble(dl_interpolation = unname(df$data))) %>% rename(bs_interpolation = value, time_point = x)
    all_data$species_ID <- species_ID
    all_data$subject_ID <- subject_ID

    return(all_data)
}


all_data <- as_tibble()
for (i in (1:length(file_list))) {
    temp_df <- read.csv(paste("split_data/", file_list[i], sep = ""))[, -1]
    for (species_ID in (0:max(temp_df["species"]))) {
        temp_df_species <- temp_df %>% filter(species == species_ID)
        all_data <- bind_rows(all_data, B_spline(temp_df_species, species_ID, i))
    }
}

write.csv(all_data, "real_data/all_interpolation.csv", row.names = FALSE)