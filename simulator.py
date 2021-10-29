# %%
import random
import numpy as np
import math
from numpy import random
from scipy.stats import norm
# import matplotlib.pylab as plt
import pandas as pd

# %%
def ZILN(n, m, t, p, seed):
    sig2 = math.log(t + 1)
    mm = np.log(m) - sig2 / 2
    np.random.seed(seed)
    rvec = np.exp(np.random.normal(loc=mm, scale=math.sqrt(sig2), size=n))
    np.random.seed(seed)
    pvec = np.random.binomial(1, 1 - p, size=n)
    xvec = rvec * pvec
    return rvec, xvec


def brown(dt):
    dW = norm.rvs(scale=math.sqrt(dt))
    return dW


# %%
def CorSig_gen(n_feature, pi_s, pi_b, seed):
    np.random.seed(seed)
    X = np.triu(np.random.rand(n_feature ** 2).reshape(n_feature, n_feature))
    np.random.seed(seed)
    Y = np.triu(np.random.rand(n_feature ** 2).reshape(n_feature, n_feature))
    for i in range(n_feature):
        for j in range(n_feature):
            if i == j:
                X[i, j] = 0
            elif X[i, j] > pi_s:
                X[i, j] = 1

            elif X[i, j] > 0:
                X[i, j] = -1

    for i in range(n_feature):
        for j in range(n_feature):
            if i == j:
                Y[i, j] = 0
            elif Y[i, j] > pi_b:
                Y[i, j] = 1

            elif Y[i, j] > 0:
                Y[i, j] = 0

    Z = X * Y
    Z += Z.T - np.diag(Z.diagonal())
    return Z


seed = 10
sig_mat = CorSig_gen(5, 0.7, 0.4, seed)
# %%
def CorMat_gen(n_feature, mean, sd, sig_mat, seed):
    np.random.seed(seed)
    mat = np.abs(np.random.normal(mean, sd, n_feature ** 2).reshape(n_feature, -1))
    for i in range(n_feature):
        mat[i, i] = 0
    i_lower = np.tril_indices(n_feature, -1)
    mat[i_lower] = mat.T[i_lower]

    return mat * sig_mat


sample_mat = CorMat_gen(5, 0.1, 0.1, sig_mat, seed)

# %%
def RanMat_gen(n_feature, time_sd, seed):
    np.random.seed(seed)
    mat = np.random.normal(0, time_sd, n_feature ** 2).reshape(n_feature, -1)
    for i in range(n_feature):
        mat[i, i] = 0
    i_lower = np.tril_indices(n_feature, -1)
    mat[i_lower] = mat.T[i_lower]

    return mat


RanMat_gen(5, 0.0001, seed) + sample_mat
# %%
def dX_BS(X, k, sigma, delta_t):
    return k * X * delta_t + sigma * X * brown(delta_t)


def dX_OU(X, k, m, sigma, delta_t):
    return -k * (X - m) * delta_t + sigma * brown(delta_t)


# %%
def multi_time_trend_generator_without_time_variation(
    subject_num,
    time_point_num,
    n_feature,
    time_trend_class,
    BS_k,
    BS_sigma,
    OU_k,
    OU_m,
    OU_sigma,
    delta_t,
    start_value_min,
    start_value_max,
    cor_sd,
    time_sd,
    lambda_para_1,
    lambda_para_2,
    sig_mat,
    seed,
):
    x = np.zeros((subject_num, time_point_num, n_feature))

    for i in range(x.shape[0]):  # each subject
        seed = int(pow(seed + i * seed, 2 / (i + 1))) + i
        sample_mat = CorMat_gen(n_feature, 0.1, cor_sd, sig_mat, seed)
        for j in range(x.shape[1]):  # each time point
            cor_mat = sample_mat  # + RanMat_gen(n_feature, time_sd)
            BS_index = -1
            OU_index = -1
            for k in range(x.shape[2]):  # each feature
                if j > 0:
                    if time_trend_class[k] == "BS":
                        BS_index = BS_index + 1
                        dX = (
                            lambda_para_1
                            * (
                                dX_BS(
                                    x[i, (j - 1), k],
                                    BS_k[BS_index],
                                    BS_sigma[BS_index],
                                    delta_t,
                                )
                            )
                            + lambda_para_2 * np.dot(x[i, (j - 1), :], cor_mat[:, k])
                        )
                    else:
                        OU_index = OU_index + 1
                        dX = (
                            lambda_para_1
                            * (
                                dX_OU(
                                    x[i, (j - 1), k],
                                    OU_k[OU_index],
                                    OU_m[OU_index],
                                    OU_sigma[OU_index],
                                    delta_t,
                                )
                            )
                            + lambda_para_2 * np.dot(x[i, (j - 1), :], cor_mat[:, k])
                        )
                    x[i, j, k] = abs(x[i, j - 1, k] + dX)
                else:
                    np.random.seed(seed)
                    x[i, 0, k] = np.random.uniform(
                        start_value_min[k], start_value_max[k], 1
                    )[0]

    return x


# %%
# extract value and add time variation
def extract_value(ori_delta_t, target_delta_t, data):
    interval = round(target_delta_t / ori_delta_t)
    new_data = []
    for j, row in enumerate(data):
        new_row = []
        for i, X in enumerate(row):
            if i % interval == 0:
                new_row.append(X)
        new_data.append(new_row)
    data_final = np.array(new_data)
    time_point_num = data_final.shape[1]
    time = np.linspace(0.0, time_point_num * target_delta_t, time_point_num)
    return time, data_final


# %%
# def plot_output(simulation_data, subject_num, time_points_num, microgroup_num):
#     plt.figure()
#     plt.figure(figsize=(microgroup_num * 4, 2.5))

#     for z in range(microgroup_num):
#         plt.subplot(1, microgroup_num, z + 1)
#         for xx in range(subject_num):
#             plt.plot(range(time_points_num), simulation_data[xx, :time_points_num, z])
#         plt.text(0.55, 0.85, "Microbiome %d" % (z + 1), transform=plt.gca().transAxes)
#     plt.show()


# %%
def generate_time_series_new_method(mu_matrix, t, p, original_t, target_t, seed):
    time, mu_matrix_simplified = extract_value(original_t, target_t, mu_matrix)
    out = np.zeros_like(mu_matrix_simplified)
    out_without_zeros = np.zeros_like(mu_matrix_simplified)
    for i in range(mu_matrix_simplified.shape[0]):  # different time series
        for j in range(mu_matrix_simplified.shape[1]):  # different time points
            # for k in range(mu_matrix_simplified.shape[2]): #different features
            seed = int(seed + math.sqrt(abs(i - j) * seed)) + i + j
            temp_1, temp_2 = ZILN(
                mu_matrix_simplified.shape[2], mu_matrix_simplified[i, j, :], t, p, seed
            )
            out_without_zeros[i, j, :], out[i, j, :] = temp_1, temp_2
    return mu_matrix, time, mu_matrix_simplified, out_without_zeros, out


# %%
# Time series 1
sample_num = 200
time_point_num = 300
delta_t = 0.01
n_feature = 1
time_trend_class = ["BS"]

BS_k = [0.8]
BS_sigma = [0.2]

OU_k = []
OU_m = []
OU_sigma = []

start_value_min = [1]
start_value_max = [2]

t = 0.1
p = 0.15

original_t = 0.01
target_t = 0.1

time_sd = 0

cor_sd = 0
lambda_para_1 = 1
lambda_para_2 = 0

seed = 10
sig_mat = CorSig_gen(1, 0.7, 0.4, seed)

mu_matrix = multi_time_trend_generator_without_time_variation(
    sample_num,
    time_point_num,
    n_feature,
    time_trend_class,
    BS_k,
    BS_sigma,
    OU_k,
    OU_m,
    OU_sigma,
    delta_t,
    start_value_min,
    start_value_max,
    cor_sd,
    time_sd,
    lambda_para_1,
    lambda_para_2,
    sig_mat,
    seed,
)
# %%
# plot_output(mu_matrix, 10, 300, 1)

# %%
(
    mu_matrix_1,
    time_1,
    mu_matrix_simplified_1,
    time_series_without_zero_1,
    time_series_1,
) = generate_time_series_new_method(mu_matrix, t, p, original_t, target_t, seed)

# %%
# plot_output(mu_matrix_simplified_1, 10, 30, 1)

# %%
# plot_output(time_series_without_zero_1, 30, 30, 1)

# %%
# Time series 2
sample_num = 200
time_point_num = 300
delta_t = 0.01
n_feature = 1
time_trend_class = ["BS"]

BS_k = [-0.5]
BS_sigma = [0.2]

OU_k = []
OU_m = []
OU_sigma = []

start_value_min = [15]
start_value_max = [18]

t = 0.1
p = 0.15

original_t = 0.01
target_t = 0.1

time_sd = 0

cor_sd = 0
lambda_para_1 = 1
lambda_para_2 = 0

seed = 10
sig_mat = CorSig_gen(1, 0.7, 0.4, seed)


mu_matrix = multi_time_trend_generator_without_time_variation(
    sample_num,
    time_point_num,
    n_feature,
    time_trend_class,
    BS_k,
    BS_sigma,
    OU_k,
    OU_m,
    OU_sigma,
    delta_t,
    start_value_min,
    start_value_max,
    cor_sd,
    time_sd,
    lambda_para_1,
    lambda_para_2,
    sig_mat,
    seed,
)
# %%
# plot_output(mu_matrix, 10, 300, 1)

# %%
(
    mu_matrix_2,
    time_2,
    mu_matrix_simplified_2,
    time_series_without_zero_2,
    time_series_2,
) = generate_time_series_new_method(mu_matrix, t, p, original_t, target_t, seed)

# %%
# plot_output(mu_matrix_simplified_2, 10, 30, 1)

# %%
# plot_output(time_series_without_zero_2, 20, 30, 1)

# %%
# Time series 3
sample_num = 200
time_point_num = 300
delta_t = 0.01
n_feature = 1
time_trend_class = ["OU"]

BS_k = []
BS_sigma = []

OU_k = [5]
OU_m = [20]
OU_sigma = [4]

start_value_min = [2]
start_value_max = [3]
t = 0.1
p = 0.15

original_t = 0.01
target_t = 0.1

time_sd = 0

cor_sd = 0
lambda_para_1 = 1
lambda_para_2 = 0

seed = 10
sig_mat = CorSig_gen(1, 0.7, 0.4, seed)

mu_matrix = multi_time_trend_generator_without_time_variation(
    sample_num,
    time_point_num,
    n_feature,
    time_trend_class,
    BS_k,
    BS_sigma,
    OU_k,
    OU_m,
    OU_sigma,
    delta_t,
    start_value_min,
    start_value_max,
    cor_sd,
    time_sd,
    lambda_para_1,
    lambda_para_2,
    sig_mat,
    seed,
)
# %%
# plot_output(mu_matrix, 10, 300, 1)

# %%
(
    mu_matrix_3,
    time_3,
    mu_matrix_simplified_3,
    time_series_without_zero_3,
    time_series_3,
) = generate_time_series_new_method(mu_matrix, t, p, original_t, target_t, seed)

# %%
# plot_output(mu_matrix_simplified_3, 10, 30, 1)

# %%
# plot_output(time_series_without_zero_3, 20, 30, 1)

# %%
# Time series 4
sample_num = 200
time_point_num = 300
delta_t = 0.01
n_feature = 1
time_trend_class = ["OU"]

BS_k = []
BS_sigma = []

OU_k = [5]
OU_m = [4]
OU_sigma = [1.5]

start_value_min = [15]
start_value_max = [18]
t = 0.1
p = 0.15

original_t = 0.01
target_t = 0.1

time_sd = 0

cor_sd = 0
lambda_para_1 = 1
lambda_para_2 = 0

seed = 10
sig_mat = CorSig_gen(1, 0.7, 0.4, seed)

mu_matrix = multi_time_trend_generator_without_time_variation(
    sample_num,
    time_point_num,
    n_feature,
    time_trend_class,
    BS_k,
    BS_sigma,
    OU_k,
    OU_m,
    OU_sigma,
    delta_t,
    start_value_min,
    start_value_max,
    cor_sd,
    time_sd,
    lambda_para_1,
    lambda_para_2,
    sig_mat,
    seed,
)
# %%
# plot_output(mu_matrix, 10, 300, 1)

# %%
(
    mu_matrix_4,
    time_4,
    mu_matrix_simplified_4,
    time_series_without_zero_4,
    time_series_4,
) = generate_time_series_new_method(mu_matrix, t, p, original_t, target_t, seed)

# %%
# plot_output(mu_matrix_simplified_4, 10, 30, 1)

# %%
# plot_output(time_series_without_zero_4, 20, 30, 1)

# %%
# Time series 5
sample_num = 200
time_point_num = 300
delta_t = 0.01
n_feature = 1
time_trend_class = ["OU"]

BS_k = []
BS_sigma = []

OU_k = [5]
OU_m = [10]
OU_sigma = [1.5]

start_value_min = [5]
start_value_max = [18]
t = 0.1
p = 0.15

original_t = 0.01
target_t = 0.1

time_sd = 0

cor_sd = 0
lambda_para_1 = 1
lambda_para_2 = 0

seed = 10
sig_mat = CorSig_gen(1, 0.7, 0.4, seed)

mu_matrix = multi_time_trend_generator_without_time_variation(
    sample_num,
    time_point_num,
    n_feature,
    time_trend_class,
    BS_k,
    BS_sigma,
    OU_k,
    OU_m,
    OU_sigma,
    delta_t,
    start_value_min,
    start_value_max,
    cor_sd,
    time_sd,
    lambda_para_1,
    lambda_para_2,
    sig_mat,
    seed,
)
# %%
# plot_output(mu_matrix, 10, 300, 1)

# %%
(
    mu_matrix_5,
    time_5,
    mu_matrix_simplified_5,
    time_series_without_zero_5,
    time_series_5,
) = generate_time_series_new_method(mu_matrix, t, p, original_t, target_t, seed)

# %%
# plot_output(mu_matrix_simplified_5, 10, 30, 1)

# %%
# plot_output(time_series_without_zero_5, 30, 30, 1)

# %%
# def plot_output_all(simulation_data_list, subject_num, time_points_num, microgroup_num):
#     plt.figure()
#     plt.figure(figsize=(5, 12.5))

#     cluster_num = len(simulation_data_list)

#     i = 0
#     for time_series in simulation_data_list:
#         i += 1
#         for z in range(microgroup_num):
#             plt.subplot(cluster_num, microgroup_num, (i - 1) * microgroup_num + z + 1)
#             for xx in range(subject_num):
#                 plt.plot(range(time_points_num), time_series[xx, :time_points_num, z])
#             plt.text(
#                 0.55, 0.85, "Microbiome %d" % (z + 1), transform=plt.gca().transAxes
#             )
#     plt.show()


# %%
# plot_output_all(
#     [
#         time_series_without_zero_1,
#         time_series_without_zero_2,
#         time_series_without_zero_3,
#         time_series_without_zero_4,
#         time_series_without_zero_5,
#     ],
#     30,
#     30,
#     1,
# )

# %%
all_time_series = np.concatenate(
    (time_series_1, time_series_2, time_series_3, time_series_4, time_series_5), axis=0
)
# %%
column_name = (
    ["ID", "Time"]
    + ["Value" for i in range(all_time_series.shape[2])]
    + ["Mask_" + str(i + 1) for i in range(all_time_series.shape[2])]
)
final_df = pd.DataFrame(columns=column_name)
for i in range(all_time_series.shape[0]):
    ID = pd.Series([i for _ in range(all_time_series.shape[1])])
    Time = pd.Series([j for j in range(all_time_series.shape[1])])
    value_df = pd.DataFrame(all_time_series[i])
    mask_df = (value_df != 0) * 1

    temp_df = pd.concat([ID, Time, value_df, mask_df], axis=1)
    temp_df.columns = column_name
    final_df = pd.concat([final_df, temp_df], axis=0)
# %%
final_mask_df = final_df.loc[:, [a.startswith("Mask") for a in final_df.columns]]
final_df.index = pd.Series(range(final_df.shape[0]))
final_df_dropped = pd.DataFrame(columns=column_name)
for i, row in final_df.iterrows():
    # print(i)
    np.random.seed(i)
    if random.random() > 0.15:
        temp_row = row.to_frame().T
        final_df_dropped = pd.concat([final_df_dropped, temp_row], axis=0)
    else:
        final_mask_df.iloc[i, :] = -1

# %%
all_real_time_series = np.concatenate(
    (
        time_series_without_zero_1,
        time_series_without_zero_2,
        time_series_without_zero_3,
        time_series_without_zero_4,
        time_series_without_zero_5,
    ),
    axis=0,
)
column_name = (
    ["ID", "Time"]
    + ["Value" for i in range(all_real_time_series.shape[2])]
    + ["Mask_" + str(i + 1) for i in range(all_real_time_series.shape[2])]
)
real_df = pd.DataFrame(columns=column_name)
for i in range(all_real_time_series.shape[0]):
    ID = pd.Series([i for _ in range(all_real_time_series.shape[1])])
    Time = pd.Series([j for j in range(all_real_time_series.shape[1])])
    value_df = pd.DataFrame(all_real_time_series[i])
    mask_df = (value_df != 0) * 1
    temp_df = pd.concat([ID, Time, value_df, mask_df], axis=1)
    temp_df.columns = column_name
    real_df = pd.concat([real_df, temp_df], axis=0)

# %%
real_df.loc[:, [a.startswith("Mask") for a in real_df.columns]] = final_mask_df
# %%
final_df_dropped.iloc[:, :3].to_csv("simulated_data/simulation_random_dropped.csv", index=False)
real_df.iloc[:, :3].to_csv("simulated_data/simulation_all_data.csv", index=False)

