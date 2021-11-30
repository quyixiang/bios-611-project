.PHONY: clean
SHELL: /bin/bash

clean:
	rm -f simulated_data/*
	rm -f figure/*
	rm -f aux_pdf_latex/*

figure/B_spline.pdf figure/DL_spline.pdf: plot.R real_data/all_interpolation.csv
	Rscript plot.R

real_data/all_interpolation.csv: split_data.py B_spline.R real_data/multivariate_simulation_dl_interpolation.csv simulated_data/simulation_all_data.csv simulated_data/simulation_random_dropped.csv
	python3 split_data.py
	Rscript B_spline.R

real_data/multivariate_simulation_dl_interpolation.csv: generate_interpolation_results.py real_data/multivariate_simulation.npy real_data/multivariate_simulation.npy
	python3 generate_interpolation_results.py

simulated_data/simulation_all_data.csv simulated_data/simulation_random_dropped.csv: simulator.py
	python3 simulator.py

shiny: app.R real_data/all_interpolation.csv
	Rscript -e 'library(methods); shiny::runApp("app.R", launch.browser = TRUE)'