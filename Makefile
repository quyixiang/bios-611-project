.PHONY: clean
SHELL: /bin/bash

clean:
	rm -f simulated_data/*
	rm -f figure/*
	rm -f aux_pdf_latex/*

report.pdf: report.tex figure/sim_hf_age_sex_new.png figure/sim_hf_age_sex.png
	pdflatex -output-directory="aux_pdf_latex" report.tex

figure/sim_hf_age_sex_new.png figure/sim_hf_age_sex.png: spline.R simulated_data/simulation_all_data.csv simulated_data/simulation_random_dropped.csv
	Rscript spline.R

simulated_data/simulation_all_data.csv simulated_data/simulation_random_dropped.csv: simulator.py
	python simulator.py
