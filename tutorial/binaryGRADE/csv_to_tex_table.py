from fairlib.src.utils import csv_results_to_tex_table
# RR Sex
#f = "/home/simon/Apps/fairlib/tutorial/RoB/dev/RRSex_results.csv"
#f_calib = "/home/simon/Apps/fairlib/tutorial/RoB/dev/RRSex_calib_results.csv"

# RR Area
f = "/home/simon/Apps/fairlib/tutorial/RoB/dev/RRArea_results.csv"
f_calib = "/home/simon/Apps/fairlib/tutorial/RoB/dev/RRArea_calib_results.csv"

# EG Area
#f = "/home/simon/Apps/fairlib/tutorial/binaryGRADE/dev/EGArea_folds_results.csv"
#f_calib = "/home/simon/Apps/fairlib/tutorial/binaryGRADE/dev/EGArea_folds_calib_results.csv"

# EG Sex
#f = "/home/simon/Apps/fairlib/tutorial/binaryGRADE/dev/EGSex_folds_results.csv"
#f_calib = "/home/simon/Apps/fairlib/tutorial/binaryGRADE/dev/EGSex_folds_calib_results.csv"

csv_results_to_tex_table(f, f_calib)
