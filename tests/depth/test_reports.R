source("run_depth_experiment.R")
stopifnot(identical(depth_args(character())$depths, "1,2,3,4,5,10,20"))
stopifnot(inherits(try(depth_args(c("--depths", "0,2")), silent=TRUE), "try-error"))
stopifnot(inherits(try(depth_args(c("--datasets", "ETTh1,ETTh1")), silent=TRUE), "try-error"))
root <- tempfile("depth_report_check_")
dir.create(file.path(root, "records"), recursive=TRUE, showWarnings=FALSE)
opt <- depth_defaults; opt$results_dir <- root; opt$figures_dir <- file.path(root, "figures")
for (ds in c("ETTh1", "ETTh2", "ETTm1")) for (L in c(1,3,20)) {
  row <- data.frame(dataset=ds, layers=L, test_mse=.3+L/100, test_mae=.2+L/200,
    minutes=L/2, params=1000*L)
  curves <- data.frame(dataset=ds, layers=L, epoch=1:3, val_mse=c(.5,.4,.45)+L/100,
    train_mse=c(.7,.5,.4), seconds=10*L)
  saveRDS(list(summary=row, history=curves), file.path(root,"records",paste0(ds,"_L",L,"_complete.rds")))
}
depth_tables(opt)
stopifnot(nrow(read.csv(file.path(root,"summary.csv")))==9,
  nrow(read.csv(file.path(root,"learning_curves.csv")))==27,
  all(file.info(list.files(opt$figures_dir,full.names=TRUE))$size>10000))
csv_only <- tempfile("depth_csv_only_")
dir.create(csv_only)
stopifnot(all(file.copy(file.path(root,c("summary.csv","learning_curves.csv")),csv_only)))
opt$results_dir <- csv_only
depth_tables(opt)
stopifnot(nrow(read.csv(file.path(csv_only,"summary.csv")))==9)
cat("DEPTH_ARGUMENTS_AGGREGATION_AND_PLOTTING_OK (synthetic fixtures; not experiment results)\n")

