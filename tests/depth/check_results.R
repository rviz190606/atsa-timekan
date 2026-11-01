args <- commandArgs(TRUE)
if(length(args)!=1) stop("Provide the path to an extracted full-results archive")
root <- args[1]
files <- list.files(file.path(root,"results/depth/records"),"_complete.rds$",full.names=TRUE)
stopifnot(length(files)>0)
tab <- read.csv(file.path(root,"results/depth/summary.csv"))
stopifnot(length(files)==21,setequal(tab$dataset,c("ETTh1","ETTh2","ETTm1")),
  setequal(tab$layers,c(1,2,3,4,5,10,20)),all(table(tab$dataset)==7),nrow(tab)==length(files),!anyDuplicated(paste(tab$dataset,tab$layers)),all(tab$seed==0))
reference <- list()
for(f in files) {
  r <- readRDS(f); s <- r$summary; h <- r$history; c <- r$config
  preprocessing <- list(data_hash=r$hash[["data"]],scaling=r$scaling)
  if(is.null(reference[[s$dataset]])) reference[[s$dataset]] <- preprocessing else
    stopifnot(identical(reference[[s$dataset]],preprocessing))
  stopifnot(c$seed==0,c$seq_len==96,c$pred_len==96,c$epochs==10,c$patience==3,
    c$lr==.001,c$batch_size==128,c$d_model==16,c$levels==3,c$window==2,
    c$base_order==1,c$fixed_order==0,c$order_mode=="paper",c$basis=="chebyshev",c$cfd_mode=="paper")
  stopifnot(s$best_epoch==which.min(h$val_mse),abs(s$val_mse-min(h$val_mse))<1e-12,
    nrow(h)==s$epochs_run,all(is.finite(h$train_mse)),all(is.finite(h$val_mse)),
    all(h$seconds>0),s$hit_epoch_limit==(s$epochs_run==10),
    s$best_in_last_two==(s$best_epoch>=s$epochs_run-1),
    s$test_mse>=0,s$test_mae>=0,s$test_mae<=sqrt(s$test_mse)+1e-6,
    s$params==9425+2496*s$layers)
  if(!s$hit_epoch_limit) stopifnot(s$epochs_run-s$best_epoch==3)
  mult <- if(grepl("^ETTm",s$dataset)) 4 else 1
  stopifnot(identical(as.numeric(r$windows),c(8640*mult-191,2880*mult-95,2880*mult-95)))
  stopifnot(r$hash[["model"]]==unname(tools::md5sum(file.path(root,"timekan.R"))),
    r$hash[["runner"]]==unname(tools::md5sum(file.path(root,"run_depth_experiment.R"))))
  row <- tab[tab$dataset==s$dataset & tab$layers==s$layers,]
  stopifnot(abs(row$test_mse-s$test_mse)<1e-12,abs(row$test_mae-s$test_mae)<1e-12)
}
cat("VERIFIED",length(files),"actual records: settings, ETT splits, checkpoint selection, stopping, metrics, parameter counts and source hashes.\n")
