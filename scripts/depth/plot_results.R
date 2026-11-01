# Presentation figures from the completed CSV tables; no torch or training needed.
# Usage from repository root: Rscript scripts/depth/plot_results.R
args <- commandArgs(TRUE)
results <- if(length(args)>=1) args[1] else "results/depth"
figures <- if(length(args)>=2) args[2] else "figures/depth"
s <- read.csv(file.path(results,"summary.csv"))
h <- read.csv(file.path(results,"learning_curves.csv"))
stopifnot(nrow(s)>0,!anyDuplicated(paste(s$dataset,s$layers,s$seed)))
dir.create(figures,recursive=TRUE,showWarnings=FALSE)
datasets <- sort(unique(s$dataset)); depths <- sort(unique(s$layers))
colors <- c("#087E9D","#D65E32","#4E963E","#7F5BA7")
symbols <- c(16,17,18,15)

png(file.path(figures,"depth_accuracy_cost.png"),1800,1250,res=160)
par(oma=c(1.3,0,0,0),las=1)
layout(matrix(c(1,1,2,3,4,5),nrow=3,byrow=TRUE),heights=c(.16,1,1))
par(mar=rep(0,4)); plot.new()
legend("center",datasets,col=colors[seq_along(datasets)],pch=symbols[seq_along(datasets)],
  lty=1,lwd=2,horiz=TRUE,bty="n",cex=1)
par(mar=c(4.6,5.8,3.7,1))
for(metric in c("test_mse","test_mae","minutes","params")) {
  y <- range(s[[metric]])
  pad <- max(diff(y),max(abs(y))*.05)*.12
  ylabel <- switch(metric,
      test_mse="Standardized test MSE",test_mae="Standardized test MAE",
      minutes="Training + validation + test (min)",params="Trainable parameters")
  plot(c(1,length(depths)),c(y[1]-pad,y[2]+pad),type="n",xaxt="n",
    xlab="Layers L (tested configurations)",ylab="",
    main=switch(metric,test_mse="Forecast MSE",test_mae="Forecast MAE",
      minutes="Actual run time",params="Model size"),cex.lab=1.05)
  mtext(ylabel,side=2,line=4.6,las=0,cex=.95)
  axis(1,at=seq_along(depths),labels=depths)
  if(3 %in% depths) abline(v=match(3,depths),lty=3,col="grey65")
  for(j in seq_along(datasets)) {
    r <- s[s$dataset==datasets[j],]; r <- r[order(r$layers),]
    lines(match(r$layers,depths),r[[metric]],type="b",col=colors[j],pch=symbols[j],lwd=2)
  }
  if(metric=="params") mtext("Identical counts across datasets",side=3,line=.4,cex=.75)
}
mtext("Single seed 0; T=F=96; up to 10 epochs. L values are shown as equally spaced categories.",
  side=1,outer=TRUE,line=.25,cex=.85)
dev.off()

png(file.path(figures,"depth_validation_curves.png"),1800,470*length(datasets),res=160)
par(mfrow=c(length(datasets),1),mar=c(4,5.4,3,1),las=1)
palette <- hcl.colors(length(depths),"Dark 3")
for(ds in datasets) {
  r <- h[h$dataset==ds,]
  plot(range(r$epoch),range(r$val_mse),type="n",xlab="Epoch",ylab="Validation MSE",
    main=paste(ds,"- validation curves (seed 0)"))
  for(j in seq_along(depths)) {
    z <- r[r$layers==depths[j],]; z <- z[order(z$epoch),]
    lines(z$epoch,z$val_mse,type="b",col=palette[j],pch=j,lwd=1.5)
  }
  legend("topright",paste0("L=",depths),col=palette,lty=1,pch=seq_along(depths),
    bty="n",ncol=4,cex=.85)
}
dev.off()
cat("Saved two presentation figures from existing results; no models retrained.\n")
