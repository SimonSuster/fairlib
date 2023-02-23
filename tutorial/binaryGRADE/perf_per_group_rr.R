library(Rmisc)
library(ggplot2)
library(stringr)

criterion = "RoB"
protected_group = "Sex"
file1 = paste("/home/simon/Apps/SysRevData/data/modelling/plots/robotreviewer/robotreviewer_perf_raw_", criterion, "_", tolower(protected_group), "_debias.csv", sep="")
f1 = read.table(file1, skip = 0, sep = "\t", header=TRUE, na.strings = "NA", dec = ".", strip.white = TRUE)
all = data.frame(f1)
all$model=factor(all$model)
all$topic=factor(all$topic)
all$criterion=factor(all$criterion)
protected_group_substr = substr(protected_group, nchar(protected_group)-2, nchar(protected_group))
all$topic <- factor(all$topic, levels = all$topic[order(all$recall[all$model==paste(protected_group_substr, "vanilla", sep="")])])
#all$topic <- factor(all$topic, levels = all$topic[order(all$precision[all$model==paste(protected_group_substr, "vanilla", sep="")])])

#selected_models = c(paste(protected_group_substr, "vanilla", sep=""), paste(protected_group_substr, "ADV", sep=""))
#all_subset = subset(all, all$model %in% selected_models)
all_subset = all

my = ggplot(data = all_subset, aes(x = topic, y = recall, group=model)) + geom_point(aes(colour = str_wrap(model, 15)), size = 0.5) +
#my = ggplot(data = all_subset, aes(x = topic, y = precision, group=model)) + geom_point(aes(colour = str_wrap(model, 15)), size = 0.5) +
geom_line(aes(colour = str_wrap(model, 15)), size = 0.5) + theme_bw() +
      #scale_y_continuous(breaks = 0:10 / 10, limits=c(0,0.45)) + xlab("protected attribute") +
    theme(plot.title = element_text(size = 14, hjust=0.5),
          axis.text.x = element_text(size = 8, angle=45, vjust=1, hjust=1),
          axis.text.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.title.y = element_text(size = 12),
          legend.title = element_blank(),
          legend.text = element_text(size = 10),
          #legend.position = "none",
          panel.border = element_blank())
f_out=paste("/home/simon/Apps/SysRevData/data/dataset/plots/risk_coverages_curves/evidencegrader/debias/recall_compare", gsub(" ", "_", criterion), tolower(protected_group), ".png", sep="")
#f_out=paste("/home/simon/Apps/SysRevData/data/dataset/plots/risk_coverages_curves/evidencegrader/debias/precision_compare", gsub(" ", "_", criterion), tolower(protected_group), ".png", sep="")
ggsave(f_out, width = 8, height = 7)

my = ggplot(data = all, aes(x = model, y = recall, color=model)) + geom_boxplot() + theme_bw() +
      #scale_y_continuous(breaks = 0:10 / 10, limits=c(0,0.45)) + xlab("protected attribute") +
    theme(plot.title = element_text(size = 14, hjust=0.5),
          axis.text.x = element_text(size = 8, angle=45, vjust=1, hjust=1),
          axis.text.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.title.y = element_text(size = 12),
          legend.title = element_blank(),
          legend.text = element_text(size = 10),
          #legend.position = "none",
          panel.border = element_blank())
