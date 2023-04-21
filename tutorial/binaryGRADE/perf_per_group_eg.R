library(Rmisc)
library(ggplot2)
library(stringr)
library(dplyr)

# EVIDENCEGRADER
#criteria = c("binaryGRADE", "imprecision", "risk of bias", "indirectness", "inconsistency", "publication bias")
criterion = "binaryGRADE"
models = c("BTDownsampling", "BTResampling", "BTReweighting", "ADV", "vanilla", "DADV", "FCL")

# EVIDENCEGRADER single criterion, all
criterion = "binaryGRADE"
protected_attribute = "area"

file1 = paste("/home/simon/Apps/SysRevData/data/modelling/plots/evidencegrader/evidencegrader_perf_raw_", criterion, "_", protected_attribute, "_debias.csv", sep="")
f1 = read.table(file1, skip = 0, sep = ",", header=TRUE, na.strings = "NA", dec = ".", strip.white = TRUE)
all = data.frame(f1)
all$model=factor(all$model)
all$topic=factor(all$topic)
all$criterion=factor(all$criterion)
#all$topic <- factor(all$topic, levels = all$topic[order(all$recall[all$model=="vanilla"])])
selected_models = c("vanilla", "ADV")
all_subset = subset(all, all$model %in% selected_models)

all_subset %>% group_by(topic) %>% mutate(recall_diff = recall - recall[model=="vanilla"]) %>% ungroup() %>% filter(model=="ADV") %>% arrange(recall_diff) %>% mutate(topic=factor(topic, levels=topic)) %>%
ggplot(aes(y=topic, x=recall_diff)) + geom_point(size = 1.5) + geom_vline(xintercept=0) + theme_bw() +
      #scale_y_continuous(breaks = 0:10 / 10, limits=c(0,0.45)) + xlab("protected attribute") +
    scale_y_discrete(label = function(x) stringr::str_trunc(x, 23)) +
    theme(plot.title = element_text(size = 14, hjust=0.5),
          #plot.margin = margin(0.1, 0.1, 0.1, 2.5, "cm"),
          axis.text.x = element_text(size = 13),#, angle=45, vjust=1, hjust=1),
          axis.text.y = element_text(size = 13),
          axis.title.x = element_text(size = 13),
          axis.title.y = element_text(size = 13),
          legend.title = element_blank(),
          legend.text = element_text(size = 10),
          #legend.position = "none",
          panel.border = element_blank()) + ylab("Medical area") + xlab("TPR difference")
f_out=paste("/home/simon/Apps/SysRevData/data/dataset/plots/risk_coverages_curves/evidencegrader/debias/recall_diff", gsub(" ", "_", criterion), "_", protected_attribute, ".png", sep="")
ggsave(f_out, width = 8, height = 7)


my = ggplot(data = all_subset, aes(x = topic, y = recall, group=model)) + geom_point(aes(colour = str_wrap(model, 15)), size = 0.5) +
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
f_out=paste("/home/simon/Apps/SysRevData/data/dataset/plots/risk_coverages_curves/evidencegrader/debias/recall_compare", gsub(" ", "_", criterion), "_", protected_attribute, ".png", sep="")
ggsave(f_out, width = 8, height = 7)

my = ggplot(data = all, aes(x = model, y = precision, color=model)) + geom_boxplot() + theme_bw() +
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
