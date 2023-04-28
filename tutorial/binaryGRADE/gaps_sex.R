library(Rmisc)
library(ggplot2)
library(stringr)
library(dplyr)

models = c("BTDownsampling", "BTResampling", "BTReweighting", "ADV", "vanilla", "DADV", "FCL")

protected_attribute = "sex"

criterion = "binaryGRADE"
file1 = paste("/home/simon/Apps/SysRevData/data/modelling/plots/evidencegrader/evidencegrader_perf_raw_", criterion, "_", protected_attribute, "_debias.csv", sep="")
f1 = read.table(file1, skip = 0, sep = ",", header=TRUE, na.strings = "NA", dec = ".", strip.white = TRUE)
criterion = "RoB"
file2 = paste("/home/simon/Apps/SysRevData/data/modelling/plots/robotreviewer/robotreviewer_perf_raw_", criterion, "_", protected_attribute, "_debias.csv", sep="")
f2 = read.table(file2, skip = 0, sep = ",", header=TRUE, na.strings = "NA", dec = ".", strip.white = TRUE)

rbind(f1, f2) %>% filter(grepl("vanilla", model)) %>% mutate(criterion=str_replace(criterion, "binaryGRADE", "EvidenceGRADEr")) %>%
mutate(criterion=str_replace(criterion, "RoB", "TrialstreamerRoB")) %>% ggplot(aes(x=topic, y=recall, fill=criterion)) +
geom_col(position="dodge2") + scale_fill_grey() + theme_bw() + scale_y_continuous("TPR", breaks = 0:10 / 10) +
xlab("Sex") + theme(plot.title = element_text(size = 14, hjust=0.5),
          #plot.margin = margin(0.1, 0.1, 0.1, 2.5, "cm"),
          axis.text.x = element_text(size = 13),#, angle=45, vjust=1, hjust=1),
          axis.text.y = element_text(size = 13),
          axis.title.x = element_text(size = 13),
          axis.title.y = element_text(size = 13),
          legend.title = element_blank(),
          legend.text = element_text(size = 13),
          #legend.position = "top",
          panel.border = element_blank())
f_out=paste("/home/simon/Apps/SysRevData/data/dataset/plots/risk_coverages_curves/evidencegrader/debias/recall_gaps", "_", protected_attribute, ".png", sep="")
ggsave(f_out, width = 8, height = 4)

