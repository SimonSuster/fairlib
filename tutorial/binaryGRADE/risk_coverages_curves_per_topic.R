library(Rmisc)
library(ggplot2)
library(stringr)

# EVIDENCEGRADER
#criteria = c("binaryGRADE", "imprecision", "risk of bias", "indirectness", "inconsistency", "publication bias")
criterion = "binaryGRADE"
models = c("BTDownsampling", "BTResampling", "BTReweighting", "ADV", "vanilla", "DADV", "FCL")
#models = c("vanilla")
plot_list = vector("list", length(models))
file1 = paste("/home/simon/Apps/SysRevData/data/modelling/plots/evidencegrader/evidencegrader_aurc_raw_", criterion, "_debias.csv", sep="")
f1 = read.table(file1, skip = 0, sep = "\t", header=TRUE, na.strings = "NA", dec = ".", strip.white = TRUE)
all = data.frame(f1)
all$model=factor(all$model)
all$topic=factor(all$topic)
all$criterion=factor(all$criterion)

for (i in seq_along(models))
{
  all_subset = subset(all, all$model==models[i])
  worst_topics = unique(all_subset[all_subset$coverage==1,]$topic[order(all_subset[all_subset$coverage==1,]$risk, decreasing=TRUE)])[1:3]
  best_topics = unique(all_subset[all_subset$coverage==1,]$topic[order(all_subset[all_subset$coverage==1,]$risk, decreasing=FALSE)])[1:3]
  #best_topic = all[all$coverage==1,]$topic[which.min(all[all$coverage==1,]$risk)]
  freq_topics = c("Mental health", "Cancer", "Rheumatology")
  generally_bad_topics = c("Gynaecology", "Skin disorders", "Wounds", "Kidney disease", "Neurology")
  #all_subset = subset(all_subset, all_subset$topic %in% worst_topics | all_subset$topic=="all" | all_subset$topic %in% best_topics)
  #all_subset = subset(all_subset, all_subset$topic %in% best_topics)
  #all_subset = subset(all_subset, all_subset$topic %in% worst_topics)
  all_subset = subset(all_subset, all_subset$topic %in% generally_bad_topics)
  #all_subset = subset(all_subset, all_subset$topic %in% freq_topics | all_subset$topic=="all")
  #all_subset = subset(all_subset, all_subset$topic %in% freq_topics)
  background_color = if (models[i]=="vanilla") "darkgoldenrod1" else "white"
  my = ggplot(data = all_subset, aes(x = coverage, y = risk)) + geom_line(aes(colour = str_wrap(topic, 15)), size = 1) + geom_point(aes(colour = str_wrap(topic, 15)), size = 1) + theme_bw() + ggtitle(models[i]) +
    scale_y_continuous(breaks = 0:10 / 10, limits=c(0,1)) +
    theme(plot.title = element_text(size = 16, hjust=0.5),
          axis.text.x = element_text(size = 12),
          axis.text.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.title.y = element_text(size = 12),
          legend.title = element_blank(),
          legend.text = element_text(size = 10),
          legend.position="bottom",
          panel.border = element_blank(),
          panel.background = element_rect(fill = background_color, colour = background_color))

  plot_list[[i]] = my
}
png("/home/simon/Apps/SysRevData/data/dataset/plots/risk_coverages_curves/evidencegrader/debias/aurc_raw.png", width = 2000, height=1000, res=100, units="px")
multiplot(plotlist=plot_list, cols=3)
dev.off()

# EVIDENCEGRADER single criterion, all
criterion = "binaryGRADE"

file1 = paste("/home/simon/Apps/SysRevData/data/modelling/plots/evidencegrader/evidencegrader_aurc_raw_", criterion, "_debias.csv", sep="")
f1 = read.table(file1, skip = 0, sep = "\t", header=TRUE, na.strings = "NA", dec = ".", strip.white = TRUE)
all = data.frame(f1)
all$model=factor(all$model)
all$topic=factor(all$topic)
all$criterion=factor(all$criterion)
#topic_name = "all"
#topic_name = "Neurology"
topic_name = "Cancer"
#generally_bad_topics = c("Gynaecology", "Skin disorders", "Wounds", "Kidney disease", "Neurology")

all_subset = subset(all, all$topic==topic_name)

my = ggplot(data = all_subset, aes(x = coverage, y = risk)) + geom_line(aes(colour = str_wrap(model, 15)), size = 0.5) + theme_bw() + ggtitle(paste(criterion, topic_name, sep=": ")) +
      scale_y_continuous(breaks = 0:10 / 10, limits=c(0,0.5)) +
    theme(plot.title = element_text(size = 16, hjust=0.5),
          axis.text.x = element_text(size = 12),
          axis.text.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.title.y = element_text(size = 12),
          legend.title = element_blank(),
          legend.text = element_text(size = 10),
          #legend.position = "none",
          panel.border = element_blank())
f_out=paste("/home/simon/Apps/SysRevData/data/dataset/plots/risk_coverages_curves/evidencegrader/debias/aurc_raw_compare", gsub(" ", "_", criterion), ".png", sep="")
ggsave(f_out, width = 8, height = 5)

# EVIDENCEGRADER single criterion
criteria = c("binary GRADE", "imprecision", "risk of bias", "indirectness", "inconsistency", "publication bias")
for (criterion in criteria){
#criterion = "risk of bias"
file1 = paste("/home/simon/Apps/SysRev/data/modelling/plots/evidencegrader/evidencegrader_aurc_raw", criterion, ".csv", sep="")
f1 = read.table(file1, skip = 0, sep = "\t", header=TRUE, na.strings = "NA", dec = ".", strip.white = TRUE)
all = data.frame(f1)
all$topic=factor(all$topic)
all$criterion=factor(all$criterion)
worst_topics = unique(all[all$coverage==1,]$topic[order(all[all$coverage==1,]$risk, decreasing=TRUE)])[1:3]
#best_topic = all[all$coverage==1,]$topic[which.min(all[all$coverage==1,]$risk)]
all_subset = subset(all, all$topic %in% worst_topics | all$topic=="all")
my = ggplot(data = all_subset, aes(x = coverage, y = risk)) + geom_line(aes(colour = str_wrap(topic, 15)), size = 2) + theme_bw() + ggtitle(criterion) +
      scale_y_continuous(breaks = 0:10 / 10, limits=c(0,1)) +
    theme(plot.title = element_text(size = 20, hjust=0.5),
          axis.text.x = element_text(size = 16),
          axis.text.y = element_text(size = 16),
          axis.title.x = element_text(size = 18),
          axis.title.y = element_text(size = 18),
          legend.title = element_blank(),
          legend.text = element_text(size = 16),
          legend.position = "bottom",
          panel.border = element_blank())
f_out=paste("/home/simon/Apps/SysRev/data/dataset/plots/risk_coverages_curves/evidencegrader/aurc_raw_with_worst_", gsub(" ", "_", criterion), ".png", sep="")
ggsave(f_out)#, width = width, height = height)
}


# EVIDENCEGRADER with TS
criteria = c("binary GRADE", "imprecision", "risk of bias", "indirectness", "inconsistency", "publication bias")
plot_list = vector("list", length(criteria))
for (i in seq_along(criteria))
{
  file1 = paste("/home/simon/Apps/SysRev/data/modelling/plots/evidencegrader/evidencegrader_aurc_raw_ts", criteria[i], ".csv", sep="")
  f1 = read.table(file1, skip = 0, sep = "\t", header=TRUE, na.strings = "NA", dec = ".", strip.white = TRUE)
  all = data.frame(f1)
  all$topic=factor(all$topic)
  all$criterion=factor(all$criterion)

  worst_topics = unique(all[all$coverage==1,]$topic[order(all[all$coverage==1,]$risk, decreasing=TRUE)])[1:3]
  #best_topic = all[all$coverage==1,]$topic[which.min(all[all$coverage==1,]$risk)]
  all_subset = subset(all, all$topic %in% worst_topics | all$topic=="all")

  my = ggplot(data = all_subset, aes(x = coverage, y = risk)) + geom_line(aes(colour = str_wrap(topic, 15)), size = 1) + theme_bw() + ggtitle(criteria[i]) +
    theme(plot.title = element_text(size = 16, hjust=0.5),
          axis.text.x = element_text(size = 12),
          axis.text.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.title.y = element_text(size = 12),
          legend.title = element_blank(),
          legend.text = element_text(size = 10),
          legend.position="bottom",
          panel.border = element_blank())

  plot_list[[i]] = my
}
png("/home/simon/Apps/SysRev/data/dataset/plots/risk_coverages_curves/evidencegrader/aurc_raw_ts.png", width = 1100, units="px")
multiplot(plotlist=plot_list, cols=3)
dev.off()


# ROBOTREVIEWER
criteria = c("Allocation concealment", "Blinding of outcome assessment", "Blinding of participants and personnel", "Random sequence generation")
plot_list = vector("list", length(criteria))
for (i in seq_along(criteria))
{
  file1 = paste("/home/simon/Apps/SysRev/data/modelling/plots/robotreviewer/robotreviewer_aurc_raw", criteria[i], ".csv", sep="")
  f1 = read.table(file1, skip = 0, sep = "\t", header=TRUE, na.strings = "NA", dec = ".", strip.white = TRUE)
  all = data.frame(f1)
  all$topic=factor(all$topic)
  all$criterion=factor(all$criterion)

  worst_topics = unique(all[all$coverage==1,]$topic[order(all[all$coverage==1,]$risk, decreasing=TRUE)])[1:3]
  #best_topic = all[all$coverage==1,]$topic[which.min(all[all$coverage==1,]$risk)]
  all_subset = subset(all, all$topic %in% worst_topics | all$topic=="all")

  my = ggplot(data = all_subset, aes(x = coverage, y = risk)) + geom_line(aes(colour = str_wrap(topic, 15)), size = 2) + theme_bw() + ggtitle(criteria[i]) +
       theme(plot.title = element_text(size = 20, hjust=0.5),
          axis.text.x = element_text(size = 16),
          axis.text.y = element_text(size = 16),
          axis.title.x = element_text(size = 18),
          axis.title.y = element_text(size = 18),
          legend.title = element_blank(),
          legend.text = element_text(size = 16),
          legend.position="bottom",
          panel.border = element_blank())

  plot_list[[i]] = my
  f_out=paste("/home/simon/Apps/SysRev/data/dataset/plots/risk_coverages_curves/robotreviewer/aurc_raw_with_worst_", gsub(" ", "_", criteria[i]), ".png", sep="")
  ggsave(f_out)#, width = width, height = height)

}
png("/home/simon/Apps/SysRev/data/dataset/plots/risk_coverages_curves/robotreviewer/aurc_raw.png", width = 800, units="px")
multiplot(plotlist=plot_list, cols=2)
dev.off()

# ROBOTREVIEWER with TS
criteria = c("Allocation concealment", "Blinding of outcome assessment", "Blinding of participants and personnel", "Random sequence generation")
plot_list = vector("list", length(criteria))
for (i in seq_along(criteria))
{
  file1 = paste("/home/simon/Apps/SysRev/data/modelling/plots/robotreviewer/robotreviewer_aurc_raw", criteria[i], ".csv", sep="")
  f1 = read.table(file1, skip = 0, sep = "\t", header=TRUE, na.strings = "NA", dec = ".", strip.white = TRUE)
  all = data.frame(f1)
  all$topic=factor(all$topic)
  all$criterion=factor(all$criterion)

  worst_topics = unique(all[all$coverage==1,]$topic[order(all[all$coverage==1,]$risk, decreasing=TRUE)])[1:3]
  #best_topic = all[all$coverage==1,]$topic[which.min(all[all$coverage==1,]$risk)]
  all_subset = subset(all, all$topic %in% worst_topics | all$topic=="all")

  my = ggplot(data = all_subset, aes(x = coverage, y = risk)) + geom_line(aes(colour = str_wrap(topic, 15)), size = 1) + theme_bw() + ggtitle(criteria[i]) +
        scale_y_continuous(breaks = 0:10 / 10, limits=c(0,1)) +
    theme(plot.title = element_text(size = 16, hjust=0.5),
          axis.text.x = element_text(size = 12),
          axis.text.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.title.y = element_text(size = 12),
          legend.title = element_blank(),
          legend.text = element_text(size = 10),
          legend.position="bottom",
          panel.border = element_blank())

  plot_list[[i]] = my
}
png("/home/simon/Apps/SysRev/data/dataset/plots/risk_coverages_curves/robotreviewer/aurc_raw.png", width = 800, units="px")
multiplot(plotlist=plot_list, cols=2)
dev.off()
