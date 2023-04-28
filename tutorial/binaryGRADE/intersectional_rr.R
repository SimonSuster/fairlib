library(Rmisc)
library(ggplot2)
library(stringr)

file1 = "/home/simon/Apps/SysRevData/data/modelling/plots/robotreviewer/intersectional.csv"
f1 = read.table(file1, skip = 0, sep = ",", header=TRUE, na.strings = "NA", dec = ".", strip.white = TRUE)
all = data.frame(f1)
all$Area=factor(all$Area)
all$Sex=factor(all$Sex)

ggplot(data = all, aes(x=Area, y=Count, fill=Sex)) + geom_col() + theme_bw() +  scale_x_discrete(label = function(x) stringr::str_trunc(x, 16)) +
theme(plot.title = element_text(size = 14, hjust=0.5),
          #plot.margin = margin(0.1, 0.1, 0.1, 2.5, "cm"),
          axis.text.x = element_text(size = 13, angle=45, vjust=1, hjust=1),
          axis.text.y = element_text(size = 13),
          axis.title.x = element_text(size = 13),
          axis.title.y = element_text(size = 13),
          legend.title = element_blank(),
          legend.text = element_text(size = 13),
          legend.position = "top",
          panel.border = element_blank())

f_out="/home/simon/Apps/SysRevData/data/dataset/plots/intersectional.png"
ggsave(f_out, width = 8, height = 7)
