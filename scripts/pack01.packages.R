################################################################################
#                                PACKAGES TO LOAD                              #
################################################################################

cat("\n* General packages...\n")
install.packages.auto("credentials")
library("credentials")
credentials::set_github_pat()

install.packages.auto("R.utils")

install.packages.auto("readr")
install.packages.auto("optparse")
install.packages.auto("tools")
install.packages.auto("dplyr")
install.packages.auto("tidyr")
install.packages.auto("naniar")

# To get 'data.table' with 'fwrite' to be able to directly write gzipped-files
# Ref: https://stackoverflow.com/questions/42788401/is-possible-to-use-fwrite-from-data-table-with-gzfile
# install.packages("data.table", repos = "https://Rdatatable.gitlab.io/data.table")
install.packages.auto("data.table")
# install.packages("data.table", dependencies = TRUE, force = TRUE)
library(data.table)

install.packages.auto("tidyverse")
install.packages.auto("knitr")
install.packages.auto("DT")
# needed for eeptools if it doesn't work straightaway
# install.packages("minqa", dependencies = TRUE, force = TRUE)
# install.packages("nloptr", dependencies = TRUE, force = TRUE)
# install.packages("lmtest", dependencies = TRUE, force = TRUE)
# install.packages("eeptools", dependencies = TRUE, force = TRUE)
install.packages.auto("eeptools")

install.packages.auto("haven")
install.packages.auto("tableone")

install.packages.auto("BlandAltmanLeh")

# Install the devtools package from Hadley Wickham
install.packages.auto('devtools')

# for plotting
install.packages.auto("pheatmap")
install.packages.auto("forestplot")
install.packages.auto("ggplot2")

install.packages.auto("ggpubr")

install.packages.auto("UpSetR")

devtools::install_github("thomasp85/patchwork")

# needed for sjPlot if it doesn't work straightaway
# install.packages("mvtnorm", dependencies = TRUE, force = TRUE)
# install.packages("sjPlot", dependencies = TRUE, force = TRUE)
install.packages.auto("sjPlot")
