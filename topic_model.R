rm(list = ls()) #clear the variables (just in case)

this_dir <- dirname(parent.frame(2)$ofile) # find the directory from which the R Script is being sourced
setwd(this_dir)

#load text mining library
library(tm)
library(slam)
#load topic models library
library(topicmodels)
library(rjson)
library(snow)
library(parallel)
library(stringr)
library(stringi)
#FOr topic visualization
library(LDAvis)
library(dplyr)

harmonicMean <- function(logLikelihoods, precision = 2000L) {
  library("Rmpfr")
  llMed <- median(logLikelihoods)
  as.double(llMed - log(mean(exp(
    -mpfr(logLikelihoods,
          prec = precision) + llMed
  ))))
}

#define all stopwords
genericStopwords <- c(
  stopwords("english"),
  stopwords("SMART")
)
genericStopwords <- gsub("'", "", genericStopwords)  #remove apostrophes
genericStopwords <- unique(genericStopwords)
#genericStopwords <- stemDocument(genericStopwords, language = "porter")

#Set parameters for Gibbs sampling for LDA
nstart <- 5
seed <-
  list(5,
       46225,
       500,
       6300,
       190000)
best <- TRUE
burnin <- 5000
iter <- 10000
keep <- 100

#Range of topic numbers to search for optimum number
sequ <-
  seq(2, 25, 1)

#load files into corpus
#get listing of .txt files in directory
filenames <- list.files(paste(getwd(),"textmining",sep = "/"),pattern="*.txt",full.names = TRUE)

#read files into a character vector
data_orig <- lapply(filenames,readLines)

#pre-processing:
data <- tolower(data_orig)  #force to lowercase
data[stri_count(data, regex="\\S+") < 8] = ""
data <- gsub("'", "", data)  #remove apostrophes
data <-
  gsub("[[:punct:]]", " ", data)  #replace punctuation with space
data <-
  gsub("[[:cntrl:]]", " ", data)  #replace control characters with space
data <-
  gsub("[[:digit:]]", "", data)  #remove digits
data <-
  gsub("^[[:space:]]+", "", data) #remove whitespace at beginning of documents
data <-
  gsub("[[:space:]]+$", "", data) #remove whitespace at end of documents
data <- stripWhitespace(data)

#load files into corpus
#create corpus from vector
data_docs <- Corpus(VectorSource(data))

#inspect a particular document in corpus
writeLines(as.character(data_docs[[2]]))

#Removal of stopwords
data_docs <- tm_map(data_docs, removeWords, genericStopwords)

#Good practice to check every now and then
writeLines(as.character(data_docs[[2]]))

#Create document-term matrix
dtm <- DocumentTermMatrix(data_docs)
#remove terms that occur in less than 1% of the documents
ind <- col_sums(dtm) < length(data) * 0.01
dtm <- dtm[,!ind]
#remove documents with no terms
ind <- row_sums(dtm) == 0
dtm <- dtm[!ind,]
data_docs <- data_docs[!ind]
#collapse matrix by summing over columns
freq <- col_sums(dtm)
#create sort order (descending)
freq <- freq[order(freq, decreasing = TRUE)]
#List all terms in decreasing order of freq
term_count_table <-
  data.frame(
    Term = names(freq),
    Count = unname(freq)
  )

#Run LDA using Gibbs sampling
##Calculate the number of cores
no_cores <- detectCores()
cl <- makeCluster(no_cores)
clusterExport(
  cl,
  list(
    "dtm",
    "sequ",
    "nstart",
    "seed",
    "best",
    "burnin",
    "iter",
    "keep"
  )
)

#Run LDA through all the numbers in the range and store all the models found in fitted_many
fitted_many <- clusterApplyLB(
  cl,
  sequ,
  fun = function(i) {
    library(topicmodels)
    LDA(
      dtm,
      k = i,
      method = "Gibbs",
      control = list(
        nstart = nstart,
        seed = seed,
        best = best,
        burnin = burnin,
        iter = iter,
        keep = keep
      )
    )
  }
)
stopCluster(cl)

#extract logliks from each topic
#where keep indicates that every keep iteration the log-likelihood is evaluated and stored. This returns all log-likelihood values including burnin, i.e., these need to be omitted before calculating the harmonic mean:
logLiks_many <-
  lapply(fitted_many, function(L)
    L@logLiks[-c(1:(burnin / keep))])

#compute harmonic means
hm_many <- sapply(logLiks_many, function(h)
  harmonicMean(h))

#inspect
plot(sequ, hm_many, type = "l")

#compute optimum number of topics & extract relevant topic models
ind <- which.max(hm_many)
print(paste("The optimum number of topics for the data set is ",sequ[ind]))
ldaOut <- fitted_many[ind]
ldaOut <- ldaOut[[1]]

##Prepare data for Visualization
#Calculate the number of cores
no_cores <- detectCores()
cl <- makeCluster(no_cores)

#Find required quantities
phi <- posterior(ldaOut)$terms %>% as.matrix
theta <- posterior(ldaOut)$topics %>% as.matrix
vocab <- colnames(phi)
doc_length <- vector()
for (i in 1:length(data_docs)) {
  temp <- paste(data_docs[[i]]$content, collapse = ' ')
  doc_length <- c(doc_length, stri_count(temp, regex = '\\S+'))
}

json_lda <- LDAvis::createJSON(
  phi = phi,
  theta = theta,
  vocab = vocab,
  doc.length = doc_length,
  term.frequency = col_sums(dtm),
  R = 10,
  cluster = cl,
  plot.opts = list(xlab = "Dimension 1", ylab = "Dimension 2")
)
stopCluster(cl)

#Topics visualization
serVis(
  json_lda,
  out.dir = "Vis"
)
