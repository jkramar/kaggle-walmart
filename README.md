# kaggle-walmart

######Introduction

This repo contains hacky code I cobbled together for my first [Kaggle contest](https://www.kaggle.com/c/walmart-recruiting-trip-type-classification). I'm putting this documentation together after the fact to show how I thought about the problem and what I learned. My process during the contest wasn't optimized for sharing, so I won't be able to accurately tell the story behind every tweak and result; but the rough outlines are correct.

[//]: # (Again: All code here is hacky contest code, not cleaned up for maintenance or teamwork. It's not intended to be representative of non-throwaway code that I write. :)

######At first glance ...

At first glance this contest problem seems relatively easy - there's no image or text processing, and not that many features to deal with. The main thing about it that isn't bog-standard is that the training data contains multiple entries for each unit, because each unit is actually a shopping trip, and the data is about every item that was purchased in each trip. One naïve way to handle this would be to do the classification for each item, and then ensemble these classifications to make a prediction for each shopping trip. This is easy to deal with, but also seems unsatisfying - maybe this would lose some essential features of the data. So instead, I observed that these bags-of-items are somewhat analogous to the bags-of-words that are ubiquitous in text analysis. What if we used [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) to represent the interestingness of each purchased item?

But this is getting ahead of ourselves - in fact I started out generating features by just counting how many item-types were purchased from each department.

######Notes on preliminaries

Regarding cross-validation: once upon a time I did a project with a small dataset, very noisy data with a temporal component, and a fairly large class of possible predictors. That project taught me that nested cross-validation is annoying to interpret. It's nice to be able to work with such a large dataset that I can just do holdout validation.

Regarding data cleaning: one issue that comes up early on is that some of the output labels are quite rare, especially TripType 14. (Arguably the fact that the output labels don't have any explained meanings makes this a less interesting problem, though it makes little difference when solving it.) It seems to me that the conceptually cleanest way to deal with these rare labels would be to ensure that each classifier is able to operate with missing labels, by making use of some prior probabilities on the labels; however, in practice it was clearly easiest to exclude TripType 14 from the training set, and when generating outputs, always assign probability 0.00004 to TripType 14 (which is roughly its frequency in the data) while scaling all the other probabilities by 0.99996.)

######Logistic regression

My first classification attempts used logistic regression via the [glmnet](https://cran.r-project.org/web/packages/glmnet/index.html) library in CRAN, which fits various generalized linear models with mixtures of L1 and L2 regularization. However, it seemed to be too slow to cope with this dataset. (Maybe this is unfair - I'm not sure whether glmnet was impractically slow immediately or only after adding some of the sparse features.) So I moved on to [liblinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/), which is also the default logistic regression implementation in [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). After switching to liblinear, I could empirically confirm the folk wisdom that with such a large amount of data, adding large numbers of noisy features wouldn't harm the algorithm very much, and would sometimes help substantially; this was heartening.

######Features

Using this logistic regression, I tested out various features, mostly based on the 3 categorical features each item-type had: the department, the "fineline number", and the UPC code. For each shopping trip, and for each value of these categorical variables, I tried coding a feature that looked at:
* whether the category was present in the trip,
* the number of item-types of that category,
* the total number of items of that category,
* the number of returned items of that category, or
* TF-IDF transformations of the above 3.

The logistic regression allowed me to discard most of those features as being irrelevant, but this still left me with a fairly large number of features. (There are 69 departments, 5196 fineline numbers, and 97715 UPC codes in the dataset.)

Additional features that seemed intuitive to try included the day of the week (which seemed to not help, to my surprise) and polynomials in the number of item-types, the number of items, and the number of items returned (which did).

After implementing this, I tried looking diagnostically at how the validation loss was distributed over true labels and over predicted labels, and found that 36 and 39 seemed harder to distinguish than most other pairs - eyeballing some examples to see if the existing features were missing something, I had a conjecture that the first half of the 4-digit "fineline numbers" were meaningful. So I tried adding features for every subset of the digits, and indeed the leading 2 digits improved validation error and none of the other features did.

I also considered doing something with the UPC codes, but my first idea for this involved googling some of the codes (which generally yielded no matches, to my surprise), and when I followed this up by finding Walmart's API, I learned that the contest didn't permit external datasets to be used. Perhaps some data exploration would have been within the rules, but I stopped here. After the contest I learned that indeed a manufacturer code could have been extracted with [some more background research](https://www.kaggle.com/c/walmart-recruiting-trip-type-classification/forums/t/18158/decoding-upc/103032). Oh, well.

######Non-neural algorithms

With the above feature selection and some logistic regression tuning, I could get validation loss ~0.75, which seemed respectable. (Right now that's ~70-th percentile on the leaderboard.) I also tried linear and Gaussian-kernel SVMs; the former brought loss down to ~0.70, the latter did similarly well but with much worse runtime. The comparison between logistic regression and linear SVMs is interesting; obviously the model is similar. Perhaps the fact that SVMs don't natively generate predictive probabilities means that a more sophisticated process is used for deriving the probabilities, which may be better-calibrated than the simple logistic link function. I wonder whether some recalibration of the logistic regression outputs would bring the performance to par with the SVM. Another possibility is that since libsvm's multiclass classification works by fitting a number of one-vs-one SVMs, it's benefiting from using more parameters (and hence greater modeling capacity).

I then tried [XGBoost](https://github.com/dmlc/xgboost), which did even better with the same feature matrix, after some tuning. (Roughly ~0.65 loss.) The XGBoost package also permits regularization via early stopping, which is a nice feature to have - I wonder whether logistic regression / SVM could have benefitted from it.

######Neural nets

I only implemented neural nets during the last week or so, partly because of software configuration issues. (Hello, CUDA.) Because I'm pretty familiar with the Python toolset, I decided to use one of the Theano-based libraries; [Lasagne](https://github.com/Lasagne/Lasagne) was the one whose philosophy seemed most appealing. (It's apparently a much thinner wrapper around Theano than things like [Keras](http://keras.io/) or [Blocks](https://github.com/mila-udem/blocks).)

I quickly ran into issues with my high-dimensional, sparse feature matrix. Firstly, Lasagne and the other libraries don't seem to have much support for sparse inputs; consequently I ended up writing sparse version of Lasagne's `DenseLayer` and `DropoutLayer`, which are in `sparse_layers.py`. Secondly, as soon as I started trying to run a simple network based on these on my laptop's GPU (a GT750M), I got out-of-memory errors. It seems that Theano doesn't support sparse matrices on the GPU. What I did was to abandon the GPU and run everything on my CPU; but I wonder whether a random lower-dimensional projection (à la compressive sensing) of the sparse matrix would have allowed me to do fast, successful training using the GPU.

Sparsity issues aside, I found that I could get good results from the neural net (in the same ballpark as XGBoost's performance). The best configuration I found used a single hidden ReLU layer with dropout, with Nesterov momentum to help with tuning. I got a significant performance boost when I read [Sutskever's paper on this](http://jmlr.org/proceedings/papers/v28/sutskever13.html), which explained that the momentum method (with very little decay) helps accumulate momentum in directions with small-but-consistent gradient, and is thus effective at quickly moving to a better region of the space, where a more local optimization (with more momentum decay) then helps fine-tune the network.

I played with different priors on the weights (which made limited difference), with SGD (which didn't do as well as momentum did), and [out-of-the-Lasagne-box](http://lasagne.readthedocs.org/en/latest/modules/updates.html) gradient-scaling methods like RMSProp (which caused numeric issues that I didn't debug) and Adagrad (which converged to a reasonably good region as quickly as momentum did, but hit a higher floor). Obviously I have more to learn and see about which of these methods work well when and why. In particular, just as it was reassuring to replicate the common wisdom that adding noisy features doesn't hurt, it was a little unnerving to not be able to replicate the common wisdom that adding more layers, if tuned correctly, should yield better results. It may be that that wisdom was just based on harder problems than this, and there's just not very much interesting structure to be found in this problem. (The fact that XGBoost did best with small trees of only 2-3 nodes seems to weakly support this hypothesis.)

######Ensembles

The day before the end of the contest, I recognized that it was time to build an ensemble, since everyone knows that's how to do well in a machine learning contest. My quickie solution to this was to just run lots of small variations of the neural-net and XGBoost algorithms, produce a lot of model output files (with fitted model predictions on the training set, validation set, and testing set), and then pick a convex combination that does as well as possible on the validation set. This was how I generated my final submission, and it worked reasonably well (loss ~0.575 on the final leaderboard), handily outperforming all my individual predictors.

######Reproducibility

However, while this development process worked well at actually finding good predictors, it doesn't make it easy to reproduce the experiments I ran. I think the biggest improvements I could make here would be to use version control, make sure my scripts generate artifacts and/or experiment results and nothing else (rather than having some of the work done in interactive sessions, and some of the work left hanging around as dead / commented-out code), and tag artifacts and conclusions with the revision numbers that were used to generate them. These seem like a good idea for Real Science™, and really for any situation where I want to be able to look back and easily see what I did - future Kaggle contests are probably in this category.
