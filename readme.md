**Last updated on August 24th, 2017** by [Mirith](https://github.com/Mirith)

# Overview

Uses a neural network to predict the language of a single word.  It uses [nltk's](http://www.nltk.org/) [uhdr](http://research.ics.aalto.fi/cog/data/udhr/) (Universal Declaration of Human Rights).  Also [sklearn](http://scikit-learn.org/stable/modules/classes.html) for the machine learning bit, and its [joblib](https://pythonhosted.org/joblib/generated/joblib.load.html#joblib.load) to save and load trained models.  For the graphing, it uses [numpy and scipy](https://docs.scipy.org/doc/) and [matplotlib](https://matplotlib.org/).  



**Note**:  Some bits of code were pulled from online, and I forgot to source one of them.  It's in graph prediction.py and is apparently called [Jensen-Shannon divergence](https://stackoverflow.com/questions/15880133/jensen-shannon-divergence)...  If I ever find the source I'll update stuff.  

# Usage

You'll need python (this was done in python 3).  And a heck of a lot of time.  But at the end you'll have a trained network!  Or you could just look at the pretty pictures on [github](https://github.com/Mirith/word-language-prediction) as well...  If you want the trained analyzer, it's [here](https://drive.google.com/file/d/0ByNf-Gd6Z75pMm9JamZKaVlKa0U/view?usp=sharing).  It's almost a gigabyte and was way past githubs upload limits.  

# Files

## word prediction.py

This file loads the data, splits it and trains a neural network.  It takes a very, very long time.  I have optimized a few parameters with [gridsearch](http://scikit-learn.org/stable/modules/grid_search.html) from I think sklearn?  

## graph prediction.py

Using the trained model from word prediction.py, this ranks languages by their predicted similarity to English and then graphs the most and least similar ones.  

## lowest sim.png and highest sim.png

Results of graph prediction.py.  Just screengrabs of the graphs produced.  

