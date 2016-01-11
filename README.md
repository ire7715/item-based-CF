# item-based-CF

An implementation of [Amazon item-based collaborative filtering](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf)  
The dataset is provided by [MovieLens](grouplens.org/datasets/movielens/)

----

Execution:
  python amazon_i2i.py traningFile [testingFile]  
  e.g. python amazon_i2.py ./ml-100k/u1.base ./ml-100k/u1.test

Giving a training file is necessary and it will generate you the following files:

1. item similariy
2. user majored rating matrix
3. adjacency list from items to users
4. adjacency list from users to items

**tip** If there are some trained models have the same prefix to the training file, this program will read it directly without re-calculation.

By assigning a testing file, it will test the testing file from the result of training file.

References:

[1] [Item-to-item Collaborative Filtering Tutorial by Francesco Ricci](https://www.ics.uci.edu/~welling/teaching/CS77Bwinter12/presentations/course_Ricci/13-Item-to-Item-Matrix-CF.pdf)  
[2] [Recommender System Using Collaborative Filtering Algorithm](http://core.ac.uk/download/files/261/10687433.pdf)  
