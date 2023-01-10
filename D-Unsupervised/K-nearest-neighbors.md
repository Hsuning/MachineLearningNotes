#KNNAlgorithms #InstanceBasedLearning 
- #Classification: look at the k closest neighbors, if more neighbors are X, then it is probably a X.
- #Recommendation 
	- find the five users closest to the user you want to recommend
	- how to figure out the similarity between them ?
		- convert each user to a set of coordinates
		- Use the Pythagorean formula to find the distance between points in N-dimensions
		- $\sqrt{(3-4)^2 + (4-3)^2 + (4-5)^2 + (1-1)^2 + (4-5)^2}$
		- Or consine similarity (better) that compares the angles of the two vectors
ConsumerA  | ConsumerB | ConsumerC
------------- | ------------- | -------------
3  | 4 | 2
4  | 3 | 5
4  | 5 | 1
1  | 1 | 3
4  | 5 | 1
	- How to handle the diffrence in rating strategies ? Like A always gives 5 for not too bad movies but B only gives 5 for the best => normalization: look at the average rating for each person and use it to scale their ratings
	- Weighted rating: give more weight to the ratings of the influencers =>  (3 + 4 + 5 / 3 = 4 stars) to (3 + 4 + 5 + 5 + 5 / 5 = 4.4) stars
- #Regression : look at the k closest neighbors, take the average
- #FeatureExtraction: converting an item (like a fruit or a user) into a list of numbers that can be compared.
	- Features that directly correlate to the movies you’re trying to recommend
	- Features that don’t have a bias (for example, if you ask the users to only rate comedy movies, that doesn’t tell you whether they like action movies)
- How to choose K: if you have N users, you should look at sqrt(N) neighbors.