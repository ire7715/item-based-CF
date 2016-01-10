import numpy as np
import math
import sys
import time

def cosSimilarity(v1, v2):
	if len(v1) != len(v2):
		return 0

	sop = 0
	v1SquareSum = 0
	v2SquareSum = 0
	for i in range(len(v1)):
		sop += v1[i] * v2[i]
		v1SquareSum += v1[i] * v1[i]
		v2SquareSum += v2[i] * v2[i]

	return sop / math.sqrt(v1SquareSum * v2SquareSum)

def amazonSimilarity(reviews, userIndex, itemIndex, ratingIndex):
	dims = np.amax(reviews, axis=0)
	maxUser = int(dims[userIndex] + 1)
	maxItem = int(dims[itemIndex] + 1)

	i2uMatrix = np.zeros(shape=(maxItem, maxUser))
	itemsOrderedBy = [[] for i in range(maxUser)]
	usersPurchased = [[] for i in range(maxItem)]
	for review in reviews:
		thisUser = int(review[userIndex])
		thisItem = int(review[itemIndex])
		i2uMatrix[thisItem][thisUser] = review[ratingIndex]
		itemsOrderedBy[thisUser].append(thisItem)
		usersPurchased[thisItem].append(thisUser)

	print("preprocessing is done, calculating similarity...")

	similarities = np.zeros(shape=(maxItem, maxItem))
	for i in range(maxItem):
		relatedItems = set()
		for customer in usersPurchased[i]:
			relatedItems |= set(itemsOrderedBy[customer])
		for j in relatedItems:
			if similarities[j][i] != 0:
				similarities[i][j] = similarities[j][i]
			else:
				similarities[i][j] = cosSimilarity(i2uMatrix[i], i2uMatrix[j])
		if i % 50:
			sys.stdout.write("#")

	return similarities

def main():
	if len(sys.argv) < 2:
		print("python amazon_i2i.py training_file [testing_file]")
		return 0
	trainFile = sys.argv[1]
	if len(sys.argv) >= 3:
		testFile = sys.argv[2]

	# user, item, rating, time
	reviews = np.genfromtxt(trainingFile, delimiter="\t")

	print("start at: " + str(time.time()))
	similarities = amazonSimilarity(reviews=reviews, userIndex=0, itemIndex=1, ratingIndex=2)
	reviews = None
	print()
	print("simiarity calculation ended: " + str(time.time()))
	np.save("amazon_similarity", similarities)
	print("file outputed at:" + str(time.time()))

	return 0

if __name__ == "__main__":
	main()
